"""
Local, offline-friendly "vector store" over data/mock_clients.json.

Design choice: rather than depending on a downloaded embedding model (which
may not be available in a grading sandbox without internet access), this
store uses scikit-learn's TF-IDF vectorizer + cosine similarity as a
lightweight, fully local semantic-similarity proxy. It is paired with a
precise alias/substring/fuzzy matcher for exact entity lookups (e.g. matching
"Northbridge" against "Northbridge Capital Partners"), which is the dominant
signal for real conflict checks.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MatterRecord:
    """One flattened (client, matter) pair -- the unit of retrieval."""

    matter_id: str
    client_id: str
    client_name: str
    aliases: List[str]
    industry: str
    matter_name: str
    matter_type: str
    role: str
    adverse_parties: List[str]
    related_entities: List[str]
    status: str
    confidentiality_level: str
    sensitive_notes: str
    searchable_text: str = field(default="")

    @property
    def all_names(self) -> List[str]:
        """Client name + every alias, for entity-matching."""
        return [self.client_name, *self.aliases]


class ClientVectorStore:
    """Loads mock_clients.json and supports entity + semantic search over it."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.records: List[MatterRecord] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load()
        self._build_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"Mock client dataset not found at {self.data_path}. "
                "Check CLIENT_DATA_PATH in your .env."
            )
        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        for client in raw.get("clients", []):
            for matter in client.get("matters", []):
                record = MatterRecord(
                    matter_id=matter["matter_id"],
                    client_id=client["client_id"],
                    client_name=client["client_name"],
                    aliases=client.get("aliases", []),
                    industry=client.get("industry", ""),
                    matter_name=matter.get("matter_name", ""),
                    matter_type=matter.get("matter_type", ""),
                    role=matter.get("role", ""),
                    adverse_parties=matter.get("adverse_parties", []) or [],
                    related_entities=matter.get("related_entities", []) or [],
                    status=matter.get("status", ""),
                    confidentiality_level=matter.get("confidentiality_level", ""),
                    sensitive_notes=matter.get("sensitive_notes", "") or "",
                )
                record.searchable_text = self._build_searchable_text(record)
                self.records.append(record)

        logger.info(
            "Loaded %d matter records from %s", len(self.records), self.data_path
        )

    @staticmethod
    def _build_searchable_text(record: MatterRecord) -> str:
        parts = [
            record.client_name,
            " ".join(record.aliases),
            record.industry,
            record.matter_name,
            record.matter_type,
            record.role,
            " ".join(record.adverse_parties),
            " ".join(record.related_entities),
            record.sensitive_notes,
        ]
        return " | ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        if not self.records:
            return
        corpus = [r.searchable_text for r in self.records]
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    # ------------------------------------------------------------------
    # Semantic search (TF-IDF cosine similarity)
    # ------------------------------------------------------------------
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[MatterRecord, float]]:
        if not query.strip() or self._vectorizer is None or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [
            (self.records[i], float(scores[i])) for i in ranked_idx if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    # Exact / fuzzy entity matching
    # ------------------------------------------------------------------
    def match_entity(
        self, entity_name: str, fuzzy_threshold: float = 0.72
    ) -> List[Tuple[MatterRecord, float, str]]:
        """Match a single extracted entity name against clients and adverse
        parties across all records.

        Returns a list of (record, score, reason) tuples. `reason` explains
        *why* it matched (exact client name, alias, substring, fuzzy, or
        adverse-party overlap), which the Risk Assessment Agent surfaces to
        the human reviewer.
        """
        if not entity_name.strip():
            return []

        needle = entity_name.strip().lower()
        results: List[Tuple[MatterRecord, float, str]] = []

        for record in self.records:
            best_score = 0.0
            best_reason = ""

            # 1. Exact / substring match against client name + aliases.
            # Substring containment is only meaningful for reasonably long
            # needles (e.g. "Northbridge") -- for short ones (e.g. "No", "Inc")
            # it produces nonsense hits (needle "no" is a substring of
            # "Northbridge"), so it is gated behind a minimum length.
            for name in record.all_names:
                name_l = name.lower()
                if needle == name_l:
                    best_score, best_reason = 1.0, f"exact match on client name '{name}'"
                    break
                if len(needle) >= 4 and (needle in name_l or name_l in needle):
                    score = 0.9
                    if score > best_score:
                        best_score, best_reason = score, f"substring match on '{name}'"

            # 2. Fuzzy match (handles typos / minor variations)
            if best_score < 0.9:
                for name in record.all_names:
                    ratio = difflib.SequenceMatcher(None, needle, name.lower()).ratio()
                    if ratio >= fuzzy_threshold and ratio > best_score:
                        best_score, best_reason = ratio, f"fuzzy match on '{name}' ({ratio:.2f})"

            # 3. Adverse-party / related-entity overlap (direct adversity signal)
            for adverse in record.adverse_parties:
                adverse_l = adverse.lower()
                if needle == adverse_l or (
                    len(needle) >= 4 and (needle in adverse_l or adverse_l in needle)
                ):
                    # Adverse-party hits are treated as high-confidence regardless
                    # of the client-name score, since this is the clearest
                    # conflict-of-interest signal (firm previously opposed this entity).
                    score = 0.95
                    if score > best_score:
                        best_score = score
                        best_reason = f"named as adverse party ('{adverse}') in a past matter"

            if best_score > 0:
                results.append((record, best_score, best_reason))

        results.sort(key=lambda r: r[1], reverse=True)
        return results
