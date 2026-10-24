"""
High-level retrieval API used by the Conflict Vector DB Agent.

Combines the precise entity/alias/adverse-party matcher with the broader
TF-IDF semantic search over deal context (industry, deal type, free text),
de-duplicates by matter, and returns ranked ConflictCandidate objects.
"""

from __future__ import annotations

from typing import Dict, List

from app.models.schemas import ConflictCandidate, EntityExpansion
from app.rag.vector_store import ClientVectorStore, MatterRecord
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _record_to_candidate(
    record: MatterRecord, score: float, reason: str
) -> ConflictCandidate:
    return ConflictCandidate(
        matter_id=record.matter_id,
        client_name=record.client_name,
        matter_name=record.matter_name,
        role=record.role,
        adverse_parties=record.adverse_parties,
        similarity_score=round(min(score, 1.0), 3),
        match_reason=reason,
        confidentiality_level=record.confidentiality_level,
        status=record.status,
    )


def search_conflicts(
    entity_expansion: EntityExpansion,
    store: ClientVectorStore,
    semantic_threshold: float = 0.15,
    entity_fuzzy_threshold: float = 0.72,
    top_k_semantic: int = 8,
) -> List[ConflictCandidate]:
    """Search the local client/matter store for potential conflicts.

    Two retrieval paths are merged:
      1. Entity matching: every extracted entity (and its expanded aliases)
         is checked against client names, aliases, and adverse parties.
         This is high-precision and catches the clearest conflicts.
      2. Semantic search: a TF-IDF similarity search over the full deal
         context (industry + deal type + entity names) catches topically
         related matters that entity matching alone would miss.

    Results are de-duplicated by matter_id, keeping the highest score and
    the most informative match reason for each.
    """

    best_by_matter: Dict[str, ConflictCandidate] = {}

    def _consider(candidate: ConflictCandidate) -> None:
        existing = best_by_matter.get(candidate.matter_id)
        if existing is None or candidate.similarity_score > existing.similarity_score:
            best_by_matter[candidate.matter_id] = candidate

    # --- 1. Entity / alias / adverse-party matching ---------------------
    all_entity_names: List[str] = list(entity_expansion.primary_entities)
    for aliases in entity_expansion.expanded_aliases.values():
        all_entity_names.extend(aliases)

    for name in dict.fromkeys(n for n in all_entity_names if n and n.strip()):
        for record, score, reason in store.match_entity(
            name, fuzzy_threshold=entity_fuzzy_threshold
        ):
            _consider(_record_to_candidate(record, score, reason))

    # --- 2. Semantic search over the broader deal context ----------------
    context_parts = [
        " ".join(entity_expansion.primary_entities),
        entity_expansion.deal_type or "",
        entity_expansion.industry or "",
    ]
    context_query = " ".join(p for p in context_parts if p).strip()

    if context_query:
        for record, score in store.semantic_search(context_query, top_k=top_k_semantic):
            if score >= semantic_threshold:
                reason = (
                    f"semantic similarity to deal context (industry/deal type) "
                    f"score={score:.2f}"
                )
                _consider(_record_to_candidate(record, score, reason))

    candidates = sorted(
        best_by_matter.values(), key=lambda c: c.similarity_score, reverse=True
    )
    logger.info(
        "Conflict search found %d candidate matter(s) for entities=%s",
        len(candidates),
        entity_expansion.primary_entities,
    )
    return candidates
