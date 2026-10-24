"""
Conflict Vector DB Agent.

Deterministic retrieval agent -- no LLM call, so it costs zero tokens. It
queries the local mock client database (via app.rag.retriever) for matters
that overlap with, or were adverse to, the entities named in the new deal.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Tuple

from app.core.config import Settings
from app.models.schemas import AgentLogEntry, ConflictCandidate, EntityExpansion
from app.rag.retriever import search_conflicts
from app.rag.vector_store import ClientVectorStore
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Conflict Vector DB Agent"


def run(
    entity_expansion: EntityExpansion,
    store: ClientVectorStore,
    settings: Settings,
) -> Tuple[List[ConflictCandidate], AgentLogEntry]:
    """Execute the Conflict Vector DB Agent and return (candidates, log entry)."""

    start = time.perf_counter()

    # CONFLICT_SIMILARITY_THRESHOLD in .env is tuned for cosine-similarity-style
    # embedding scores (0-1, typically >0.7 for a strong match). TF-IDF cosine
    # scores run much lower in practice, so we cap the *semantic* threshold here;
    # entity/alias/adverse-party matches (the higher-precision signal) are
    # unaffected and always pass through at their own confidence score.
    candidates = search_conflicts(
        entity_expansion,
        store,
        semantic_threshold=min(settings.conflict_similarity_threshold, 0.3),
    )

    duration_ms = (time.perf_counter() - start) * 1000
    log_entry = AgentLogEntry(
        agent_name=AGENT_NAME,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_summary=f"entities={entity_expansion.primary_entities}",
        output_summary=(
            f"{len(candidates)} candidate matter(s): "
            + ", ".join(f"{c.matter_id} ({c.similarity_score})" for c in candidates[:5])
        )[:300],
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=0.0,
        duration_ms=duration_ms,
    )
    return candidates, log_entry
