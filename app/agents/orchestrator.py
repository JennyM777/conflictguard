"""
ConflictGuard Orchestrator.

Coordinates the full multi-agent pipeline:

  1. Safety Guardrail        -- scans raw input for prompt-injection attempts
  2. Entity Expansion Agent  -- extracts & expands entities from the input
  3. Conflict Vector DB Agent -- retrieves candidate conflicting matters
  4. Risk Assessment Agent   -- synthesizes a RED/YELLOW/GREEN verdict

The orchestrator owns the long-lived resources (LLM client, vector store) so
the Streamlit UI can construct it once per session and reuse it across runs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.agents import conflict_search_agent, entity_expansion_agent, guardrail, risk_assessment_agent
from app.core.config import Settings, get_settings
from app.core.llm_client import LLMClient
from app.models.schemas import (
    AgentLogEntry,
    ConflictReport,
    EntityExpansion,
    RiskVerdict,
)
from app.rag.vector_store import ClientVectorStore
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

GUARDRAIL_AGENT_NAME = "Safety Guardrail"


class ConflictGuardOrchestrator:
    """Owns the LLM client and vector store; runs the end-to-end pipeline."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm_client = LLMClient(self.settings)
        self.store = ClientVectorStore(self.settings.client_data_path)

    def run_check(self, user_input: str) -> ConflictReport:
        agent_logs: list[AgentLogEntry] = []
        start_time = datetime.now(timezone.utc).isoformat()

        # --- 1. Safety Guardrail (always runs first, on raw input only) ---
        guardrail_result = guardrail.check_prompt_injection(user_input)
        agent_logs.append(
            AgentLogEntry(
                agent_name=GUARDRAIL_AGENT_NAME,
                timestamp=start_time,
                input_summary=user_input[:200],
                output_summary=(
                    f"flagged={guardrail_result.flagged} risk={guardrail_result.risk_level} "
                    f"reasons={guardrail_result.reasons}"
                )[:300],
            )
        )

        if guardrail_result.flagged and guardrail_result.risk_level == "high":
            logger.warning("Guardrail blocked input: %s", guardrail_result.reasons)
            blocked_verdict = RiskVerdict(
                verdict="BLOCKED",
                confidence=1.0,
                summary="Input blocked by the Safety Guardrail before analysis.",
                reasoning=(
                    "The submitted text appears to attempt to manipulate the "
                    "agent pipeline (e.g. instructing it to ignore its "
                    "instructions or force a favorable verdict). "
                    f"Reasons: {'; '.join(guardrail_result.reasons)}"
                ),
                cited_matters=[],
                recommended_action=(
                    "Have a human compliance officer review this submission "
                    "directly; do not resubmit without removing the "
                    "suspicious instructions."
                ),
            )
            return ConflictReport(
                verdict=blocked_verdict,
                entity_expansion=EntityExpansion(),
                candidates=[],
                guardrail=guardrail_result,
                agent_logs=agent_logs,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_estimated_cost_usd=0.0,
                provider=self.llm_client.provider,
            )

        # --- 2. Entity Expansion Agent ---------------------------------
        entity_expansion, log_entry = entity_expansion_agent.run(user_input, self.llm_client)
        agent_logs.append(log_entry)

        # --- 3. Conflict Vector DB Agent --------------------------------
        candidates, log_entry = conflict_search_agent.run(
            entity_expansion, self.store, self.settings
        )
        agent_logs.append(log_entry)

        # --- 4. Risk Assessment Agent ------------------------------------
        verdict, log_entry = risk_assessment_agent.run(
            user_input, entity_expansion, candidates, self.llm_client
        )
        agent_logs.append(log_entry)

        usage = self.llm_client.total_usage()

        return ConflictReport(
            verdict=verdict,
            entity_expansion=entity_expansion,
            candidates=candidates,
            guardrail=guardrail_result,
            agent_logs=agent_logs,
            total_prompt_tokens=usage.total_prompt_tokens,
            total_completion_tokens=usage.total_completion_tokens,
            total_estimated_cost_usd=usage.total_estimated_cost_usd,
            provider=self.llm_client.provider,
        )
