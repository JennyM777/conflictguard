"""Pydantic data models shared across ConflictGuard's agents and UI."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

VerdictLiteral = Literal["RED", "YELLOW", "GREEN", "BLOCKED"]
GuardrailRiskLiteral = Literal["none", "low", "high"]


class EntityExpansion(BaseModel):
    """Output of the Entity Expansion Agent."""

    primary_entities: List[str] = Field(
        default_factory=list,
        description="Organizations/individuals named in the new-matter description.",
    )
    expanded_aliases: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of primary entity -> plausible alias/short-name variants.",
    )
    deal_type: Optional[str] = Field(
        default=None, description="e.g. M&A Advisory, Litigation, Financing, JV"
    )
    industry: Optional[str] = Field(default=None)
    raw_notes: Optional[str] = Field(
        default=None, description="Any other agent commentary about the extraction."
    )


class ConflictCandidate(BaseModel):
    """A single historical matter retrieved as a possible conflict."""

    matter_id: str
    client_name: str
    matter_name: str
    role: str
    adverse_parties: List[str] = Field(default_factory=list)
    similarity_score: float
    match_reason: str
    confidentiality_level: str
    status: str


class RiskVerdict(BaseModel):
    """Output of the Risk Assessment Agent."""

    verdict: VerdictLiteral
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    reasoning: str
    cited_matters: List[str] = Field(default_factory=list)
    recommended_action: str


class GuardrailResult(BaseModel):
    """Output of the Safety Guardrail check."""

    flagged: bool
    risk_level: GuardrailRiskLiteral
    reasons: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class AgentLogEntry(BaseModel):
    """One row in the step-by-step agent execution log shown in the UI."""

    agent_name: str
    timestamp: str
    input_summary: str
    output_summary: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    duration_ms: float = 0.0


class ConflictReport(BaseModel):
    """The final, structured result returned by the orchestrator."""

    verdict: RiskVerdict
    entity_expansion: EntityExpansion
    candidates: List[ConflictCandidate] = Field(default_factory=list)
    guardrail: GuardrailResult
    agent_logs: List[AgentLogEntry] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    provider: str = "offline"
