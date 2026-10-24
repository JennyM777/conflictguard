"""Tests for app.agents.risk_assessment_agent."""

from __future__ import annotations

from app.agents import risk_assessment_agent as raa
from app.models.schemas import ConflictCandidate, EntityExpansion, RiskVerdict

EXPANSION = EntityExpansion(
    primary_entities=["Some Entity"], expanded_aliases={}, deal_type="Litigation", industry=None
)


def _candidate(**overrides) -> ConflictCandidate:
    base = dict(
        matter_id="M-TEST-001",
        client_name="Test Client",
        matter_name="Test Matter",
        role="Represented Company",
        adverse_parties=[],
        similarity_score=0.5,
        match_reason="semantic similarity to deal context score=0.50",
        confidentiality_level="Standard",
        status="Closed",
    )
    base.update(overrides)
    return ConflictCandidate(**base)


def test_offline_no_candidates_is_green():
    verdict = raa._offline_risk_assessment(EXPANSION, [])
    assert verdict.verdict == "GREEN"
    assert verdict.cited_matters == []


def test_offline_adverse_party_match_is_red():
    candidate = _candidate(
        adverse_parties=["Test Client"],
        similarity_score=0.95,
        match_reason="named as adverse party ('Test Client') in a past matter",
        confidentiality_level="High",
        status="Active",
    )
    verdict = raa._offline_risk_assessment(EXPANSION, [candidate])
    assert verdict.verdict == "RED"
    assert "M-TEST-001" in verdict.cited_matters


def test_offline_moderate_similarity_high_confidentiality_is_yellow():
    candidate = _candidate(
        similarity_score=0.7,
        match_reason="substring match on 'Test Client'",
        confidentiality_level="High",
        status="Closed",
    )
    verdict = raa._offline_risk_assessment(EXPANSION, [candidate])
    assert verdict.verdict == "YELLOW"


def test_offline_weak_similarity_is_low_confidence_yellow():
    candidate = _candidate(similarity_score=0.2, confidentiality_level="Standard")
    verdict = raa._offline_risk_assessment(EXPANSION, [candidate])
    assert verdict.verdict == "YELLOW"
    assert "weak/topical overlap" in verdict.reasoning.lower()


def test_run_offline_mode_uses_fallback(fake_llm_client):
    client = fake_llm_client(offline=True)
    verdict, log_entry = raa.run("some input", EXPANSION, [], client)

    assert isinstance(verdict, RiskVerdict)
    assert verdict.verdict == "GREEN"
    assert log_entry.prompt_tokens == 0


def test_run_online_mode_parses_llm_response(fake_llm_client):
    response = {
        "verdict": "RED",
        "confidence": 0.9,
        "summary": "Direct conflict.",
        "reasoning": "Matches a prior adverse matter.",
        "cited_matters": ["M-TEST-001"],
        "recommended_action": "Escalate to conflicts committee.",
    }
    client = fake_llm_client(response_data=response, offline=False)

    verdict, log_entry = raa.run("some input", EXPANSION, [], client)

    assert verdict.verdict == "RED"
    assert verdict.cited_matters == ["M-TEST-001"]
    assert log_entry.prompt_tokens == 100
    assert log_entry.estimated_cost_usd == 0.0021


def test_run_falls_back_when_llm_raises(fake_llm_client):
    client = fake_llm_client(offline=False, raise_exc=ValueError("boom"))

    verdict, log_entry = raa.run("some input", EXPANSION, [], client)

    assert isinstance(verdict, RiskVerdict)
    assert verdict.verdict == "GREEN"  # falls back to rule-based scorer with no candidates
    assert log_entry.prompt_tokens == 0
