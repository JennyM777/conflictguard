"""Tests for app.agents.conflict_search_agent."""

from __future__ import annotations

from app.agents import conflict_search_agent
from app.models.schemas import EntityExpansion


def test_run_returns_candidates_and_zero_cost_log(vector_store, offline_settings):
    expansion = EntityExpansion(
        primary_entities=["Northbridge Capital Partners"],
        expanded_aliases={"Northbridge Capital Partners": ["Northbridge"]},
        deal_type="Litigation",
        industry="Private Equity",
    )

    candidates, log_entry = conflict_search_agent.run(expansion, vector_store, offline_settings)

    assert candidates
    assert any(c.matter_id == "M-2021-014" for c in candidates)
    assert log_entry.agent_name == conflict_search_agent.AGENT_NAME
    # This agent never calls an LLM, so it must always report zero token cost.
    assert log_entry.prompt_tokens == 0
    assert log_entry.completion_tokens == 0
    assert log_entry.estimated_cost_usd == 0.0


def test_run_with_no_matching_entities_returns_empty_candidates(vector_store, offline_settings):
    expansion = EntityExpansion(
        primary_entities=["Nonexistent Startup XYZ"],
        expanded_aliases={},
        deal_type="Other",
        industry=None,
    )

    candidates, log_entry = conflict_search_agent.run(expansion, vector_store, offline_settings)

    assert candidates == []
    assert "0 candidate" in log_entry.output_summary
