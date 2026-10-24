"""Tests for app.agents.entity_expansion_agent."""

from __future__ import annotations

from app.agents import entity_expansion_agent as eea
from app.models.schemas import EntityExpansion


def test_offline_extract_entities_finds_capitalized_names():
    text = "Northbridge Capital Partners wants to acquire Summit Logistics Inc. in a new deal."
    entities = eea._offline_extract_entities(text)
    assert "Northbridge Capital Partners" in entities
    assert any("Summit Logistics" in e for e in entities)


def test_offline_guess_deal_type_detects_acquisition_keywords():
    assert eea._offline_guess_deal_type("We want to acquire a competitor") == "M&A Advisory"
    assert eea._offline_guess_deal_type("We are being sued by a former partner") == "Litigation"
    assert eea._offline_guess_deal_type("Raising a Series B funding round") == "Financing"
    assert eea._offline_guess_deal_type("Filing for Chapter 11 restructuring") == "Restructuring"
    assert eea._offline_guess_deal_type("Nothing special happening here") == "Other"


def test_offline_expand_aliases_strips_suffix_and_adds_first_word():
    aliases = eea._offline_expand_aliases(["Northbridge Capital Partners"])
    assert set(aliases["Northbridge Capital Partners"]) == {"Northbridge", "Northbridge Capital"}


def test_offline_expand_aliases_single_word_entity_has_no_aliases():
    aliases = eea._offline_expand_aliases(["Acme"])
    assert aliases.get("Acme", []) == []


def test_run_uses_offline_fallback_when_client_is_offline(fake_llm_client):
    client = fake_llm_client(offline=True)
    result, log_entry = eea.run("Acquiring Summit Logistics Inc. soon.", client)

    assert isinstance(result, EntityExpansion)
    assert "Summit Logistics Inc." in result.primary_entities or any(
        "Summit Logistics" in e for e in result.primary_entities
    )
    assert log_entry.agent_name == eea.AGENT_NAME
    assert log_entry.prompt_tokens == 0
    assert log_entry.completion_tokens == 0
    assert client.calls == []  # never called the (offline) LLM


def test_run_uses_llm_response_when_online(fake_llm_client):
    response = {
        "primary_entities": ["Acme Corp"],
        "expanded_aliases": {"Acme Corp": ["Acme"]},
        "deal_type": "M&A Advisory",
        "industry": "Manufacturing",
        "raw_notes": None,
    }
    client = fake_llm_client(response_data=response, offline=False)

    result, log_entry = eea.run("Acme Corp wants to merge.", client)

    assert result.primary_entities == ["Acme Corp"]
    assert result.deal_type == "M&A Advisory"
    assert len(client.calls) == 1
    assert log_entry.prompt_tokens == 100
    assert log_entry.completion_tokens == 50
    assert log_entry.estimated_cost_usd == 0.0021


def test_run_falls_back_to_heuristic_if_llm_call_fails(fake_llm_client):
    client = fake_llm_client(offline=False, raise_exc=RuntimeError("API down"))

    result, log_entry = eea.run("Northbridge Capital Partners deal.", client)

    assert isinstance(result, EntityExpansion)
    assert log_entry.prompt_tokens == 0  # fallback path logs no token usage
