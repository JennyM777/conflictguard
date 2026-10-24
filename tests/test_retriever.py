"""Tests for app.rag.retriever.search_conflicts."""

from __future__ import annotations

from app.models.schemas import EntityExpansion
from app.rag.retriever import search_conflicts


def test_search_conflicts_finds_entity_match(vector_store):
    expansion = EntityExpansion(
        primary_entities=["Northbridge Capital Partners"],
        expanded_aliases={"Northbridge Capital Partners": ["Northbridge"]},
        deal_type="Litigation",
        industry="Private Equity",
    )
    candidates = search_conflicts(expansion, vector_store)

    assert candidates, "expected at least one candidate"
    assert any(c.matter_id == "M-2021-014" for c in candidates)
    top = candidates[0]
    assert top.similarity_score == max(c.similarity_score for c in candidates)


def test_search_conflicts_deduplicates_by_matter_id(vector_store):
    # Both the primary entity and its alias should surface the same matter;
    # it must only appear once in the results.
    expansion = EntityExpansion(
        primary_entities=["Northbridge Capital Partners", "Northbridge"],
        expanded_aliases={},
        deal_type="M&A Advisory",
        industry="Private Equity",
    )
    candidates = search_conflicts(expansion, vector_store)
    matter_ids = [c.matter_id for c in candidates]
    assert len(matter_ids) == len(set(matter_ids))


def test_search_conflicts_no_entities_no_context_returns_empty(vector_store):
    expansion = EntityExpansion(primary_entities=[], expanded_aliases={}, deal_type=None, industry=None)
    candidates = search_conflicts(expansion, vector_store)
    assert candidates == []


def test_search_conflicts_unrelated_entity_returns_empty(vector_store):
    expansion = EntityExpansion(
        primary_entities=["Totally Fictional Startup Nobody Has Heard Of"],
        expanded_aliases={},
        deal_type="Financing",
        industry="Space Tourism",
    )
    candidates = search_conflicts(expansion, vector_store)
    assert candidates == []


def test_search_conflicts_results_sorted_descending(vector_store):
    expansion = EntityExpansion(
        primary_entities=["Vantage Pharma Holdings", "Clearwater Biosciences"],
        expanded_aliases={},
        deal_type="M&A Advisory",
        industry="Pharmaceuticals",
    )
    candidates = search_conflicts(expansion, vector_store)
    scores = [c.similarity_score for c in candidates]
    assert scores == sorted(scores, reverse=True)
