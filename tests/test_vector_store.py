"""Tests for app.rag.vector_store.ClientVectorStore."""

from __future__ import annotations


def test_loads_all_matter_records(vector_store):
    # The mock dataset ships with 10 clients / 11 matters total.
    assert len(vector_store.records) == 11
    matter_ids = {r.matter_id for r in vector_store.records}
    assert "M-2021-014" in matter_ids
    assert "M-2023-002" in matter_ids


def test_exact_client_name_match(vector_store):
    results = vector_store.match_entity("Northbridge Capital Partners")
    assert results, "expected at least one match"
    top_record, top_score, top_reason = results[0]
    assert top_record.matter_id == "M-2021-014"
    assert top_score == 1.0
    assert "exact match" in top_reason


def test_substring_alias_match(vector_store):
    # "Northbridge" is a substring of the full client name "Northbridge Capital Partners".
    results = vector_store.match_entity("Northbridge")
    assert any(r.matter_id == "M-2021-014" for r, _, _ in results)


def test_adverse_party_match_is_high_confidence(vector_store):
    # Summit Logistics Inc. is both a client (M-2019-006) and a named
    # adverse party in Northbridge's acquisition matter (M-2021-014).
    results = vector_store.match_entity("Summit Logistics Inc.")
    matched_ids = {r.matter_id for r, _, _ in results}
    assert "M-2021-014" in matched_ids
    assert "M-2019-006" in matched_ids

    adverse_hit = next(r for r, score, reason in results if r.matter_id == "M-2021-014")
    reason = next(reason for r, score, reason in results if r.matter_id == "M-2021-014")
    assert "adverse party" in reason


def test_no_match_for_unrelated_entity(vector_store):
    results = vector_store.match_entity("Totally Unrelated Nonexistent Corp")
    assert results == []


def test_fuzzy_match_handles_minor_variation(vector_store):
    # Minor misspelling of "Vantage Pharma Holdings".
    results = vector_store.match_entity("Vantage Pharam Holdings")
    matched_ids = {r.matter_id for r, _, _ in results}
    assert "M-2022-031" in matched_ids or "M-2023-002" in matched_ids


def test_semantic_search_returns_relevant_results(vector_store):
    results = vector_store.semantic_search("logistics acquisition financing", top_k=5)
    assert results
    matched_ids = {r.matter_id for r, _ in results}
    # Should surface the logistics-related matters over completely unrelated ones.
    assert matched_ids & {"M-2021-014", "M-2019-006", "M-2021-014-FIN"}


def test_semantic_search_empty_query_returns_nothing(vector_store):
    assert vector_store.semantic_search("") == []
