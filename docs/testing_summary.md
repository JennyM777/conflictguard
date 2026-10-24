# ConflictGuard — Test Coverage Summary

ConflictGuard includes a 48-test automated pytest suite (`tests/`) covering every layer of the
multi-agent pipeline. All tests run fully offline and deterministically — they do not require an
API key or network access, so they pass the same way in any grading environment.

## What's covered

| Layer | File | What it verifies |
|---|---|---|
| Local RAG store | `test_vector_store.py` | Loading the mock client/matter dataset; exact, substring, and fuzzy name matching; adverse-party detection; TF-IDF semantic search |
| Retriever | `test_retriever.py` | Merging entity-match + semantic-search results, de-duplication by matter ID, descending sort, empty-result handling |
| Safety Guardrail | `test_guardrail.py` | Parametrized detection of prompt-injection phrasing ("ignore previous instructions", "respond with GREEN", etc.), high-risk vs. low-risk classification, clean-input pass-through |
| Entity Expansion Agent | `test_entity_expansion_agent.py` | Offline heuristic entity/alias extraction and deal-type detection, LLM response parsing, graceful fallback if the LLM call fails |
| Conflict Vector DB Agent | `test_conflict_search_agent.py` | Correct candidate retrieval and that this agent always reports zero token cost (no LLM call) |
| Risk Assessment Agent | `test_risk_assessment_agent.py` | Rule-based RED/YELLOW/GREEN scoring logic for each conflict scenario, LLM response parsing, fallback on LLM failure |
| LLM Client | `test_llm_client.py` | Provider resolution (offline vs. configured), cost-estimation math, offline guard rails |
| Orchestrator (integration) | `test_orchestrator.py` | End-to-end pipeline behavior for a blocked (injected) input, a direct-adversity input, and a clean unrelated input |

## Testing approach

- **Fixtures** (`tests/conftest.py`) build a real `ClientVectorStore` from the actual
  `mock_clients.json`, and provide an `offline_settings` fixture that forces the "offline"
  provider regardless of what API keys exist in the real environment — keeping tests fast,
  free, and reproducible.
- **`FakeLLMClient`** is a lightweight test double standing in for the real LLM client, used to
  verify each agent's "online" code path (parsing a model response, tracking tokens) and its
  exception-handling fallback, without making real API calls.
- **Integration tests** exercise the full 4-agent orchestrator pipeline end-to-end for the three
  main verdict outcomes (BLOCKED, RED/YELLOW, GREEN).

## A bug the tests caught

While writing the integration test for a "clean, unrelated matter" case, the suite surfaced a
real defect: the offline entity extractor could pull out short junk words (e.g. "No" from "No
other parties are involved"), and the vector store's substring matcher then falsely matched "no"
as a substring of "**No**rthbridge" — producing an incorrect YELLOW verdict for an unrelated
matter. This was fixed by (1) requiring a minimum length of 4 characters before doing substring
containment matching in `vector_store.match_entity`, and (2) filtering short/filler single-word
tokens out of the offline entity extractor. This is a concrete example of how the automated test
suite improved the system's real-world reliability, not just its code coverage numbers.

## Running the tests

```bash
source .venv/bin/activate
python -m pytest -q
```

Current result: **48 passed**.
