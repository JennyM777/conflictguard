# ConflictGuard

**Law Firm & M&A Due Diligence Multi-Agent Conflict of Interest Detector**

A Tier 3 AI agent application built for *AI Foundations for Managers*. ConflictGuard
simulates how a law firm could use a coordinated team of LLM agents, backed by a local
retrieval-augmented generation (RAG) index over historical client/matter records, to
flag potential conflicts of interest before accepting a new engagement.

## Why "Tier 3"?

- **Multi-agent orchestration**: an Orchestrator coordinates an Intake Agent, a
  Search/RAG Agent, and a Risk Analysis Agent rather than a single prompt.
- **Tool use / retrieval**: agents call a local vector store (RAG) over past matters
  instead of relying on model memory alone.
- **Structured, auditable output**: findings are returned as structured conflict
  reports (not free text), suitable for human attorney review.

## Project Structure

```
Final Project/
├── app/
│   ├── main.py                  # Streamlit entry point (UI)
│   ├── agents/
│   │   ├── intake_agent.py      # Parses new-matter intake details into a structured query
│   │   ├── search_agent.py      # Queries the RAG store for potentially conflicting past matters
│   │   ├── risk_analysis_agent.py # Reasons over retrieved matters to classify conflict severity
│   │   └── orchestrator.py      # Coordinates the agent pipeline end-to-end
│   ├── core/
│   │   ├── config.py            # Loads .env settings into a typed Settings object
│   │   └── llm_client.py        # Unified LLM client (Portkey gateway or direct OpenAI)
│   ├── rag/
│   │   ├── vector_store.py      # Builds/persists the local Chroma vector index
│   │   └── retriever.py         # Similarity search over indexed client/matter records
│   ├── models/
│   │   └── schemas.py           # Pydantic models: ClientRecord, Matter, ConflictFinding, etc.
│   └── utils/
│       └── logging_utils.py     # Shared logging configuration
├── data/
│   ├── mock_clients.json        # Mock historical client/matter dataset for local RAG
│   └── vector_store/            # Generated Chroma index (gitignored, rebuilt on demand)
├── tests/                       # Unit tests for agents and retrieval logic
├── docs/                        # Architecture notes, write-up material
├── scripts/                     # One-off utility scripts (e.g. index builder)
├── .env.example                 # Template for required environment variables
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY or PORTKEY_API_KEY
```

## Running (once the app is built out)

```bash
streamlit run app/main.py
```

## Status

This commit establishes the project scaffolding: directory structure, dependency
manifest, environment configuration template, and the mock dataset used to seed the
local RAG index. Agent logic, the vector store, and the Streamlit UI are implemented
in subsequent steps.
