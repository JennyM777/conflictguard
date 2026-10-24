"""Shared pytest fixtures for the ConflictGuard test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402
from app.rag.vector_store import ClientVectorStore  # noqa: E402

MOCK_DATA_PATH = PROJECT_ROOT / "data" / "mock_clients.json"


@pytest.fixture(scope="session")
def vector_store() -> ClientVectorStore:
    """Real ClientVectorStore built from the actual mock dataset shipped with the repo."""
    return ClientVectorStore(MOCK_DATA_PATH)


@pytest.fixture
def offline_settings(tmp_path) -> Settings:
    """A Settings object guaranteed to resolve to the 'offline' provider,
    regardless of what API keys happen to be set in the real shell
    environment running the tests."""
    return Settings(
        openai_api_key=None,
        portkey_api_key=None,
        portkey_virtual_key=None,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        client_data_path=MOCK_DATA_PATH,
        vector_store_dir=tmp_path / "vector_store",
        conflict_similarity_threshold=0.75,
        log_level="INFO",
        env_file_used=None,
    )


class FakeLLMClient:
    """Test double matching the duck-typed interface agents expect from
    app.core.llm_client.LLMClient, without making any real network calls."""

    def __init__(
        self,
        response_data: Optional[Dict[str, Any]] = None,
        offline: bool = False,
        raise_exc: Optional[Exception] = None,
    ):
        self.is_offline = offline
        self.provider = "offline" if offline else "fake"
        self._response_data = response_data
        self._raise_exc = raise_exc
        self.calls = []
        self._last = None

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
        if self._raise_exc is not None:
            raise self._raise_exc
        self.calls.append((system_prompt, user_prompt, temperature))
        self._last = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, estimated_cost_usd=0.0021
        )
        return self._response_data or {}

    def last_call(self):
        return self._last


@pytest.fixture
def fake_llm_client():
    """Factory fixture: fake_llm_client(response_data=..., offline=..., raise_exc=...)"""
    return FakeLLMClient
