"""
Configuration loading for ConflictGuard.

Implements a *dual-path* .env loader: grading setups sometimes run this app
from a directory one level below where the real .env lives (e.g. a shared
course workspace), so we check the project's own directory first and, if no
.env is found there, fall back to the parent directory before giving up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# app/core/config.py -> parents[0]=app/core, [1]=app, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment() -> Optional[Path]:
    """Load environment variables from a .env file, dual-path style.

    Search order:
      1. <project_root>/.env          (normal case)
      2. <project_root's parent>/.env  (grading/course-workspace fallback)

    Returns the path that was actually loaded, or None if neither exists (in
    which case real environment variables, e.g. from the shell or CI
    secrets, are used as-is).
    """
    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
    ]
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(dotenv_path=candidate, override=True)
            return candidate
    return None


# Load as soon as this module is imported, so every other module can just
# read from os.environ / get_settings() without worrying about ordering.
_ENV_FILE_USED = load_environment()


@dataclass(frozen=True)
class Settings:
    """Typed, read-only view of the application's configuration."""

    openai_api_key: Optional[str]
    portkey_api_key: Optional[str]
    portkey_virtual_key: Optional[str]
    llm_model: str
    embedding_model: str
    client_data_path: Path
    vector_store_dir: Path
    conflict_similarity_threshold: float
    log_level: str
    env_file_used: Optional[Path]

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.portkey_api_key or self.openai_api_key)

    @property
    def active_provider(self) -> str:
        """Which backend will actually be used, given the keys present."""
        if self.portkey_api_key:
            return "portkey"
        if self.openai_api_key:
            return "openai"
        return "offline"


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the Settings object from the current environment."""

    client_data_path = Path(
        os.environ.get("CLIENT_DATA_PATH", "data/mock_clients.json")
    )
    if not client_data_path.is_absolute():
        client_data_path = PROJECT_ROOT / client_data_path

    vector_store_dir = Path(os.environ.get("VECTOR_STORE_DIR", "data/vector_store"))
    if not vector_store_dir.is_absolute():
        vector_store_dir = PROJECT_ROOT / vector_store_dir

    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        portkey_api_key=os.environ.get("PORTKEY_API_KEY") or None,
        portkey_virtual_key=os.environ.get("PORTKEY_VIRTUAL_KEY") or None,
        llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        client_data_path=client_data_path,
        vector_store_dir=vector_store_dir,
        conflict_similarity_threshold=float(
            os.environ.get("CONFLICT_SIMILARITY_THRESHOLD", "0.75")
        ),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        env_file_used=_ENV_FILE_USED,
    )
