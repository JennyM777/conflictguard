"""
Unified LLM client for ConflictGuard.

Resolves a single "provider" at startup based on which API keys are present:

  - PORTKEY_API_KEY set  -> route calls through the Portkey AI gateway
  - OPENAI_API_KEY set   -> call OpenAI directly
  - neither set          -> "offline" mode; callers must use their own
                            rule-based fallback (see each agent's module)

Also tracks per-call token usage and an estimated USD cost ("tokenomics"),
which the Streamlit UI surfaces in real time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional

from app.core.config import Settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Approximate public pricing, USD per 1,000,000 tokens. Used only to produce a
# rough, illustrative cost estimate for the Tokenomics panel -- not billing-grade.
PRICING_PER_MILLION_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}
DEFAULT_PRICING = {"input": 0.50, "output": 1.50}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING_PER_MILLION_TOKENS.get(model, DEFAULT_PRICING)
    return (prompt_tokens / 1_000_000) * pricing["input"] + (
        completion_tokens / 1_000_000
    ) * pricing["output"]


@dataclass
class LLMCallRecord:
    """A record of one completed LLM call, for the running usage log."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    duration_ms: float


@dataclass
class UsageSummary:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    call_count: int = 0


class LLMClient:
    """Thin wrapper that normalizes OpenAI vs. Portkey vs. offline mode."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model
        self.provider = settings.active_provider  # "portkey" | "openai" | "offline"
        self._client = None
        self.usage_log: List[LLMCallRecord] = []

        if self.provider == "portkey":
            self._client = self._build_portkey_client()
        elif self.provider == "openai":
            self._client = self._build_openai_client()
        else:
            logger.warning(
                "No OPENAI_API_KEY or PORTKEY_API_KEY found - LLMClient running in "
                "OFFLINE mode. Agents will use rule-based fallbacks."
            )

    def _build_portkey_client(self):
        from portkey_ai import Portkey

        kwargs = {"api_key": self.settings.portkey_api_key}
        if self.settings.portkey_virtual_key:
            kwargs["virtual_key"] = self.settings.portkey_virtual_key
        return Portkey(**kwargs)

    def _build_openai_client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.settings.openai_api_key)

    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> dict:
        """Call the chat model expecting a JSON object response.

        Raises RuntimeError if no provider is configured (offline mode) -
        callers are expected to check `is_offline` first and use a fallback.
        """
        if self.is_offline or self._client is None:
            raise RuntimeError(
                "LLMClient is offline (no API key configured); caller must use "
                "a rule-based fallback instead of chat_json()."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            # Some backend models served through a gateway (e.g. reasoning
            # models proxied via Portkey/Azure) reject a custom temperature
            # and only support the default. Retry once without it rather
            # than failing the whole agent run over a sampling parameter.
            if "temperature" in str(exc).lower():
                logger.warning("Model rejected custom temperature, retrying with default: %s", exc)
                response = self._client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                raise
        duration_ms = (time.perf_counter() - start) * 1000

        content = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = estimate_cost_usd(self.model, prompt_tokens, completion_tokens)

        self.usage_log.append(
            LLMCallRecord(
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=cost,
                duration_ms=duration_ms,
            )
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error("Model did not return valid JSON: %s", content[:500])
            raise

    def last_call(self) -> Optional[LLMCallRecord]:
        return self.usage_log[-1] if self.usage_log else None

    def total_usage(self) -> UsageSummary:
        summary = UsageSummary()
        for record in self.usage_log:
            summary.total_prompt_tokens += record.prompt_tokens
            summary.total_completion_tokens += record.completion_tokens
            summary.total_estimated_cost_usd += record.estimated_cost_usd
            summary.call_count += 1
        return summary
