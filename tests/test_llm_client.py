"""Tests for app.core.llm_client (provider resolution, cost estimation, offline guard)."""

from __future__ import annotations

import pytest

from app.core.llm_client import LLMClient, estimate_cost_usd


def test_offline_provider_when_no_keys(offline_settings):
    client = LLMClient(offline_settings)
    assert client.provider == "offline"
    assert client.is_offline is True


def test_chat_json_raises_when_offline(offline_settings):
    client = LLMClient(offline_settings)
    with pytest.raises(RuntimeError):
        client.chat_json("system", "user")


def test_estimate_cost_usd_known_model():
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_estimate_cost_usd_unknown_model_uses_default_pricing():
    cost = estimate_cost_usd("some-unlisted-model", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.50 + 1.50)


def test_total_usage_aggregates_empty_log(offline_settings):
    client = LLMClient(offline_settings)
    summary = client.total_usage()
    assert summary.total_prompt_tokens == 0
    assert summary.total_completion_tokens == 0
    assert summary.total_estimated_cost_usd == 0.0
    assert summary.call_count == 0
