from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from src.llm import CompletionResult, LLMCache, LLMClient


def test_mock_provider_returns_completion_result() -> None:
    client = LLMClient(provider="mock")
    result = client.complete("hello")

    assert is_dataclass(result)
    assert isinstance(result, CompletionResult)
    assert isinstance(result.text, str)
    assert result.model == "mock-model-v1"
    assert result.provider == "mock"
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    assert isinstance(result.tokens_in, int)
    assert isinstance(result.tokens_out, int)
    assert isinstance(result.cost_estimate_usd, float)
    assert result.cache_status == "bypass"
    assert isinstance(result.raw, dict)


def test_mock_provider_is_deterministic() -> None:
    client = LLMClient(provider="mock")

    first = client.complete("hello")
    second = client.complete("hello")

    assert first.text == second.text


def test_mock_provider_default_is_no_cache() -> None:
    client = LLMClient(provider="mock")

    first = client.complete("hello")
    second = client.complete("hello")

    assert first.cache_status == "bypass"
    assert second.cache_status == "bypass"


def test_cache_explicit_enable_for_mock() -> None:
    client = LLMClient(provider="mock", cache=True)

    first = client.complete("hello")
    second = client.complete("hello")

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"


def test_cache_invalidation_on_prompt_version_bump() -> None:
    shared_cache = LLMCache()
    first_client = LLMClient(provider="mock", cache=shared_cache, prompt_version="v1")
    second_client = LLMClient(provider="mock", cache=shared_cache, prompt_version="v2")

    first = first_client.complete("hello")
    second = second_client.complete("hello")

    assert first.cache_status == "miss"
    assert second.cache_status == "miss"


def test_cache_invalidation_on_kwargs_change() -> None:
    client = LLMClient(provider="mock", cache=True)

    first = client.complete("hello", temperature=0.1)
    second = client.complete("hello", temperature=0.9)

    assert first.cache_status == "miss"
    assert second.cache_status == "miss"


def test_per_call_cache_bypass() -> None:
    client = LLMClient(provider="mock", cache=True)
    client.complete("hello")

    result = client.complete("hello", cache=False)

    assert result.cache_status == "bypass"


def test_ollama_provider_returns_completion_result_when_transport_is_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.llm.providers.ollama import OllamaProvider

    def fake_complete(
        self: OllamaProvider,
        prompt: str,
        system: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        return {
            "text": "hello from ollama",
            "tokens_in": 4,
            "tokens_out": 3,
            "raw": {
                "prompt": prompt,
                "system": system,
                "kwargs": kwargs,
            },
            "model": self.model,
        }

    monkeypatch.setattr(OllamaProvider, "complete", fake_complete)
    client = LLMClient(provider="ollama")

    result = client.complete("hello", system="Be brief", cache=False)

    assert isinstance(result, CompletionResult)
    assert result.text == "hello from ollama"
    assert result.model == "llama3.1:8b"
    assert result.provider == "ollama"
    assert result.cache_status == "bypass"


def test_unknown_provider_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="banana")
