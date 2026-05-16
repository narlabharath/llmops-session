from __future__ import annotations

from src.llm import CompletionResult, LLMCache, LLMClient


def _result(text: str = "cached") -> CompletionResult:
    return CompletionResult(
        text=text,
        model="mock-model-v1",
        provider="mock",
        latency_ms=1.0,
        tokens_in=1,
        tokens_out=1,
        cost_estimate_usd=0.0,
        cache_status="miss",
        raw={"source": "test"},
    )


def test_cache_event_callback_reports_hit_and_miss() -> None:
    events: list[str] = []
    cache = LLMCache(on_cache_event=events.append)
    key = cache.make_key("hello", None, "model-a", "v1", {})
    value = _result("hello")

    assert cache.get(key) is None

    cache.set(key, value)

    assert cache.get(key) == value
    assert events == ["miss", "hit"]


def test_client_cache_event_callback_reports_miss_hit_and_bypass() -> None:
    events: list[str] = []
    client = LLMClient(provider="mock", cache=LLMCache(on_cache_event=events.append))

    first = client.complete("hello")
    second = client.complete("hello")
    third = client.complete("hello", cache=False)

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert third.cache_status == "bypass"
    assert events == ["miss", "hit", "bypass"]
