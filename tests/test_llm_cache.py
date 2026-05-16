from __future__ import annotations

from src.llm import CompletionResult, LLMCache


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


def test_cache_key_is_stable() -> None:
    cache = LLMCache()

    first = cache.make_key("hello", "system", "model-a", "v1", {"temperature": 0.1})
    second = cache.make_key("hello", "system", "model-a", "v1", {"temperature": 0.1})

    assert first == second


def test_cache_key_changes_with_each_component() -> None:
    cache = LLMCache()
    baseline = cache.make_key("hello", "system", "model-a", "v1", {"temperature": 0.1})
    variants = {
        cache.make_key("hello!", "system", "model-a", "v1", {"temperature": 0.1}),
        cache.make_key("hello", "system-2", "model-a", "v1", {"temperature": 0.1}),
        cache.make_key("hello", "system", "model-b", "v1", {"temperature": 0.1}),
        cache.make_key("hello", "system", "model-a", "v2", {"temperature": 0.1}),
        cache.make_key("hello", "system", "model-a", "v1", {"temperature": 0.9}),
    }

    assert baseline not in variants
    assert len(variants) == 5


def test_cache_hit_returns_stored_value() -> None:
    cache = LLMCache()
    key = cache.make_key("hello", None, "model-a", "v1", {})
    value = _result("hello")

    cache.set(key, value)

    assert cache.get(key) == value


def test_cache_miss_returns_none() -> None:
    cache = LLMCache()

    assert cache.get("missing") is None


def test_cache_stats() -> None:
    cache = LLMCache()
    first_key = cache.make_key("hello", None, "model-a", "v1", {})
    second_key = cache.make_key("world", None, "model-a", "v1", {})

    cache.set(first_key, _result("first"))
    cache.set(second_key, _result("second"))
    cache.get(first_key)
    cache.get("missing")

    assert cache.stats()["size"] == 2
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_fifo_eviction() -> None:
    cache = LLMCache(maxsize=2)
    first_key = cache.make_key("first", None, "model-a", "v1", {})
    second_key = cache.make_key("second", None, "model-a", "v1", {})
    third_key = cache.make_key("third", None, "model-a", "v1", {})

    cache.set(first_key, _result("first"))
    cache.set(second_key, _result("second"))
    cache.set(third_key, _result("third"))

    assert cache.get(first_key) is None
    assert cache.get(second_key) is not None
    assert cache.get(third_key) is not None
