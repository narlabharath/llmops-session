from __future__ import annotations

import logging

import pytest

from src.llm import CompletionResult, LLMClient
from src.workflow.nodes import classify_question


def _result(text: str, latency_ms: float = 12.5) -> CompletionResult:
    return CompletionResult(
        text=text,
        model="mock-model-v1",
        provider="mock",
        latency_ms=latency_ms,
        tokens_in=10,
        tokens_out=2,
        cost_estimate_usd=0.0,
        cache_status="bypass",
        raw=None,
    )


class StubLLM:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        self.calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})
        return _result(self._response_text)


@pytest.mark.parametrize(
    "category",
    [
        "policy_question",
        "schedule_question",
        "assignment_question",
        "out_of_scope",
        "injection_attempt",
        "private_request",
    ],
)
def test_classify_question_parses_each_valid_category(category: str) -> None:
    llm = StubLLM(category)

    updates = classify_question({"question": "Test question"}, llm=llm)  # type: ignore[arg-type]

    assert updates["classification"] == category
    assert llm.calls[0]["system"] is not None
    assert "Test question" in str(llm.calls[0]["prompt"])

    trace = updates["trace"]
    assert isinstance(trace, list)
    assert trace[-1] == {
        "node": "classify",
        "input": "Test question",
        "output": category,
        "latency_ms": 12.5,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cache_status": "bypass",
        "cost_estimate_usd": 0.0,
        "prompt_version": "v1",
    }


def test_classify_question_falls_back_to_out_of_scope_for_invalid_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = StubLLM("schedule_question because the user asked about timing")

    with caplog.at_level(logging.WARNING):
        updates = classify_question({"question": "When is the lab?"}, llm=llm)  # type: ignore[arg-type]

    assert updates["classification"] == "out_of_scope"
    assert "Unexpected classification returned by LLM" in caplog.text
    assert "schedule_question because the user asked about timing" in caplog.text


def test_classify_question_with_mock_provider_runs_end_to_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = LLMClient(provider="mock")

    with caplog.at_level(logging.WARNING):
        updates = classify_question({"question": "What is the late submission policy?"}, llm=llm)

    assert updates["classification"] == "out_of_scope"
    trace = updates["trace"]
    assert isinstance(trace, list)
    assert trace[-1]["node"] == "classify"
    assert trace[-1]["input"] == "What is the late submission policy?"
    assert trace[-1]["output"] == "out_of_scope"
    assert isinstance(trace[-1]["latency_ms"], float)
    assert trace[-1]["prompt_tokens"] > 0
    assert trace[-1]["completion_tokens"] > 0
    assert trace[-1]["cache_status"] == "bypass"
    assert trace[-1]["prompt_version"] == "v1"
    assert "Unexpected classification returned by LLM" in caplog.text
