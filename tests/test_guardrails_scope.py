from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.guardrails import check_scope
from src.llm import load_prompt
from tests.conftest_workflow import ScriptedLLMClient


@pytest.mark.parametrize(
    ("response", "expected_severity", "expected_passed", "expected_confidence", "reason"),
    [
        pytest.param(
            (
                '{"in_scope": true, "confidence": 0.93, '
                '"reason": "The question is directly about the program material."}'
            ),
            "info",
            True,
            0.93,
            "The question is directly about the program material.",
            id="in-scope-pass",
        ),
        pytest.param(
            (
                '{"in_scope": false, "confidence": 0.81, '
                '"reason": "The request is unrelated to the course and its tooling."}'
            ),
            "block",
            False,
            0.81,
            "The request is unrelated to the course and its tooling.",
            id="out-of-scope-high-confidence-block",
        ),
        pytest.param(
            (
                '{"in_scope": false, "confidence": 0.41, '
                '"reason": "The request seems tangential rather than clearly course-related."}'
            ),
            "warn",
            False,
            0.41,
            "The request seems tangential rather than clearly course-related.",
            id="out-of-scope-low-confidence-warn",
        ),
    ],
)
def test_check_scope_maps_llm_verdicts_to_guardrail_decision(
    response: str,
    expected_severity: str,
    expected_passed: bool,
    expected_confidence: float,
    reason: str,
) -> None:
    question = "Can you help me with this request?"
    program_summary = "This program teaches practical AI engineering in Python."
    llm = ScriptedLLMClient(rules=[("Question:", response)])

    decision = check_scope(question, llm=llm, program_summary=program_summary)

    assert decision.guardrail == "scope.llm_judge"
    assert decision.severity == expected_severity
    assert decision.passed is expected_passed
    assert decision.reason == reason
    assert decision.metadata["confidence"] == pytest.approx(expected_confidence)
    assert llm.calls[0]["prompt"] == load_prompt("scope_check", version="v1").format(
        question=question,
        program_summary=program_summary,
    )
    assert llm.calls[0]["kwargs"] == {"prompt_version": "v1"}


def test_check_scope_warns_and_fails_open_on_invalid_json(caplog: pytest.LogCaptureFixture) -> None:
    garbage = "this is not valid json from the scope judge"
    llm = ScriptedLLMClient(rules=[("Question:", garbage)])

    with caplog.at_level(logging.WARNING):
        decision = check_scope("Can you plan my vacation?", llm=llm)

    assert decision.guardrail == "scope.llm_judge"
    assert decision.severity == "warn"
    assert decision.passed is True
    assert decision.reason == "scope judge returned invalid JSON"
    assert decision.metadata["raw_output_preview"] == garbage
    assert "invalid JSON" in caplog.text


def test_scope_prompt_file_loads_from_disk() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llm"
        / "prompts"
        / "scope_check.v1.md"
    )

    assert load_prompt("scope_check", version="v1") == prompt_path.read_text(encoding="utf-8")
