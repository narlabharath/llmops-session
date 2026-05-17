from __future__ import annotations

from pathlib import Path

import pytest

from src.llm import load_prompt
from src.workflow.nodes import escalate, refuse


@pytest.mark.parametrize(
    ("classification", "prompt_name"),
    [
        ("out_of_scope", "refuse_out_of_scope"),
        ("private_request", "refuse_private_request"),
        ("injection_attempt", "refuse_injection_attempt"),
    ],
)
def test_refuse_uses_expected_template_for_each_terminal_classification(
    classification: str,
    prompt_name: str,
) -> None:
    question = "Can I see another student's grade?"

    updates = refuse({"question": question, "classification": classification})

    expected = load_prompt(prompt_name, version="v1").format(question=question)
    assert updates["refusal_reason"] == expected
    assert question in str(updates["refusal_reason"])

    trace = updates["trace"]
    assert isinstance(trace, list)
    assert trace[-1]["node"] == "refuse"
    assert trace[-1]["input"] == question
    assert trace[-1]["output"] == expected
    assert trace[-1]["classification"] == classification
    assert trace[-1]["prompt_version"] == "v1"
    assert isinstance(trace[-1]["latency_ms"], float)


def test_escalate_renders_template_and_appends_trace() -> None:
    question = "What is the rule for a case not covered in the documents?"
    existing_trace = [{"node": "retrieve", "output": ["program_policy"]}]

    updates = escalate({"question": question, "trace": existing_trace})  # type: ignore[arg-type]

    expected = load_prompt("escalate_low_confidence", version="v1").format(question=question)
    assert updates["escalation_reason"] == expected
    assert updates["escalation_reason"]
    assert question in str(updates["escalation_reason"])

    trace = updates["trace"]
    assert isinstance(trace, list)
    assert trace[:-1] == existing_trace
    assert trace[-1]["node"] == "escalate"
    assert trace[-1]["input"] == question
    assert trace[-1]["output"] == expected
    assert trace[-1]["prompt_version"] == "v1"
    assert isinstance(trace[-1]["latency_ms"], float)


@pytest.mark.parametrize(
    "prompt_name",
    [
        "refuse_out_of_scope",
        "refuse_private_request",
        "refuse_injection_attempt",
        "escalate_low_confidence",
    ],
)
def test_refuse_and_escalate_prompt_files_exist(prompt_name: str) -> None:
    prompt_path = (
        Path(__file__).resolve().parents[1] / "src" / "llm" / "prompts" / f"{prompt_name}.v1.md"
    )

    assert prompt_path.exists()
    assert load_prompt(prompt_name, version="v1") == prompt_path.read_text(encoding="utf-8")
