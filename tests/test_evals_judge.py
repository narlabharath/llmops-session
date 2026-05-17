from __future__ import annotations

import logging
from pathlib import Path

from src.evals import JudgeVerdict, judge_groundedness
from src.llm import load_prompt
from tests.conftest_workflow import ScriptedLLMClient


def test_judge_groundedness_parses_valid_json_verdict() -> None:
    question = "When are office hours?"
    answer = "Office hours are on Tuesdays at 5 PM."
    retrieved_context = "Office hours are held on Tuesdays at 5 PM in the lab."
    llm = ScriptedLLMClient(
        rules=[
            (
                "Question:",
                (
                    '{"grounded": true, "confidence": 0.91, '
                    '"reason": "Every factual claim is supported by the documents."}'
                ),
            )
        ]
    )

    verdict = judge_groundedness(question, answer, retrieved_context, judge_llm=llm)

    assert verdict == JudgeVerdict(
        grounded=True,
        confidence=0.91,
        reason="Every factual claim is supported by the documents.",
        model="scripted",
    )
    assert llm.calls[0]["prompt"] == load_prompt("judge_groundedness", version="v1").format(
        question=question,
        answer=answer,
        retrieved_context=retrieved_context,
    )
    assert llm.calls[0]["kwargs"] == {"prompt_version": "v1"}


def test_judge_groundedness_returns_fallback_verdict_on_invalid_json(caplog) -> None:
    garbage = (
        "this is not json and it should force the judge fallback because it cannot be parsed "
        "safely"
    )
    llm = ScriptedLLMClient(rules=[("Question:", garbage)])

    with caplog.at_level(logging.WARNING):
        verdict = judge_groundedness(
            "Can I submit late?",
            "Yes, late submissions are always accepted for seven days.",
            "Late submissions are accepted only within 48 hours with a note.",
            judge_llm=llm,
        )

    assert verdict == JudgeVerdict(
        grounded=False,
        confidence=0.0,
        reason=f"judge returned invalid JSON: {garbage[:80]}",
        model="scripted",
    )
    assert "invalid JSON" in caplog.text


def test_judge_prompt_file_loads_from_disk() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llm"
        / "prompts"
        / "judge_groundedness.v1.md"
    )

    assert load_prompt("judge_groundedness", version="v1") == prompt_path.read_text(
        encoding="utf-8"
    )
