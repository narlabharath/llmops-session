from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.app import streamlit_app
from src.guardrails import GuardrailDecision, GuardrailReport
from src.observability import Span, Trace

_RENDER_ANSWER_PANEL_SCRIPT = """
import streamlit as st
from src.app import streamlit_app

streamlit_app.render_answer_panel(st.session_state.get("entry"))
"""


def _run_answer_panel(
    entry: streamlit_app.ChatHistoryEntry | None,
) -> AppTest:
    at = AppTest.from_string(_RENDER_ANSWER_PANEL_SCRIPT)
    at.session_state["entry"] = entry
    at.run()
    return at


def _build_entry(
    *,
    kind: streamlit_app.HistoryKind,
    result_text: str,
    input_report: GuardrailReport | None = None,
    output_report: GuardrailReport | None = None,
    trace: Trace | None = None,
) -> streamlit_app.ChatHistoryEntry:
    return {
        "query": "Example question",
        "input_report": input_report or GuardrailReport(decisions=[], overall="pass"),
        "result_text": result_text,
        "output_report": output_report,
        "trace": trace,
        "kind": kind,
    }


def _build_block_report(
    *,
    guardrail: str,
    reason: str,
    blocked_by: list[str],
    metadata: dict[str, object] | None = None,
) -> GuardrailReport:
    return GuardrailReport(
        decisions=[
            GuardrailDecision(
                passed=False,
                guardrail=guardrail,
                severity="block",
                reason=reason,
                metadata=metadata or {},
            )
        ],
        overall="block",
        blocked_by=blocked_by,
    )


def test_render_answer_panel_prompts_before_submission() -> None:
    at = _run_answer_panel(None)

    assert at.subheader[0].value == "Answer"
    assert at.info[0].value == "Submit a question to see the latest answer and retrieved context."
    assert len(at.expander) == 0


def test_render_answer_panel_answered_shows_status_and_retrieved_context() -> None:
    entry = _build_entry(
        kind="answered",
        result_text="Late submissions are allowed for two grace days.",
        output_report=GuardrailReport(decisions=[], overall="pass"),
        trace=Trace(
            trace_id="trace-answered",
            query="What is the late submission policy?",
            start_ms=0.0,
            retrieved_count=2,
            spans=[
                Span(
                    name="retrieve",
                    start_ms=0.0,
                    end_ms=1.0,
                    attributes={"retrieved_doc_ids": ["program_policy", "calendar"]},
                ),
                Span(
                    name="answer",
                    start_ms=1.0,
                    end_ms=2.0,
                    attributes={"retrieved_doc_ids": ["program_policy", "calendar"]},
                ),
            ],
        ),
    )

    at = _run_answer_panel(entry)

    assert any("Answered" in markdown.value for markdown in at.markdown)
    assert any(
        "Late submissions are allowed for two grace days." in markdown.value
        for markdown in at.markdown
    )
    assert len(at.expander) == 1
    assert "Retrieved context (2 chunks)" in at.expander[0].label
    assert at.expander[0].proto.expanded is False
    assert at.expander[0].caption[0].value == "Retrieved document IDs"
    assert at.expander[0].markdown[0].value == "- `program_policy`\n- `calendar`"


@pytest.mark.parametrize(
    ("entry", "expected_label", "expected_body", "expected_caption"),
    [
        pytest.param(
            _build_entry(
                kind="refused",
                result_text="That question is outside the scope of this assistant.",
            ),
            "Refused",
            "outside the scope",
            None,
            id="refused",
        ),
        pytest.param(
            _build_entry(
                kind="escalated",
                result_text="I am routing this request to a human reviewer.",
            ),
            "Escalated",
            "routing this request to a human reviewer",
            None,
            id="escalated",
        ),
        pytest.param(
            _build_entry(
                kind="failed",
                result_text="",
            ),
            "Failed",
            "The assistant could not complete this request.",
            None,
            id="failed",
        ),
        pytest.param(
            _build_entry(
                kind="input-blocked",
                result_text="",
                input_report=_build_block_report(
                    guardrail="prompt_injection",
                    reason="input attempts to override prior instructions",
                    blocked_by=["prompt_injection"],
                ),
            ),
            "Input blocked",
            "input guardrail: input attempts to override prior instructions.",
            "Blocked by: prompt_injection",
            id="input-blocked",
        ),
        pytest.param(
            _build_entry(
                kind="output-blocked",
                result_text="",
                output_report=_build_block_report(
                    guardrail="pii_output",
                    reason="output contains an email address",
                    blocked_by=["email_address"],
                    metadata={"pii_type": "email_address"},
                ),
            ),
            "Output blocked",
            "[Answer redacted for safety - output guardrail caught: email_address]",
            "Blocked by: email_address",
            id="output-blocked",
        ),
    ],
)
def test_render_answer_panel_outcome_states(
    entry: streamlit_app.ChatHistoryEntry,
    expected_label: str,
    expected_body: str,
    expected_caption: str | None,
) -> None:
    at = _run_answer_panel(entry)
    captions = [caption.value for caption in at.caption]

    assert any(expected_label in markdown.value for markdown in at.markdown)
    assert any(expected_body in markdown.value for markdown in at.markdown)
    if expected_caption is None:
        assert captions == []
    else:
        assert expected_caption in captions
    assert len(at.expander) == 0
