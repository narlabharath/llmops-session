from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.observability import Span, Trace

_RENDER_TRACE_PANEL_SCRIPT = """
import streamlit as st
from src.app import streamlit_app

streamlit_app.render_trace_panel(st.session_state.get("trace"))
"""


def _run_trace_panel(
    trace: Trace | None,
    *,
    history: list[dict[str, object]] | None = None,
) -> AppTest:
    at = AppTest.from_string(_RENDER_TRACE_PANEL_SCRIPT)
    at.session_state["trace"] = trace
    if history is not None:
        at.session_state["chat_history"] = history
    at.run()
    return at


def _build_trace(*, cache_status: str) -> Trace:
    return Trace(
        trace_id="trace-answered",
        query="What is the late submission policy?",
        start_ms=0.0,
        category="policy_question",
        backend="scripted",
        model="trace-test-model",
        prompt_tokens=11,
        completion_tokens=9,
        total_tokens=20,
        latency_ms=18.4,
        retrieved_count=2,
        cache_status=cache_status,
        refused=False,
        escalated=False,
        workflow_steps=["classify", "retrieve", "answer"],
        spans=[
            Span(
                name="classify",
                start_ms=0.0,
                end_ms=9.5,
                attributes={"output": "policy_question", "route_to": "retrieve"},
            ),
            Span(
                name="retrieve",
                start_ms=9.5,
                end_ms=12.0,
                attributes={
                    "retrieved_count": 2,
                    "retrieved_doc_ids": ["program_policy", "calendar"],
                    "route_to": "answer",
                },
            ),
            Span(
                name="answer",
                start_ms=12.0,
                end_ms=18.4,
                attributes={
                    "cache_status": cache_status,
                    "output": "Late submissions are allowed for two grace days.",
                },
            ),
        ],
    )


def test_render_trace_panel_prompts_before_submission() -> None:
    at = _run_trace_panel(None)

    assert at.subheader[0].value == "Trace"
    assert len(at.expander) == 1
    assert at.expander[0].label == "🔍 Trace — what actually happened"
    assert at.expander[0].proto.expanded is False
    assert at.expander[0].caption[0].value == "Submit a question to capture a workflow trace."


def test_render_trace_panel_shows_input_blocked_message_without_trace() -> None:
    at = _run_trace_panel(None, history=[{"kind": "input-blocked"}])

    assert at.expander[0].caption[0].value == "No trace — request blocked at input guard."


def test_render_trace_panel_renders_summary_fields_and_span_narrative() -> None:
    at = _run_trace_panel(_build_trace(cache_status="hit"))
    markdown_values = [markdown.value for markdown in at.expander[0].markdown]
    captions = [caption.value for caption in at.expander[0].caption]

    assert any("**Trace ID:** `trace-answered`" in value for value in markdown_values)
    assert any(
        "**Query:** What is the late submission policy?" in value
        for value in markdown_values
    )
    assert any("**Category:** `policy_question`" in value for value in markdown_values)
    assert any(
        "**Backend / model:** `scripted` / `trace-test-model`" in value
        for value in markdown_values
    )
    assert any("**Latency:** `18.4 ms`" in value for value in markdown_values)
    assert any(
        "**Prompt / completion / total tokens:** `11` / `9` / `20`" in value
        for value in markdown_values
    )
    assert any("**Cache status:** `hit`" in value for value in markdown_values)
    assert any("**Retrieved chunks:** `2`" in value for value in markdown_values)
    assert any("**Refused:** `No`" in value for value in markdown_values)
    assert any("**Escalated:** `No`" in value for value in markdown_values)
    assert any("**Input guard flag:** `—`" in value for value in markdown_values)
    assert any("**Output guard flag:** `—`" in value for value in markdown_values)
    assert "Span narrative" in captions
    assert any(
        "**[classify]** duration_ms=9.5" in value
        and '"route_to": "retrieve"' in value
        for value in markdown_values
    )
    assert any(
        "**[answer]** duration_ms=6.4" in value and '"cache_status": "hit"' in value
        for value in markdown_values
    )
    assert any(
        "**Workflow steps:** `classify → retrieve → answer`" in value
        for value in markdown_values
    )


def test_render_trace_panel_marks_missing_cache_status() -> None:
    at = _run_trace_panel(_build_trace(cache_status=""))
    markdown_values = [markdown.value for markdown in at.expander[0].markdown]
    captions = [caption.value for caption in at.expander[0].caption]

    assert any("**Cache status:** `—`" in value for value in markdown_values)
    assert "cache_status not propagated by workflow" in captions
