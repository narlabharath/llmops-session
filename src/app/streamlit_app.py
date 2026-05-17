"""Streamlit MVP scaffold for the PLAN.md S-13 composition contract.

This entry point wires the existing src layers into the four-panel app shell.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.guardrails import (
    GuardrailDecision,
    GuardrailReport,
    run_input_guardrails,
    run_output_guardrails,
)
from src.llm import LLMClient
from src.observability import LocalTraceStore, Trace, compute_metrics, trace_workflow
from src.rag import ingest
from src.workflow import run_workflow

LOGGER = logging.getLogger(__name__)

PAGE_TITLE = "Learning Program Support Assistant"
PAGE_ICON = "🎓"
DEFAULT_PROVIDER = "anthropic"
PERSIST_DIR_NAME = "data/chroma_app"
INPUT_BLOCK_MESSAGE_PREFIX = "This question was blocked before reaching the assistant"
OUTPUT_BLOCK_MESSAGE_PREFIX = "[Answer redacted for safety"
HISTORY_QUERY_PREVIEW_CHARS = 60

HistoryKind = Literal[
    "answered",
    "refused",
    "escalated",
    "failed",
    "input-blocked",
    "output-blocked",
]


class ChatHistoryEntry(TypedDict):
    query: str
    input_report: GuardrailReport
    result_text: str
    output_report: GuardrailReport | None
    trace: Trace | None
    kind: HistoryKind


def _is_streamlit_runtime() -> bool:
    return bool(getattr(st.runtime, "exists", lambda: False)())


def _configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
    )


def _build_llm_client() -> LLMClient:
    selected_provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    provider = selected_provider or DEFAULT_PROVIDER
    try:
        return LLMClient(provider=provider)
    except Exception as exc:
        LOGGER.warning(
            "Falling back to mock provider for Streamlit startup after %s provider failed: %s",
            provider,
            exc,
        )
        return LLMClient(provider="mock")


@st.cache_resource(show_spinner="Loading program documents...")
def build_pipeline() -> tuple[Path, LLMClient, int]:
    persist_dir = ROOT / PERSIST_DIR_NAME
    ingestion_result = ingest(ROOT / "data", persist_dir)
    llm = _build_llm_client()
    return persist_dir, llm, ingestion_result.documents_loaded


def _bootstrap_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "trace_store" not in st.session_state:
        st.session_state["trace_store"] = LocalTraceStore()


def _initialize_runtime() -> tuple[Path | None, LLMClient | None, int]:
    if not _is_streamlit_runtime():
        return None, None, 0

    _configure_page()
    persist_dir, llm, doc_count = build_pipeline()
    _bootstrap_session_state()
    return persist_dir, llm, doc_count


PERSIST_DIR, LLM, DOC_COUNT = _initialize_runtime()


def render_question_panel() -> None:
    st.subheader("Question")
    query = st.text_input(
        "Your question",
        placeholder="e.g. What is the late-submission policy?",
        key="query_input",
    )
    submitted = st.button("Ask", type="primary")

    if submitted:
        normalized_query = query.strip()
        if not normalized_query:
            st.warning("Enter a question before submitting.")
        elif PERSIST_DIR is None or LLM is None:
            st.error("Pipeline is not ready yet.")
        else:
            with st.spinner("Running guardrails and workflow..."):
                _store_submitted_question(
                    normalized_query,
                    persist_dir=PERSIST_DIR,
                    llm=LLM,
                    trace_store=st.session_state["trace_store"],
                )
            st.rerun()

    history = st.session_state["chat_history"]
    trace_store = st.session_state["trace_store"]
    if history:
        latest_entry = history[-1]
        st.caption(f"Latest stored outcome: `{latest_entry['kind']}`")
    st.caption(
        "Chat history entries: "
        f"{len(history)} | Traces captured: {len(trace_store.all_traces())}"
    )


def render_answer_panel(last_entry: ChatHistoryEntry | None) -> None:
    st.subheader("Answer")
    if last_entry is None:
        st.info("Submit a question to see the latest answer and retrieved context.")
        return

    _render_answer_status_pill(last_entry)

    subtitle = _get_answer_status_subtitle(last_entry)
    if subtitle:
        st.caption(subtitle)

    st.markdown(_get_answer_panel_text(last_entry))

    if last_entry["kind"] == "answered":
        _render_retrieved_context(last_entry["trace"])


def render_trace_panel(trace: Trace | None) -> None:
    st.subheader("Trace")
    with st.expander("🔍 Trace — what actually happened", expanded=False):
        if trace is None:
            history = st.session_state.get("chat_history", [])
            if history and history[-1]["kind"] == "input-blocked":
                st.caption("No trace — request blocked at input guard.")
            else:
                st.caption("Submit a question to capture a workflow trace.")
            return

        _render_trace_summary(trace)
        st.caption("Span narrative")
        if trace.spans:
            for span in trace.spans:
                st.markdown(
                    (
                        f"**[{span.name}]** duration_ms={span.duration_ms:.1f} "
                        f"attrs={_format_span_attributes(span.attributes)}"
                    )
                )
        else:
            st.caption("No spans recorded for the latest request.")

        st.markdown(f"**Workflow steps:** `{_format_workflow_steps(trace.workflow_steps)}`")


def render_history_panel(history: list[ChatHistoryEntry]) -> None:
    st.subheader("History")
    if not history:
        st.info("Session history will appear here after you submit a question.")
    else:
        for index, entry in enumerate(history):
            _render_history_entry(entry, index=index)

    _render_session_metrics(history_count=len(history))


def _submit_question(
    query: str,
    *,
    persist_dir: Path,
    llm: LLMClient,
    trace_store: LocalTraceStore,
) -> ChatHistoryEntry:
    input_report = run_input_guardrails(query, llm=llm)
    if input_report.overall == "block":
        LOGGER.warning(
            "Input guardrails blocked a Streamlit submission. blocked_by=%s query=%r",
            ",".join(input_report.blocked_by),
            query,
        )
        return ChatHistoryEntry(
            query=query,
            input_report=input_report,
            result_text=_build_input_block_message(input_report),
            output_report=None,
            trace=None,
            kind="input-blocked",
        )

    trace = trace_workflow(
        run_workflow,
        question=query,
        persist_dir=persist_dir,
        llm=llm,
        store=trace_store,
    )
    result_text = _extract_result_text_from_trace(trace)
    kind: HistoryKind = _derive_history_kind(trace)
    output_report: GuardrailReport | None = None

    if kind == "answered":
        output_report = run_output_guardrails(result_text, system_prompt=None)
        if output_report.overall == "block":
            kind = "output-blocked"
            result_text = _build_output_block_message(output_report)
            LOGGER.warning(
                "Output guardrails blocked a Streamlit answer. blocked_by=%s query=%r",
                ",".join(output_report.blocked_by),
                query,
            )

    LOGGER.info(
        "Streamlit submission completed. kind=%s steps=%s query=%r",
        kind,
        trace.workflow_steps,
        query,
    )
    return ChatHistoryEntry(
        query=query,
        input_report=input_report,
        result_text=result_text,
        output_report=output_report,
        trace=trace,
        kind=kind,
    )


def _store_submitted_question(
    query: str,
    *,
    persist_dir: Path,
    llm: LLMClient,
    trace_store: LocalTraceStore,
) -> ChatHistoryEntry:
    entry = _submit_question(
        query,
        persist_dir=persist_dir,
        llm=llm,
        trace_store=trace_store,
    )
    history: list[ChatHistoryEntry] = st.session_state["chat_history"]
    history.append(entry)
    st.session_state["query_input"] = query
    return entry


def _build_input_block_message(report: GuardrailReport) -> str:
    reason = _get_first_block_reason(report, fallback="request was blocked")
    return f"{INPUT_BLOCK_MESSAGE_PREFIX} - input guardrail: {reason}."


def _build_output_block_message(report: GuardrailReport) -> str:
    decision = _get_first_block_decision(report)
    detail = "rule"
    if decision is not None:
        pii_type = decision.metadata.get("pii_type")
        detail = pii_type if isinstance(pii_type, str) and pii_type else decision.guardrail
    return f"{OUTPUT_BLOCK_MESSAGE_PREFIX} - output guardrail caught: {detail}]"


def _get_first_block_reason(report: GuardrailReport, *, fallback: str) -> str:
    decision = _get_first_block_decision(report)
    return decision.reason if decision is not None else fallback


def _get_first_block_decision(report: GuardrailReport) -> GuardrailDecision | None:
    for decision in report.decisions:
        if decision.severity == "block":
            return decision
    return None


def _render_answer_status_pill(entry: ChatHistoryEntry) -> None:
    kind = entry["kind"]
    label, background, border, foreground = _get_status_pill_style(kind)
    st.markdown(
        (
            "<div style=\"display:inline-block;padding:0.35rem 0.8rem;"
            "border-radius:999px;font-weight:600;font-size:0.95rem;"
            f"background:{background};border:1px solid {border};color:{foreground};\">"
            f"{label}</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_history_entry(entry: ChatHistoryEntry, *, index: int) -> None:
    summary_column, badge_column, repeat_column = st.columns([6, 2, 1])

    with summary_column:
        st.markdown(f"**{_truncate_history_query(entry['query'])}**")

    with badge_column:
        _render_history_status_pill(entry["kind"])

    with repeat_column:
        if st.button("↻", key=f"repeat_{index}", help="Submit this question again"):
            _repeat_history_query(entry["query"])


def _render_history_status_pill(kind: HistoryKind) -> None:
    label, background, border, foreground = _get_status_pill_style(kind)
    st.markdown(
        (
            "<div style=\"display:inline-block;padding:0.2rem 0.55rem;"
            "border-radius:999px;font-weight:600;font-size:0.78rem;"
            f"background:{background};border:1px solid {border};color:{foreground};\">"
            f"{label}</div>"
        ),
        unsafe_allow_html=True,
    )


def _repeat_history_query(query: str) -> None:
    if PERSIST_DIR is None or LLM is None:
        st.error("Pipeline is not ready yet.")
        return

    with st.spinner("Repeating question..."):
        _store_submitted_question(
            query,
            persist_dir=PERSIST_DIR,
            llm=LLM,
            trace_store=st.session_state["trace_store"],
        )
    st.rerun()


def _truncate_history_query(query: str) -> str:
    if len(query) <= HISTORY_QUERY_PREVIEW_CHARS:
        return query
    return f"{query[:HISTORY_QUERY_PREVIEW_CHARS]}..."


def _get_status_pill_style(kind: HistoryKind) -> tuple[str, str, str, str]:
    styles: dict[HistoryKind, tuple[str, str, str, str]] = {
        "answered": ("🟢 Answered", "#dcfce7", "#86efac", "#166534"),
        "refused": ("🟠 Refused", "#ffedd5", "#fdba74", "#9a3412"),
        "escalated": ("🔴 Escalated", "#fee2e2", "#fca5a5", "#991b1b"),
        "failed": ("⚪ Failed", "#f3f4f6", "#d1d5db", "#374151"),
        "input-blocked": ("🟠 Input blocked", "#ffedd5", "#fdba74", "#9a3412"),
        "output-blocked": ("🟠 Output blocked", "#ffedd5", "#fdba74", "#9a3412"),
    }
    return styles[kind]


def _get_answer_status_subtitle(entry: ChatHistoryEntry) -> str | None:
    kind = entry["kind"]
    if kind == "input-blocked":
        return _format_blocked_by(entry["input_report"])
    if kind == "output-blocked" and entry["output_report"] is not None:
        return _format_blocked_by(entry["output_report"])
    return None


def _format_blocked_by(report: GuardrailReport) -> str | None:
    if not report.blocked_by:
        return None
    return f"Blocked by: {', '.join(report.blocked_by)}"


def _get_answer_panel_text(entry: ChatHistoryEntry) -> str:
    kind = entry["kind"]
    if kind == "input-blocked":
        return _build_input_block_message(entry["input_report"])
    if kind == "output-blocked" and entry["output_report"] is not None:
        return _build_output_block_message(entry["output_report"])
    if entry["result_text"].strip():
        return entry["result_text"]
    if kind == "failed":
        return "The assistant could not complete this request."
    return "No answer text was available for the latest request."


def _render_trace_summary(trace: Trace) -> None:
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(f"**Trace ID:** `{trace.trace_id}`")
        st.markdown(f"**Query:** {trace.query}")
        st.markdown(f"**Category:** `{trace.category or '—'}`")
        st.markdown(
            f"**Backend / model:** `{trace.backend or '—'}` / `{trace.model or '—'}`"
        )
        st.markdown(f"**Latency:** `{trace.latency_ms:.1f} ms`")
        st.markdown(
            (
                "**Prompt / completion / total tokens:** "
                f"`{trace.prompt_tokens}` / `{trace.completion_tokens}` / `{trace.total_tokens}`"
            )
        )

    with right_column:
        st.markdown(f"**Cache status:** `{_format_cache_status(trace.cache_status)}`")
        if not trace.cache_status:
            st.caption("cache_status not propagated by workflow")
        st.markdown(f"**Retrieved chunks:** `{trace.retrieved_count}`")
        st.markdown(f"**Refused:** `{_format_trace_flag(trace.refused)}`")
        st.markdown(f"**Escalated:** `{_format_trace_flag(trace.escalated)}`")
        st.markdown(
            f"**Input guard flag:** `{_format_trace_text(trace.guardrail_input_flag)}`"
        )
        st.markdown(
            f"**Output guard flag:** `{_format_trace_text(trace.guardrail_output_flag)}`"
        )


def _render_retrieved_context(trace: Trace | None) -> None:
    if trace is None:
        return

    retrieved_doc_ids = _extract_retrieved_doc_ids(trace)
    retrieved_count = trace.retrieved_count or len(retrieved_doc_ids)
    with st.expander(f"📄 Retrieved context ({retrieved_count} chunks)", expanded=False):
        if retrieved_doc_ids:
            st.caption("Retrieved document IDs")
            st.markdown(
                "\n".join(f"- `{document_id}`" for document_id in retrieved_doc_ids)
            )
            return

        st.caption("Retrieved context count is available, but document IDs were not exposed.")


def _extract_retrieved_doc_ids(trace: Trace) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for span in trace.spans:
        value = span.attributes.get("retrieved_doc_ids")
        if not isinstance(value, list):
            continue
        for document_id in value:
            if not isinstance(document_id, str) or not document_id or document_id in seen:
                continue
            seen.add(document_id)
            doc_ids.append(document_id)
    return doc_ids


def _format_cache_status(cache_status: str) -> str:
    return cache_status or "—"


def _format_trace_flag(flag: bool) -> str:
    return "Yes" if flag else "No"


def _format_trace_text(value: str) -> str:
    return value or "—"


def _format_workflow_steps(workflow_steps: list[str]) -> str:
    return " → ".join(workflow_steps) if workflow_steps else "—"


def _format_span_attributes(attributes: dict[str, object]) -> str:
    return json.dumps(attributes, default=str, sort_keys=True)


def _derive_history_kind(trace: Trace) -> HistoryKind:
    if trace.refused:
        return "refused"
    if trace.escalated:
        return "escalated"
    return "answered"


def _extract_result_text_from_trace(trace: Trace) -> str:
    for span in reversed(trace.spans):
        output = span.attributes.get("output")
        if isinstance(output, str) and output.strip():
            return output.strip()
    LOGGER.warning("Trace %s did not expose a terminal output string.", trace.trace_id)
    return ""


def main() -> None:
    if not _is_streamlit_runtime():
        return
    if PERSIST_DIR is None or LLM is None:
        st.error("Pipeline failed to initialize.")
        return

    with st.sidebar:
        st.title("⚙️ Config")
        st.markdown(f"**Backend:** `{LLM.provider_name}`")
        st.markdown(f"**Model:** `{LLM.model_name}`")
        st.markdown(f"**Docs indexed:** {DOC_COUNT}")
        st.markdown(f"**Persist dir:** `{PERSIST_DIR.relative_to(ROOT)}`")

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("PLAN.md S-13 MVP scaffold: question, answer, trace, and history.")

    history: list[ChatHistoryEntry] = st.session_state["chat_history"]
    last_entry = history[-1] if history else None

    render_question_panel()
    render_answer_panel(last_entry=last_entry)
    render_trace_panel(trace=last_entry["trace"] if last_entry is not None else None)
    render_history_panel(history=history)


if _is_streamlit_runtime():
    main()
