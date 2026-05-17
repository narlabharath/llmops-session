"""Workflow node implementations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Final

from src.rag.types import RetrievedDocument

from src.llm import LLMClient, load_prompt

from .state import TraceEntry, WorkflowState

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM_PROMPT: Final[str] = (
    "You classify TalentSprint program-assistant questions. "
    "Reply with exactly one category label."
)
_PROMPT_VERSION: Final[str] = "v1"
_ANSWER_SYSTEM_PROMPT_NAME: Final[str] = "program_assistant_system"
_GROUND_ANSWER_PROMPT_NAME: Final[str] = "ground_answer"
_ESCALATION_PROMPT_NAME: Final[str] = "escalate_low_confidence"
_VALID_CLASSIFICATIONS: Final[set[str]] = {
    "policy_question",
    "schedule_question",
    "assignment_question",
    "out_of_scope",
    "injection_attempt",
    "private_request",
}
_FALLBACK_CLASSIFICATION: Final[str] = "out_of_scope"
_REFUSAL_PROMPT_NAMES: Final[dict[str, str]] = {
    "out_of_scope": "refuse_out_of_scope",
    "private_request": "refuse_private_request",
    "injection_attempt": "refuse_injection_attempt",
}


def classify_question(state: WorkflowState, llm: LLMClient | None = None) -> dict[str, object]:
    """Classify the participant question into a workflow routing category."""

    question = state["question"]
    template = load_prompt("classify_question", version=_PROMPT_VERSION)
    prompt = template.format(question=question)
    llm_client = llm or LLMClient(provider="mock", prompt_version=_PROMPT_VERSION)
    result = llm_client.complete(
        prompt,
        system=_CLASSIFIER_SYSTEM_PROMPT,
        prompt_version=_PROMPT_VERSION,
    )
    classification = _parse_classification(result.text)
    trace_entry: TraceEntry = {
        "node": "classify",
        "input": question,
        "output": classification,
        "latency_ms": result.latency_ms,
        "prompt_version": _PROMPT_VERSION,
    }
    trace = [*state.get("trace", []), trace_entry]
    return {"classification": classification, "trace": trace}


def retrieve_documents(
    state: WorkflowState,
    persist_dir: Path,
    k: int = 5,
) -> dict[str, object]:
    """Retrieve supporting documents for the participant question."""

    from src.rag import retrieve

    question = state["question"]
    started = perf_counter()
    retrieved_docs = retrieve(Path(persist_dir), question, k=k)
    latency_ms = (perf_counter() - started) * 1000
    retrieved_doc_ids = _get_retrieved_doc_ids(retrieved_docs)
    trace_entry: TraceEntry = {
        "node": "retrieve",
        "input": question,
        "output": retrieved_doc_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieved_count": len(retrieved_docs),
        "max_score": max((document.score for document in retrieved_docs), default=None),
        "latency_ms": latency_ms,
    }
    trace = [*state.get("trace", []), trace_entry]
    return {"retrieved_docs": retrieved_docs, "trace": trace}


def generate_answer(state: WorkflowState, llm: LLMClient | None = None) -> dict[str, object]:
    """Generate a grounded answer from the retrieved documents."""

    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])
    system_template = load_prompt(_ANSWER_SYSTEM_PROMPT_NAME, version=_PROMPT_VERSION)
    user_template = load_prompt(_GROUND_ANSWER_PROMPT_NAME, version=_PROMPT_VERSION)
    retrieved_context = _format_retrieved_context(retrieved_docs)
    prompt = user_template.format(retrieved_context=retrieved_context, question=question)
    llm_client = llm or LLMClient(provider="mock", prompt_version=_PROMPT_VERSION)
    result = llm_client.complete(
        prompt,
        system=system_template,
        prompt_version=_PROMPT_VERSION,
    )
    answer = _extract_answer_text(result.text)
    retrieved_doc_ids = _get_retrieved_doc_ids(retrieved_docs)
    trace_entry: TraceEntry = {
        "node": "answer",
        "input": question,
        "output": answer,
        "retrieved_doc_ids": retrieved_doc_ids,
        "latency_ms": result.latency_ms,
        "cost_estimate_usd": result.cost_estimate_usd,
        "cache_status": result.cache_status,
        "prompt_version": _PROMPT_VERSION,
    }
    trace = [*state.get("trace", []), trace_entry]
    return {"answer": answer, "trace": trace}


def refuse(state: WorkflowState) -> dict[str, object]:
    """Render a refusal message for unsupported or unsafe requests."""

    question = state["question"]
    classification = state["classification"]
    try:
        prompt_name = _REFUSAL_PROMPT_NAMES[classification]
    except KeyError as exc:
        raise ValueError(f"Unsupported refusal classification: {classification!r}") from exc

    started = perf_counter()
    template = load_prompt(prompt_name, version=_PROMPT_VERSION)
    refusal_reason = template.format(question=question)
    latency_ms = (perf_counter() - started) * 1000
    trace_entry: TraceEntry = {
        "node": "refuse",
        "input": question,
        "output": refusal_reason,
        "classification": classification,
        "latency_ms": latency_ms,
        "prompt_version": _PROMPT_VERSION,
    }
    trace = [*state.get("trace", []), trace_entry]
    return {"refusal_reason": refusal_reason, "trace": trace}


def escalate(state: WorkflowState) -> dict[str, object]:
    """Render an escalation note for low-confidence workflow outcomes."""

    question = state["question"]
    started = perf_counter()
    template = load_prompt(_ESCALATION_PROMPT_NAME, version=_PROMPT_VERSION)
    escalation_reason = template.format(question=question)
    latency_ms = (perf_counter() - started) * 1000
    trace_entry: TraceEntry = {
        "node": "escalate",
        "input": question,
        "output": escalation_reason,
        "latency_ms": latency_ms,
        "prompt_version": _PROMPT_VERSION,
    }
    trace = [*state.get("trace", []), trace_entry]
    return {"escalation_reason": escalation_reason, "trace": trace}


def _parse_classification(raw_output: str) -> str:
    normalized = raw_output.strip().lower()
    if normalized in _VALID_CLASSIFICATIONS:
        return normalized

    logger.warning(
        "Unexpected classification returned by LLM; falling back to %s. raw_output=%r",
        _FALLBACK_CLASSIFICATION,
        raw_output,
    )
    return _FALLBACK_CLASSIFICATION


def _format_retrieved_context(retrieved_docs: list[RetrievedDocument]) -> str:
    if not retrieved_docs:
        return "[source: none, priority: 99, score: 0.000, rank: 0]\n<no retrieved documents>"

    rendered_documents: list[str] = []
    for document in retrieved_docs:
        rendered_documents.append(
            (
                f"[source: {_get_retrieved_doc_id(document)}, "
                f"priority: {document.chunk.metadata.source_priority}, "
                f"score: {document.score:.3f}, rank: {document.rank}]\n"
                f"{document.chunk.text}"
            )
        )
    return "\n\n".join(rendered_documents)


def _extract_answer_text(raw_output: str) -> str:
    normalized = raw_output.strip()
    if not normalized:
        return ""

    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        logger.warning(
            "Answer generation returned non-JSON content; using raw text. raw_output=%r",
            raw_output,
        )
        return normalized

    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    logger.warning(
        "Answer generation returned JSON without a usable answer; using raw text. raw_output=%r",
        raw_output,
    )
    return normalized


def _get_retrieved_doc_ids(retrieved_docs: list[RetrievedDocument]) -> list[str]:
    return [_get_retrieved_doc_id(document) for document in retrieved_docs]


def _get_retrieved_doc_id(document: RetrievedDocument) -> str:
    return (
        document.chunk.metadata.document_id
        or document.chunk.source_path
        or f"rank-{document.rank}"
    )


__all__ = [
    "classify_question",
    "retrieve_documents",
    "generate_answer",
    "refuse",
    "escalate",
]
