"""Local in-memory trace storage helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from pathlib import Path
import time
import uuid
from typing import Any

from src.llm import LLMClient
from src.workflow.types import WorkflowResult

from .types import Span, Trace

LOGGER = logging.getLogger(__name__)
_LLM_TOUCHING_SPANS = {"classify", "answer", "scope_guard", "judge"}

_TRACE_DATAFRAME_COLUMNS = [
    "trace_id",
    "query",
    "category",
    "backend",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "retrieved_count",
    "cache_status",
    "refused",
    "escalated",
    "guardrail_input_flag",
    "guardrail_output_flag",
    "workflow_steps",
]
_store: LocalTraceStore | None = None


def _truncate_query(query: str) -> str:
    if len(query) <= 60:
        return query
    return f"{query[:60]}..."


def _trace_to_row(trace: Trace) -> dict[str, object]:
    return {
        "trace_id": trace.trace_id,
        "query": _truncate_query(trace.query),
        "category": trace.category,
        "backend": trace.backend,
        "model": trace.model,
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "total_tokens": trace.total_tokens,
        "latency_ms": trace.latency_ms,
        "retrieved_count": trace.retrieved_count,
        "cache_status": trace.cache_status,
        "refused": trace.refused,
        "escalated": trace.escalated,
        "guardrail_input_flag": trace.guardrail_input_flag,
        "guardrail_output_flag": trace.guardrail_output_flag,
        "workflow_steps": " \u2192 ".join(trace.workflow_steps),
    }


class LocalTraceStore:
    """In-memory trace collection for notebook and test use."""

    def __init__(self) -> None:
        self._traces: list[Trace] = []

    def new_trace(self, query: str) -> Trace:
        trace = Trace(
            trace_id=uuid.uuid4().hex[:8],
            query=query,
            start_ms=time.perf_counter() * 1000,
        )
        self._traces.append(trace)
        return trace

    def all_traces(self) -> list[Trace]:
        return list(self._traces)

    def to_dataframe(self) -> Any:
        import pandas as pd

        rows = [_trace_to_row(trace) for trace in self._traces]
        return pd.DataFrame(rows, columns=_TRACE_DATAFRAME_COLUMNS)

    def clear(self) -> None:
        self._traces.clear()


def get_store() -> LocalTraceStore:
    global _store

    if _store is None:
        _store = LocalTraceStore()
    return _store


def trace_workflow(
    run_workflow_fn: Callable[..., WorkflowResult],
    question: str,
    persist_dir: Path,
    llm: LLMClient | None = None,
    store: LocalTraceStore | None = None,
    **workflow_kwargs: Any,
) -> Trace:
    trace_store = store or get_store()
    trace = trace_store.new_trace(question)
    result = run_workflow_fn(
        question,
        Path(persist_dir),
        llm=llm,
        **workflow_kwargs,
    )
    _finish_trace_from_result(trace, result, llm)
    return trace


def _finish_trace_from_result(
    trace: Trace,
    result: WorkflowResult,
    llm: object | None,
) -> None:
    llm_results = _extract_llm_completion_results(llm)
    llm_result_index = 0
    synthesized_start_ms = trace.start_ms
    cache_statuses: list[str] = []
    input_flags: list[str] = []
    output_flags: list[str] = []

    trace.spans = []
    for entry in result.trace:
        span_name, name_key = _get_span_name(entry)
        llm_result: object | None = None
        if span_name in _LLM_TOUCHING_SPANS and llm_result_index < len(llm_results):
            llm_result = llm_results[llm_result_index]
            llm_result_index += 1

        span, synthesized_start_ms = _build_span_from_entry(
            entry,
            synthesized_start_ms,
            span_name=span_name,
            name_key=name_key,
            llm_result=llm_result,
        )
        trace.spans.append(span)

        cache_status = _coerce_string(entry.get("cache_status"))
        if cache_status:
            cache_statuses.append(cache_status)

        input_flag = _coerce_string(entry.get("guardrail_input_flag"))
        if input_flag and input_flag not in input_flags:
            input_flags.append(input_flag)

        output_flag = _coerce_string(entry.get("guardrail_output_flag"))
        if output_flag and output_flag not in output_flags:
            output_flags.append(output_flag)

    trace.category = result.classification
    trace.backend = _resolve_trace_backend(llm, llm_results)
    trace.model = _resolve_trace_model(llm, llm_results)
    trace.prompt_tokens = sum(_get_span_prompt_tokens(span) for span in trace.spans)
    trace.completion_tokens = sum(_get_span_completion_tokens(span) for span in trace.spans)
    trace.total_tokens = trace.prompt_tokens + trace.completion_tokens
    trace.latency_ms = round(time.perf_counter() * 1000 - trace.start_ms, 1)
    trace.retrieved_count = len(result.retrieved_doc_ids)
    trace.cache_status = _derive_trace_cache_status(cache_statuses)
    trace.refused = result.outcome == "refused"
    trace.escalated = result.outcome == "escalated"
    trace.guardrail_input_flag = ",".join(input_flags)
    trace.guardrail_output_flag = ",".join(output_flags)
    trace.workflow_steps = [span.name for span in trace.spans]

    if not cache_statuses and any(span.name in _LLM_TOUCHING_SPANS for span in trace.spans):
        LOGGER.warning(
            "Workflow trace entries did not expose cache_status; leaving Trace.cache_status empty."
        )


def _build_span_from_entry(
    entry: Mapping[str, object],
    synthesized_start_ms: float,
    *,
    span_name: str,
    name_key: str | None,
    llm_result: object | None,
) -> tuple[Span, float]:
    attributes = dict(entry)
    if name_key is not None:
        attributes.pop(name_key, None)

    _merge_llm_result_attributes(attributes, llm_result)

    start_ms = _coerce_float(entry.get("start_ms"))
    if start_ms is None:
        start_ms = synthesized_start_ms

    end_ms = _coerce_float(entry.get("end_ms"))
    latency_ms = _coerce_float(entry.get("latency_ms"))
    if end_ms is None:
        end_ms = start_ms + (latency_ms or 0.0)

    return (
        Span(name=span_name, start_ms=start_ms, end_ms=end_ms, attributes=attributes),
        end_ms,
    )


def _get_span_name(entry: Mapping[str, object]) -> tuple[str, str | None]:
    for key in ("node", "name", "step"):
        value = _coerce_string(entry.get(key))
        if value:
            return value, key
    return "unknown", None


def _merge_llm_result_attributes(
    attributes: dict[str, object],
    llm_result: object | None,
) -> None:
    if llm_result is None:
        return

    tokens_in = _coerce_int(getattr(llm_result, "tokens_in", None))
    tokens_out = _coerce_int(getattr(llm_result, "tokens_out", None))
    cost_estimate = _coerce_float(getattr(llm_result, "cost_estimate_usd", None))

    if tokens_in is not None:
        attributes.setdefault("tokens_in", tokens_in)
        attributes.setdefault("prompt_tokens", tokens_in)
    if tokens_out is not None:
        attributes.setdefault("tokens_out", tokens_out)
        attributes.setdefault("completion_tokens", tokens_out)
    if cost_estimate is not None:
        attributes.setdefault("cost_estimate_usd", cost_estimate)


def _extract_llm_completion_results(llm: object | None) -> list[object]:
    if llm is None:
        return []

    explicit_results = getattr(llm, "results", None)
    if isinstance(explicit_results, list):
        return list(explicit_results)

    calls = getattr(llm, "calls", None)
    if not isinstance(calls, list):
        return []

    results: list[object] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        result = call.get("result")
        if result is not None:
            results.append(result)
    return results


def _resolve_trace_backend(llm: object | None, llm_results: Sequence[object]) -> str:
    if llm is None:
        return "mock"

    backend = _resolve_object_string(llm, ("provider", "provider_name", "_provider_name"))
    if backend:
        return backend

    for result in llm_results:
        provider = _coerce_string(getattr(result, "provider", None))
        if provider:
            return provider
    return ""


def _resolve_trace_model(llm: object | None, llm_results: Sequence[object]) -> str:
    if llm is None:
        return ""

    model = _resolve_object_string(llm, ("model", "model_name", "_model"))
    if model:
        return model

    for result in llm_results:
        completion_model = _coerce_string(getattr(result, "model", None))
        if completion_model:
            return completion_model
    return ""


def _resolve_object_string(target: object, attributes: Sequence[str]) -> str:
    for attribute in attributes:
        value = getattr(target, attribute, None)
        if callable(value):
            continue
        normalized = _coerce_string(value)
        if normalized:
            return normalized
    return ""


def _get_span_prompt_tokens(span: Span) -> int:
    return _coerce_int(
        span.attributes.get("prompt_tokens", span.attributes.get("tokens_in"))
    ) or 0


def _get_span_completion_tokens(span: Span) -> int:
    return _coerce_int(
        span.attributes.get("completion_tokens", span.attributes.get("tokens_out"))
    ) or 0


def _derive_trace_cache_status(cache_statuses: Sequence[str]) -> str:
    normalized = [status for status in cache_statuses if status]
    if not normalized:
        return ""
    if "miss" in normalized:
        return "miss"
    if "hit" in normalized:
        return "hit"
    if "bypass" in normalized:
        return "bypass"
    return normalized[0]


def _coerce_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
