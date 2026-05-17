"""Local in-memory trace storage helpers."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from .types import Trace

LOGGER = logging.getLogger(__name__)

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
