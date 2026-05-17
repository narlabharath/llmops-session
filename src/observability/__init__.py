"""Public observability exports for traces, spans, stores, and metrics.

This package is the pure-Python observability surface referenced by
PLAN.md S-12 and used by the Batch 08 workflow demos.
"""

from __future__ import annotations

from .metrics import compute_metrics
from .tracing import LocalTraceStore, get_store, trace_workflow
from .types import SessionMetrics, Span, Trace

__all__ = [
    "Span",
    "Trace",
    "SessionMetrics",
    "LocalTraceStore",
    "get_store",
    "trace_workflow",
    "compute_metrics",
]
