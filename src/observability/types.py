"""Dataclass types shared across the observability package.

These shapes mirror the PLAN.md S-12 contract for per-request traces,
per-node spans, and aggregate session metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round(max(self.end_ms - self.start_ms, 0.0), 2)

    def finish(self, end_ms: float | None = None, **attributes: Any) -> None:
        self.end_ms = time.perf_counter() * 1000 if end_ms is None else end_ms
        self.attributes.update(attributes)


@dataclass
class Trace:
    trace_id: str
    query: str
    start_ms: float
    spans: list[Span] = field(default_factory=list)
    category: str = ""
    backend: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    retrieved_count: int = 0
    cache_status: str = ""
    refused: bool = False
    escalated: bool = False
    guardrail_input_flag: str = ""
    guardrail_output_flag: str = ""
    workflow_steps: list[str] = field(default_factory=list)


@dataclass
class SessionMetrics:
    total_queries: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    refused_count: int = 0
    escalated_count: int = 0
    retrieval_count: int = 0
    guardrail_blocks: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_queries, 1)

    @property
    def avg_tokens(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round(self.total_tokens / self.total_queries, 1)

    @property
    def refuse_rate_float(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.refused_count / self.total_queries

    @property
    def refuse_rate(self) -> str:
        return f"{self.refuse_rate_float * 100:.1f}%"

    def as_dict(self) -> dict[str, object]:
        return {
            "total_queries": self.total_queries,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens": self.avg_tokens,
            "refused_count": self.refused_count,
            "refuse_rate": self.refuse_rate,
            "escalated_count": self.escalated_count,
            "retrieval_count": self.retrieval_count,
            "guardrail_blocks": self.guardrail_blocks,
            "category_counts": dict(self.category_counts),
        }
