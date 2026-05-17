"""Shared dataclasses for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenRow:
    id: str
    query: str
    category: str
    expected_behavior: str
    reference_answer: str | None
    required_context: str | None
    risk_level: str
    should_retrieve: bool
    should_refuse: bool
    should_escalate: bool
    notes: str | None


@dataclass(frozen=True)
class JudgeVerdict:
    grounded: bool
    confidence: float
    reason: str
    model: str


@dataclass(frozen=True)
class RowResult:
    row: GoldenRow
    workflow_outcome: str
    workflow_text: str
    behavior_match: bool
    groundedness: JudgeVerdict | None
    similarity_score: float | None
    passed: bool
    failures: list[str]
    latency_ms: float
    cost_estimate_usd: float


@dataclass(frozen=True)
class EvalReport:
    rows: list[RowResult]
    pass_rate: float
    failed_ids: list[str]
    total_cost_usd: float
    total_latency_ms: float
    summary: dict[str, object]


__all__ = [
    "EvalReport",
    "GoldenRow",
    "JudgeVerdict",
    "RowResult",
]
