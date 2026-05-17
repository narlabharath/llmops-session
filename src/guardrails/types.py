"""Shared dataclasses for guardrail decisions and red-team cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GuardrailDecision:
    passed: bool
    guardrail: str
    severity: Literal["info", "warn", "block"]
    reason: str
    matched_pattern: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RedTeamCase:
    id: str
    category: str
    input: str
    expected_response_type: Literal["refuse", "escalate", "clarify"]
    risk: Literal["low", "medium", "high"]
    notes: str | None = None


@dataclass(frozen=True)
class GuardrailReport:
    decisions: list[GuardrailDecision]
    overall: Literal["pass", "warn", "block"]
    blocked_by: list[str] = field(default_factory=list)


__all__ = [
    "GuardrailDecision",
    "GuardrailReport",
    "RedTeamCase",
]
