"""Composed guardrail entry points for input and output safety checks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

from src.llm import LLMClient

from .input import check_input
from .output import check_output
from .scope import check_scope
from .types import GuardrailDecision, GuardrailReport, RedTeamCase

LOGGER = logging.getLogger(__name__)


def run_input_guardrails(
    question: str,
    llm: LLMClient | None = None,
    skip_scope: bool = False,
) -> GuardrailReport:
    """Run the input guardrails and optionally the LLM-based scope guard."""

    decisions = list(check_input(question).decisions)
    if not skip_scope:
        decisions.append(check_scope(question, llm=llm))

    report = _compose_report(decisions)
    if report.overall == "block":
        LOGGER.warning("Input guardrails blocked question via: %s", ", ".join(report.blocked_by))
    return report


def run_output_guardrails(
    answer: str,
    system_prompt: str | None = None,
) -> GuardrailReport:
    """Run the output guardrails for a model answer."""

    return check_output(answer, system_prompt=system_prompt)


def evaluate_against_red_team(
    cases: list[RedTeamCase],
    llm: LLMClient | None = None,
) -> dict[str, GuardrailReport]:
    """Evaluate red-team cases with the composed input guardrail pipeline."""

    reports: dict[str, GuardrailReport] = {}
    for case in cases:
        reports[case.id] = run_input_guardrails(case.input, llm=llm)
    return reports


def _compose_report(decisions: Iterable[GuardrailDecision]) -> GuardrailReport:
    resolved_decisions = list(decisions)
    blocked_by = [
        decision.guardrail
        for decision in resolved_decisions
        if decision.severity == "block"
    ]

    if blocked_by:
        overall: Literal["pass", "warn", "block"] = "block"
    elif any(decision.severity == "warn" for decision in resolved_decisions):
        overall = "warn"
    else:
        overall = "pass"

    return GuardrailReport(
        decisions=resolved_decisions,
        overall=overall,
        blocked_by=blocked_by,
    )


__all__ = [
    "evaluate_against_red_team",
    "run_input_guardrails",
    "run_output_guardrails",
]
