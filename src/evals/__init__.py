"""Evaluation harness public API."""

from __future__ import annotations

from .judge import judge_groundedness
from .report import build_report, dump_report, print_report
from .runner import load_golden_rows, run_evals
from .scoring import score_behavior, score_similarity
from .types import EvalReport, GoldenRow, JudgeVerdict, RowResult

__all__ = [
    "EvalReport",
    "GoldenRow",
    "JudgeVerdict",
    "RowResult",
    "build_report",
    "dump_report",
    "judge_groundedness",
    "load_golden_rows",
    "print_report",
    "run_evals",
    "score_behavior",
    "score_similarity",
]
