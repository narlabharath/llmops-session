"""Reporting helpers for the evaluation harness."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .types import EvalReport, RowResult

LOGGER = logging.getLogger(__name__)

_CSV_FILENAME = "eval_report.csv"
_JSON_FILENAME = "eval_report.json"


def build_report(row_results: list[RowResult]) -> EvalReport:
    """Aggregate row-level eval results into a summary report."""

    rows = list(row_results)
    total_rows = len(rows)
    passed_rows = sum(1 for row in rows if row.passed)

    return EvalReport(
        rows=rows,
        pass_rate=(passed_rows / total_rows) if total_rows else 0.0,
        failed_ids=[row.row.id for row in rows if not row.passed],
        total_cost_usd=sum(row.cost_estimate_usd for row in rows),
        total_latency_ms=sum(row.latency_ms for row in rows),
        summary=_build_category_pass_rates(rows),
    )


def print_report(report: EvalReport) -> None:
    """Log a human-readable eval summary.

    Logging is used here instead of ``print`` so callers can route the report
    through the application's existing logging configuration.
    """

    rows_total = len(report.rows)
    failed_total = len(report.failed_ids)
    passed_total = rows_total - failed_total

    for line in [
        "Eval report",
        (
            f"Rows: {rows_total} | Passed: {passed_total} | Failed: {failed_total} | "
            f"Pass rate: {report.pass_rate:.2%}"
        ),
        f"Cost: ${report.total_cost_usd:.4f} | Latency: {report.total_latency_ms:.2f} ms",
        "Category pass rates:",
        *[f"  {category}: {pass_rate:.2%}" for category, pass_rate in report.summary.items()],
        "Failed rows:" if report.failed_ids else "Failed rows: none",
        *[
            f"  {row.row.id}: {'; '.join(row.failures)}"
            for row in report.rows
            if not row.passed
        ],
    ]:
        LOGGER.info(line)


def dump_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    """Write the report to flattened CSV and structured JSON files."""

    resolved_out_dir = Path(out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolved_out_dir / _CSV_FILENAME
    json_path = resolved_out_dir / _JSON_FILENAME

    _write_csv_report(csv_path, report.rows)
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def _build_category_pass_rates(rows: list[RowResult]) -> dict[str, float]:
    category_totals: dict[str, tuple[int, int]] = {}
    for row in rows:
        passed_count, total_count = category_totals.get(row.row.category, (0, 0))
        category_totals[row.row.category] = (
            passed_count + int(row.passed),
            total_count + 1,
        )

    return {
        category: passed_count / total_count
        for category, (passed_count, total_count) in sorted(category_totals.items())
    }


def _write_csv_report(csv_path: Path, rows: list[RowResult]) -> None:
    fieldnames = [
        "id",
        "query",
        "category",
        "expected_behavior",
        "reference_answer",
        "required_context",
        "risk_level",
        "should_retrieve",
        "should_refuse",
        "should_escalate",
        "notes",
        "workflow_outcome",
        "workflow_text",
        "behavior_match",
        "grounded",
        "grounded_confidence",
        "grounded_reason",
        "grounded_model",
        "similarity_score",
        "passed",
        "failures",
        "latency_ms",
        "cost_estimate_usd",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            groundedness = row.groundedness
            writer.writerow(
                {
                    "id": row.row.id,
                    "query": row.row.query,
                    "category": row.row.category,
                    "expected_behavior": row.row.expected_behavior,
                    "reference_answer": row.row.reference_answer,
                    "required_context": row.row.required_context,
                    "risk_level": row.row.risk_level,
                    "should_retrieve": row.row.should_retrieve,
                    "should_refuse": row.row.should_refuse,
                    "should_escalate": row.row.should_escalate,
                    "notes": row.row.notes,
                    "workflow_outcome": row.workflow_outcome,
                    "workflow_text": row.workflow_text,
                    "behavior_match": row.behavior_match,
                    "grounded": groundedness.grounded if groundedness is not None else None,
                    "grounded_confidence": (
                        groundedness.confidence if groundedness is not None else None
                    ),
                    "grounded_reason": groundedness.reason if groundedness is not None else None,
                    "grounded_model": groundedness.model if groundedness is not None else None,
                    "similarity_score": row.similarity_score,
                    "passed": row.passed,
                    "failures": " | ".join(row.failures),
                    "latency_ms": row.latency_ms,
                    "cost_estimate_usd": row.cost_estimate_usd,
                }
            )


__all__ = ["build_report", "dump_report", "print_report"]
