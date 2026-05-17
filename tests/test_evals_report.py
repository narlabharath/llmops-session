from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest

from src.evals import (
    GoldenRow,
    JudgeVerdict,
    RowResult,
    build_report,
    dump_report,
    print_report,
)


def test_build_report_aggregates_eval_totals_and_category_pass_rates() -> None:
    report = build_report(
        [
            _make_row_result(
                row_id="GQ001",
                category="policy",
                passed=True,
                latency_ms=7.0,
                cost_estimate_usd=0.05,
            ),
            _make_row_result(
                row_id="GQ002",
                category="policy",
                passed=False,
                failures=["expected refusal, got answered"],
                groundedness=JudgeVerdict(
                    grounded=False,
                    confidence=0.1,
                    reason="Unsupported claim.",
                    model="scripted",
                ),
                latency_ms=6.5,
                cost_estimate_usd=0.10,
            ),
            _make_row_result(
                row_id="GQ003",
                category="schedule",
                passed=True,
                workflow_outcome="refused",
                workflow_text="I can't help with that request.",
                latency_ms=2.5,
                cost_estimate_usd=0.01,
            ),
        ]
    )

    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.failed_ids == ["GQ002"]
    assert report.total_latency_ms == pytest.approx(16.0)
    assert report.total_cost_usd == pytest.approx(0.16)
    assert report.summary == {"policy": 0.5, "schedule": 1.0}


def test_print_report_and_dump_report_render_human_and_machine_outputs(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    report = build_report(
        [
            _make_row_result(
                row_id="GQ010",
                category="out_of_scope",
                passed=True,
                workflow_outcome="refused",
                workflow_text="I'm only able to help with program questions.",
                latency_ms=1.0,
                cost_estimate_usd=0.0,
            ),
            _make_row_result(
                row_id="GQ011",
                category="ambiguous",
                passed=False,
                failures=["answer not grounded: unsupported schedule detail"],
                groundedness=JudgeVerdict(
                    grounded=False,
                    confidence=0.2,
                    reason="unsupported schedule detail",
                    model="scripted",
                ),
                latency_ms=3.0,
                cost_estimate_usd=0.02,
            ),
        ]
    )

    with caplog.at_level(logging.INFO):
        print_report(report)

    rendered_output = "\n".join(caplog.messages)
    assert "Eval report" in rendered_output
    assert "Pass rate: 50.00%" in rendered_output
    assert "GQ011: answer not grounded: unsupported schedule detail" in rendered_output

    csv_path, json_path = dump_report(report, tmp_path)

    assert csv_path == tmp_path / "eval_report.csv"
    assert json_path == tmp_path / "eval_report.json"
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 2
    assert csv_rows[0]["id"] == "GQ010"
    assert csv_rows[1]["grounded"] == "False"
    assert csv_rows[1]["failures"] == "answer not grounded: unsupported schedule detail"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["failed_ids"] == ["GQ011"]
    assert payload["summary"] == {"ambiguous": 0.0, "out_of_scope": 1.0}
    assert payload["rows"][1]["groundedness"]["reason"] == "unsupported schedule detail"


def _make_row_result(
    *,
    row_id: str,
    category: str,
    passed: bool,
    failures: list[str] | None = None,
    groundedness: JudgeVerdict | None = None,
    workflow_outcome: str = "answered",
    workflow_text: str = "Scripted answer",
    latency_ms: float = 0.0,
    cost_estimate_usd: float = 0.0,
) -> RowResult:
    return RowResult(
        row=GoldenRow(
            id=row_id,
            query=f"Question for {row_id}",
            category=category,
            expected_behavior="answer",
            reference_answer="Reference answer",
            required_context="sample_source",
            risk_level="low",
            should_retrieve=True,
            should_refuse=False,
            should_escalate=False,
            notes=None,
        ),
        workflow_outcome=workflow_outcome,
        workflow_text=workflow_text,
        behavior_match=passed,
        groundedness=groundedness,
        similarity_score=1.0 if passed else 0.25,
        passed=passed,
        failures=failures or [],
        latency_ms=latency_ms,
        cost_estimate_usd=cost_estimate_usd,
    )
