from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.evals import load_golden_rows

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_queries.csv"


def test_load_golden_rows_reads_real_csv() -> None:
    rows = load_golden_rows(CSV_PATH)

    assert len(rows) == 12
    assert [row.id for row in rows] == sorted(row.id for row in rows)


def test_load_golden_rows_parses_bool_flags() -> None:
    rows = load_golden_rows(CSV_PATH)

    for row in rows:
        assert isinstance(row.should_retrieve, bool)
        assert isinstance(row.should_refuse, bool)
        assert isinstance(row.should_escalate, bool)

    prompt_injection_row = next(row for row in rows if row.id == "GQ006")
    assert prompt_injection_row.should_retrieve is False
    assert prompt_injection_row.should_refuse is True
    assert prompt_injection_row.should_escalate is True


def test_load_golden_rows_warns_and_raises_on_missing_columns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    csv_path = tmp_path / "golden_missing_columns.csv"
    csv_path.write_text("id,query,category\nGQ999,Question,informational\n", encoding="utf-8")

    caplog.set_level(logging.WARNING)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_golden_rows(csv_path)

    assert "missing expected columns" in caplog.text
    assert "expected_behavior" in caplog.text
