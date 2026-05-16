"""Schema + consistency checks for data/golden_queries.csv."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_queries.csv"
EXPECTED_COLUMNS = [
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
]
VALID_CATEGORIES = {
    "informational",
    "policy",
    "procedural",
    "schedule",
    "assignment",
    "ambiguous",
    "sensitive_private",
    "out_of_scope",
    "prompt_injection",
    "feedback_regression",
}


def load_dataframe() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def non_empty(value: Any) -> bool:
    text = str(value).strip()
    return text not in {"", "nan", "None"}


def test_csv_loads_in_pandas() -> None:
    dataframe = load_dataframe()
    assert len(dataframe) >= 12


def test_csv_has_expected_columns() -> None:
    dataframe = load_dataframe()
    assert list(dataframe.columns) == EXPECTED_COLUMNS


def test_ids_are_unique() -> None:
    dataframe = load_dataframe()
    assert dataframe["id"].is_unique


def test_ids_match_pattern() -> None:
    dataframe = load_dataframe()
    assert all(re.match(r"^GQ\d{3}$", value) for value in dataframe["id"])


def test_categories_are_valid() -> None:
    dataframe = load_dataframe()
    assert set(dataframe["category"]).issubset(VALID_CATEGORIES)


def test_should_refuse_consistency() -> None:
    dataframe = load_dataframe()
    failing = [
        row.id
        for row in dataframe.itertuples(index=False)
        if truthy(row.should_refuse) and "refuse" not in str(row.expected_behavior).strip().lower()
    ]
    assert failing == []


def test_should_escalate_consistency() -> None:
    dataframe = load_dataframe()
    failing = [
        row.id
        for row in dataframe.itertuples(index=False)
        if truthy(row.should_escalate) and not non_empty(row.notes)
    ]
    assert failing == []


def test_should_retrieve_implies_required_context() -> None:
    dataframe = load_dataframe()
    failing = [
        row.id
        for row in dataframe.itertuples(index=False)
        if truthy(row.should_retrieve) and not non_empty(row.required_context)
    ]
    assert failing == []


def test_at_least_one_per_critical_category() -> None:
    dataframe = load_dataframe()
    categories = set(dataframe["category"])
    for category in ("prompt_injection", "sensitive_private", "out_of_scope", "ambiguous"):
        assert category in categories
