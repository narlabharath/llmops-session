from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from src.observability import LocalTraceStore, get_store

EXPECTED_COLUMNS = [
    "trace_id",
    "query",
    "category",
    "backend",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "retrieved_count",
    "cache_status",
    "refused",
    "escalated",
    "guardrail_input_flag",
    "guardrail_output_flag",
    "workflow_steps",
]


def test_new_trace_generates_unique_eight_character_ids() -> None:
    store = LocalTraceStore()

    trace_ids = {store.new_trace(f"q-{index}").trace_id for index in range(100)}

    assert len(trace_ids) == 100
    assert all(len(trace_id) == 8 for trace_id in trace_ids)


def test_all_traces_returns_copy_of_internal_list() -> None:
    store = LocalTraceStore()
    trace = store.new_trace("question")

    traces = store.all_traces()
    traces.clear()

    assert trace in store.all_traces()
    assert len(store.all_traces()) == 1


def test_to_dataframe_returns_expected_columns_and_rows() -> None:
    store = LocalTraceStore()
    first = store.new_trace("short question")
    first.category = "policy"
    first.backend = "mock"
    first.model = "mock-1"
    first.prompt_tokens = 12
    first.completion_tokens = 8
    first.total_tokens = 20
    first.latency_ms = 15.4
    first.retrieved_count = 2
    first.cache_status = "hit"
    first.workflow_steps = ["classify", "answer"]

    second = store.new_trace("x" * 61)
    second.category = "security"
    second.backend = "mock"
    second.model = "mock-2"
    second.prompt_tokens = 5
    second.completion_tokens = 3
    second.total_tokens = 8
    second.latency_ms = 7.2
    second.retrieved_count = 0
    second.cache_status = "miss"
    second.refused = True
    second.escalated = True
    second.guardrail_input_flag = "pii"
    second.guardrail_output_flag = "policy"
    second.workflow_steps = ["classify", "refuse"]

    dataframe = store.to_dataframe()

    assert list(dataframe.columns) == EXPECTED_COLUMNS
    assert len(dataframe) == 2
    assert dataframe.loc[0, "workflow_steps"] == "classify \u2192 answer"
    assert dataframe.loc[1, "query"] == ("x" * 60) + "..."
    assert dataframe.loc[1, "guardrail_input_flag"] == "pii"


def test_to_dataframe_returns_empty_dataframe_with_expected_columns() -> None:
    store = LocalTraceStore()

    dataframe = store.to_dataframe()

    assert dataframe.empty
    assert list(dataframe.columns) == EXPECTED_COLUMNS


def test_clear_empties_store() -> None:
    store = LocalTraceStore()
    store.new_trace("question")

    store.clear()

    assert store.all_traces() == []


def test_get_store_returns_singleton_instance() -> None:
    store = get_store()
    store.clear()

    same_store = get_store()

    assert same_store is store


def test_pandas_is_imported_lazily() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check_code = textwrap.dedent(
        """
        import json
        import sys

        import src.observability

        before = "pandas" in sys.modules
        store = src.observability.LocalTraceStore()
        store.to_dataframe()
        after = "pandas" in sys.modules
        print(json.dumps({"before": before, "after": after}))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", check_code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    lazy_import_state = json.loads(result.stdout.strip())

    assert lazy_import_state == {"before": False, "after": True}
