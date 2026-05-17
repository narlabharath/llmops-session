from __future__ import annotations

from src.observability import SessionMetrics, Trace, compute_metrics


def test_compute_metrics_empty_trace_list_returns_zeroed_metrics() -> None:
    metrics = compute_metrics([])

    assert metrics == SessionMetrics()
    assert metrics.avg_latency_ms == 0.0
    assert metrics.refuse_rate == "0.0%"


def test_compute_metrics_counts_refused_trace_and_refuse_rate() -> None:
    trace = Trace(
        trace_id="ab3e",
        query="question",
        start_ms=0.0,
        total_tokens=11,
        prompt_tokens=7,
        completion_tokens=4,
        latency_ms=12.5,
        refused=True,
        category="out_of_scope",
    )

    metrics = compute_metrics([trace])

    assert metrics.total_queries == 1
    assert metrics.total_tokens == 11
    assert metrics.total_prompt_tokens == 7
    assert metrics.total_completion_tokens == 4
    assert metrics.total_latency_ms == 12.5
    assert metrics.refused_count == 1
    assert metrics.refuse_rate == "100.0%"
    assert metrics.category_counts == {"out_of_scope": 1}


def test_compute_metrics_aggregates_categories_and_trace_level_counts() -> None:
    traces = [
        Trace(
            trace_id="a1",
            query="q1",
            start_ms=0.0,
            category="policy",
            total_tokens=20,
            prompt_tokens=12,
            completion_tokens=8,
            latency_ms=10.0,
            retrieved_count=2,
        ),
        Trace(
            trace_id="b2",
            query="q2",
            start_ms=0.0,
            category="policy",
            total_tokens=16,
            prompt_tokens=9,
            completion_tokens=7,
            latency_ms=15.0,
            escalated=True,
            guardrail_input_flag="prompt_injection",
        ),
        Trace(
            trace_id="c3",
            query="q3",
            start_ms=0.0,
            category="security",
            total_tokens=9,
            prompt_tokens=5,
            completion_tokens=4,
            latency_ms=5.0,
            retrieved_count=1,
            guardrail_input_flag="pii",
            guardrail_output_flag="policy",
        ),
        Trace(
            trace_id="d4",
            query="q4",
            start_ms=0.0,
            total_tokens=4,
            prompt_tokens=2,
            completion_tokens=2,
            latency_ms=2.5,
        ),
    ]

    metrics = compute_metrics(traces)

    assert metrics.total_queries == 4
    assert metrics.total_tokens == 49
    assert metrics.total_prompt_tokens == 28
    assert metrics.total_completion_tokens == 21
    assert metrics.total_latency_ms == 32.5
    assert metrics.retrieval_count == 2
    assert metrics.escalated_count == 1
    assert metrics.guardrail_blocks == 2
    assert metrics.category_counts == {"policy": 2, "security": 1}
