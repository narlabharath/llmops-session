from __future__ import annotations

from src.observability import SessionMetrics, Span, Trace


def test_span_finish_merges_attributes_and_updates_duration() -> None:
    span = Span(
        name="classify",
        start_ms=0.0,
        attributes={"prompt_version": "v1"},
    )

    assert span.duration_ms == 0.0

    span.finish(end_ms=10.5, cache_status="hit", prompt_tokens=12)

    assert span.duration_ms == 10.5
    assert span.attributes == {
        "prompt_version": "v1",
        "cache_status": "hit",
        "prompt_tokens": 12,
    }


def test_trace_defaults_match_lc47_contract() -> None:
    trace = Trace(trace_id="ab3e", query="x", start_ms=0.0)

    assert trace.trace_id == "ab3e"
    assert trace.query == "x"
    assert trace.start_ms == 0.0
    assert trace.spans == []
    assert trace.category == ""
    assert trace.backend == ""
    assert trace.model == ""
    assert trace.prompt_tokens == 0
    assert trace.completion_tokens == 0
    assert trace.total_tokens == 0
    assert trace.latency_ms == 0.0
    assert trace.retrieved_count == 0
    assert trace.cache_status == ""
    assert trace.refused is False
    assert trace.escalated is False
    assert trace.guardrail_input_flag == ""
    assert trace.guardrail_output_flag == ""
    assert trace.workflow_steps == []


def test_session_metrics_zero_query_properties_do_not_crash() -> None:
    metrics = SessionMetrics()

    assert metrics.avg_latency_ms == 0.0
    assert metrics.avg_tokens == 0.0
    assert metrics.refuse_rate_float == 0.0
    assert metrics.refuse_rate == "0.0%"


def test_session_metrics_refuse_rate_handles_all_refused_queries() -> None:
    metrics = SessionMetrics(
        total_queries=3,
        total_tokens=90,
        total_latency_ms=15.0,
        refused_count=3,
        category_counts={"out_of_scope": 3},
    )

    assert metrics.avg_latency_ms == 5.0
    assert metrics.avg_tokens == 30.0
    assert metrics.refuse_rate_float == 1.0
    assert metrics.refuse_rate == "100.0%"
    assert metrics.as_dict()["category_counts"] == {"out_of_scope": 3}
