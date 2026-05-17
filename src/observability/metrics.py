"""Aggregate metrics helpers for observability traces."""

from __future__ import annotations

from .types import SessionMetrics, Trace


def compute_metrics(traces: list[Trace]) -> SessionMetrics:
    metrics = SessionMetrics(total_queries=len(traces))

    for trace in traces:
        metrics.total_tokens += trace.total_tokens
        metrics.total_prompt_tokens += trace.prompt_tokens
        metrics.total_completion_tokens += trace.completion_tokens
        metrics.total_latency_ms += trace.latency_ms

        if trace.refused:
            metrics.refused_count += 1
        if trace.escalated:
            metrics.escalated_count += 1
        if trace.retrieved_count > 0:
            metrics.retrieval_count += 1
        if trace.guardrail_input_flag or trace.guardrail_output_flag:
            metrics.guardrail_blocks += 1
        if trace.category:
            metrics.category_counts[trace.category] = (
                metrics.category_counts.get(trace.category, 0) + 1
            )

    return metrics
