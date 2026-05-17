from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

import src.observability as observability
import src.rag.retriever as retriever_module
from src.llm import CompletionResult
from src.observability import (
    LocalTraceStore,
    SessionMetrics,
    Span,
    Trace,
    compute_metrics,
    get_store,
    trace_workflow,
)
from src.rag import Chunk, DocumentMetadata, build_vector_store
from src.workflow import run_workflow

EXPECTED_PUBLIC_API = [
    "Span",
    "Trace",
    "SessionMetrics",
    "LocalTraceStore",
    "get_store",
    "trace_workflow",
    "compute_metrics",
]

EXPECTED_DATAFRAME_COLUMNS = [
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


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "late" in lowered else 0.0,
            1.0 if "policy" in lowered else 0.0,
            1.0 if "support" in lowered else 0.0,
        ]


@dataclass(frozen=True)
class ScriptedResponse:
    trigger: str
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float = 1.0
    cost_estimate_usd: float = 0.01


@dataclass
class CachedScriptedLLM:
    responses: list[ScriptedResponse]
    provider: str = "scripted"
    model: str = "smoke-test-model"
    shared_cache: dict[str, CompletionResult] | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        cache: bool | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        response = self._match_response(prompt, system)
        cache_enabled = self.shared_cache is not None if cache is None else cache
        cache_key = _build_cache_key(prompt, system, kwargs)

        if cache_enabled and self.shared_cache is not None and cache_key in self.shared_cache:
            result = replace(
                self.shared_cache[cache_key],
                cache_status="hit",
                latency_ms=0.01,
            )
        else:
            result = CompletionResult(
                text=response.text,
                model=self.model,
                provider=self.provider,
                latency_ms=response.latency_ms,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                cost_estimate_usd=response.cost_estimate_usd,
                cache_status="miss" if cache_enabled else "bypass",
                raw={"matched_trigger": response.trigger},
            )
            if cache_enabled and self.shared_cache is not None:
                self.shared_cache[cache_key] = result

        self.calls.append(
            {"prompt": prompt, "system": system, "kwargs": kwargs, "result": result}
        )
        return result

    def _match_response(self, prompt: str, system: str | None) -> ScriptedResponse:
        for response in self.responses:
            if response.trigger in prompt or (system and response.trigger in system):
                return response

        raise AssertionError(
            "No scripted rule matched. "
            f"Prompt was: {prompt[:200]!r}; system was: {(system or '')[:200]!r}"
        )


def test_observability_public_api_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert observability.__all__ == EXPECTED_PUBLIC_API
    assert [getattr(observability, name).__name__ for name in EXPECTED_PUBLIC_API] == (
        EXPECTED_PUBLIC_API
    )

    singleton_store = get_store()
    singleton_store.clear()
    store = LocalTraceStore()
    policy_store = _build_policy_store(tmp_path, monkeypatch)
    empty_store = _build_empty_store(tmp_path, monkeypatch)
    shared_cache: dict[str, CompletionResult] = {}
    answered_question = "What is the late submission policy?"

    answered_miss = trace_workflow(
        run_workflow,
        question=answered_question,
        persist_dir=policy_store,
        llm=_answered_llm(answered_question, shared_cache),
        store=store,
        k=1,
        escalation_threshold=0.25,
    )
    answered_hit = trace_workflow(
        run_workflow,
        question=answered_question,
        persist_dir=policy_store,
        llm=_answered_llm(answered_question, shared_cache),
        store=store,
        k=1,
        escalation_threshold=0.25,
    )
    refused_trace = trace_workflow(
        run_workflow,
        question="How do I file my taxes?",
        persist_dir=tmp_path / "unused-store",
        llm=_refused_llm("How do I file my taxes?"),
        store=store,
    )
    escalated_trace = trace_workflow(
        run_workflow,
        question="What is the late submission policy for internship deferrals?",
        persist_dir=empty_store,
        llm=_escalated_llm("What is the late submission policy for internship deferrals?"),
        store=store,
    )

    traces = store.all_traces()
    metrics = compute_metrics(traces)
    dataframe = store.to_dataframe()

    assert singleton_store.all_traces() == []
    assert len(traces) == 4
    assert all(isinstance(trace, Trace) for trace in traces)
    assert all(isinstance(span, Span) for trace in traces for span in trace.spans)
    assert all(span.end_ms >= span.start_ms for trace in traces for span in trace.spans)

    assert answered_miss.workflow_steps == ["classify", "retrieve", "answer"]
    assert answered_miss.cache_status == "miss"
    assert answered_miss.spans[-1].attributes["cache_status"] == "miss"
    assert answered_miss.total_tokens > 0

    assert answered_hit.trace_id != answered_miss.trace_id
    assert answered_hit.workflow_steps == ["classify", "retrieve", "answer"]
    assert answered_hit.cache_status == "hit"
    assert answered_hit.spans[-1].attributes["cache_status"] == "hit"
    assert answered_hit.total_tokens > 0

    assert refused_trace.refused is True
    assert refused_trace.escalated is False
    assert refused_trace.workflow_steps == ["classify", "refuse"]
    assert isinstance(refused_trace.guardrail_input_flag, str)
    assert isinstance(refused_trace.guardrail_output_flag, str)

    assert escalated_trace.refused is False
    assert escalated_trace.escalated is True
    assert escalated_trace.workflow_steps == ["classify", "retrieve", "escalate"]

    assert isinstance(metrics, SessionMetrics)
    assert metrics.total_queries == 4
    assert metrics.refused_count == 1
    assert metrics.escalated_count == 1
    assert metrics.retrieval_count == 2
    assert metrics.guardrail_blocks == 0
    assert metrics.category_counts == {
        "policy_question": 3,
        "out_of_scope": 1,
    }
    assert metrics.total_tokens == sum(trace.total_tokens for trace in traces)
    assert metrics.avg_latency_ms > 0

    assert list(dataframe.columns) == EXPECTED_DATAFRAME_COLUMNS
    assert dataframe.shape == (4, len(EXPECTED_DATAFRAME_COLUMNS))
    assert dataframe.iloc[0]["cache_status"] == "miss"
    assert dataframe.iloc[1]["cache_status"] == "hit"
    assert dataframe.iloc[2]["workflow_steps"] == "classify → refuse"
    assert dataframe.iloc[3]["workflow_steps"] == "classify → retrieve → escalate"


def _answered_llm(
    question: str,
    shared_cache: dict[str, CompletionResult],
) -> CachedScriptedLLM:
    return CachedScriptedLLM(
        responses=[
            ScriptedResponse(
                trigger="Program documents",
                text=json.dumps(
                    {
                        "answer": "Late submissions are allowed for two grace days.",
                        "grounded": True,
                    }
                ),
                tokens_in=37,
                tokens_out=9,
                latency_ms=18.0,
            ),
            ScriptedResponse(
                trigger=question,
                text="policy_question",
                tokens_in=11,
                tokens_out=2,
                latency_ms=9.5,
            ),
        ],
        shared_cache=shared_cache,
    )


def _refused_llm(question: str) -> CachedScriptedLLM:
    return CachedScriptedLLM(
        responses=[
            ScriptedResponse(
                trigger=question,
                text="out_of_scope",
                tokens_in=13,
                tokens_out=2,
                latency_ms=8.0,
            )
        ]
    )


def _escalated_llm(question: str) -> CachedScriptedLLM:
    return CachedScriptedLLM(
        responses=[
            ScriptedResponse(
                trigger=question,
                text="policy_question",
                tokens_in=10,
                tokens_out=2,
                latency_ms=7.0,
            )
        ]
    )


def _build_cache_key(
    prompt: str,
    system: str | None,
    kwargs: dict[str, object],
) -> str:
    return json.dumps(
        {
            "prompt": prompt,
            "system": system,
            "kwargs": kwargs,
        },
        sort_keys=True,
        default=str,
    )


def _build_policy_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    persist_dir = tmp_path / "observability-smoke-policy-store"
    _patch_retriever_embeddings(monkeypatch)
    store = build_vector_store(
        chunks=[
            Chunk(
                text="Late submission policy with two grace days.",
                metadata=DocumentMetadata(
                    document_id="program_policy",
                    title="Program Policy",
                    document_type="policy",
                    source_priority=1,
                ),
                chunk_index=0,
                source_path="data/sample_program_policy.md",
            ),
        ],
        persist_directory=persist_dir,
        embeddings=FakeEmbeddings(),
    )
    _shutdown_store(store)
    return persist_dir


def _build_empty_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    persist_dir = tmp_path / "observability-smoke-empty-store"
    _patch_retriever_embeddings(monkeypatch)
    store = build_vector_store(
        chunks=[],
        persist_directory=persist_dir,
        embeddings=FakeEmbeddings(),
    )
    _shutdown_store(store)
    return persist_dir


def _patch_retriever_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module,
        "get_embeddings",
        lambda model_name=retriever_module.DEFAULT_EMBEDDING_MODEL: FakeEmbeddings(),
    )


def _shutdown_store(store: Any) -> None:
    store._client._system.stop()
    store._client.clear_system_cache()
