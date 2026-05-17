from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

import src.rag.retriever as retriever_module
from src.llm import CompletionResult
from src.observability import LocalTraceStore, get_store, trace_workflow
from src.rag import Chunk, DocumentMetadata, build_vector_store
from src.workflow import run_workflow
from src.workflow.types import WorkflowResult


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


class TraceScriptedLLM:
    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = responses
        self.provider = "scripted"
        self.model = "trace-test-model"
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        index = len(self.calls)
        if index >= len(self._responses):
            raise AssertionError(f"No scripted response available for call {index}.")

        result = self._responses[index]
        self.calls.append(
            {"prompt": prompt, "system": system, "kwargs": kwargs, "result": result}
        )
        return result


def test_trace_workflow_populates_trace_for_answered_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_dir = _build_policy_store(tmp_path, monkeypatch)
    question = "What is the late submission policy?"
    llm = TraceScriptedLLM(
        [
            _completion_result(
                text="policy_question",
                latency_ms=9.5,
                tokens_in=11,
                tokens_out=2,
                cache_status="bypass",
            ),
            _completion_result(
                text=json.dumps(
                    {
                        "answer": "Late submissions are allowed for two grace days.",
                        "grounded": True,
                    }
                ),
                latency_ms=18.0,
                tokens_in=37,
                tokens_out=9,
                cache_status="miss",
            ),
        ]
    )
    store = LocalTraceStore()

    trace = trace_workflow(
        run_workflow,
        question=question,
        persist_dir=persist_dir,
        llm=llm,
        store=store,
        k=1,
        escalation_threshold=0.25,
    )

    assert store.all_traces() == [trace]
    assert trace.query == question
    assert trace.category == "policy_question"
    assert trace.backend == "scripted"
    assert trace.model == "trace-test-model"
    assert trace.prompt_tokens == 48
    assert trace.completion_tokens == 11
    assert trace.total_tokens == 59
    assert trace.latency_ms > 0
    assert trace.retrieved_count == 1
    assert trace.cache_status == "miss"
    assert trace.refused is False
    assert trace.escalated is False
    assert trace.workflow_steps == ["classify", "retrieve", "answer"]
    assert [span.name for span in trace.spans] == ["classify", "retrieve", "answer"]
    assert trace.spans[0].attributes["route_to"] == "retrieve"
    assert trace.spans[0].attributes["prompt_tokens"] == 11
    assert trace.spans[1].attributes["route_to"] == "answer"
    assert trace.spans[2].attributes["cache_status"] == "miss"
    assert trace.spans[2].attributes["cost_estimate_usd"] == 0.01
    assert trace.spans[2].attributes["retrieved_doc_ids"] == ["program_policy"]


def test_trace_workflow_marks_refused_outcome(tmp_path: Path) -> None:
    question = "How do I file my taxes?"
    llm = TraceScriptedLLM(
        [
            _completion_result(
                text="out_of_scope",
                latency_ms=8.0,
                tokens_in=13,
                tokens_out=2,
                cache_status="bypass",
            )
        ]
    )

    trace = trace_workflow(
        run_workflow,
        question=question,
        persist_dir=tmp_path / "unused-store",
        llm=llm,
        store=LocalTraceStore(),
    )

    assert trace.refused is True
    assert trace.escalated is False
    assert trace.category == "out_of_scope"
    assert trace.retrieved_count == 0
    assert trace.workflow_steps == ["classify", "refuse"]


def test_trace_workflow_marks_escalated_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_dir = _build_empty_store(tmp_path, monkeypatch)
    llm = TraceScriptedLLM(
        [
            _completion_result(
                text="policy_question",
                latency_ms=7.0,
                tokens_in=10,
                tokens_out=2,
                cache_status="bypass",
            )
        ]
    )

    trace = trace_workflow(
        run_workflow,
        question="What is the late submission policy for internship deferrals?",
        persist_dir=persist_dir,
        llm=llm,
        store=LocalTraceStore(),
    )

    assert trace.refused is False
    assert trace.escalated is True
    assert trace.category == "policy_question"
    assert trace.workflow_steps == ["classify", "retrieve", "escalate"]


def test_trace_workflow_uses_custom_store_and_keeps_unique_trace_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_dir = _build_policy_store(tmp_path, monkeypatch)
    singleton = get_store()
    singleton.clear()
    custom_store = LocalTraceStore()
    question = "What is the late submission policy?"

    first_trace = trace_workflow(
        run_workflow,
        question=question,
        persist_dir=persist_dir,
        llm=_answered_llm(cache_status="miss"),
        store=custom_store,
        k=1,
        escalation_threshold=0.25,
    )
    second_trace = trace_workflow(
        run_workflow,
        question=question,
        persist_dir=persist_dir,
        llm=_answered_llm(cache_status="hit"),
        store=custom_store,
        k=1,
        escalation_threshold=0.25,
    )

    assert len(custom_store.all_traces()) == 2
    assert first_trace.category == second_trace.category == "policy_question"
    assert first_trace.trace_id != second_trace.trace_id
    assert singleton.all_traces() == []


def test_trace_workflow_warns_when_workflow_trace_has_no_cache_status(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_run_workflow(
        question: str,
        persist_dir: Path,
        llm: object | None = None,
        **kwargs: object,
    ) -> WorkflowResult:
        return WorkflowResult(
            outcome="answered",
            text="Synthetic answer",
            classification="policy_question",
            retrieved_doc_ids=["program_policy"],
            trace=[
                {"node": "classify", "latency_ms": 1.0, "route_to": "retrieve"},
                {
                    "node": "retrieve",
                    "latency_ms": 2.0,
                    "retrieved_doc_ids": ["program_policy"],
                    "retrieved_count": 1,
                    "route_to": "answer",
                },
                {"node": "answer", "latency_ms": 3.0, "prompt_version": "v1"},
            ],
        )

    with caplog.at_level(logging.WARNING):
        trace = trace_workflow(
            fake_run_workflow,
            question="What is the late submission policy?",
            persist_dir=tmp_path / "unused-store",
            llm=_answered_llm(cache_status="miss"),
            store=LocalTraceStore(),
        )

    assert trace.cache_status == ""
    assert trace.workflow_steps == ["classify", "retrieve", "answer"]
    assert "did not expose cache_status" in caplog.text


def _answered_llm(cache_status: str) -> TraceScriptedLLM:
    return TraceScriptedLLM(
        [
            _completion_result(
                text="policy_question",
                latency_ms=9.5,
                tokens_in=11,
                tokens_out=2,
                cache_status="bypass",
            ),
            _completion_result(
                text=json.dumps({"answer": "Late submissions are allowed for two grace days."}),
                latency_ms=18.0,
                tokens_in=37,
                tokens_out=9,
                cache_status=cache_status,
            ),
        ]
    )


def _completion_result(
    *,
    text: str,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    cache_status: str,
) -> CompletionResult:
    return CompletionResult(
        text=text,
        model="trace-test-model",
        provider="scripted",
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_estimate_usd=0.01,
        cache_status=cache_status,  # type: ignore[arg-type]
        raw=None,
    )


def _build_policy_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    persist_dir = tmp_path / "observability-policy-store"
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
    persist_dir = tmp_path / "observability-empty-store"
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
