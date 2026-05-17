from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.rag.retriever as retriever_module
from src.llm import CompletionResult, load_prompt
from src.rag import Chunk, DocumentMetadata, build_vector_store
from src.workflow import build_workflow_graph, run_workflow


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


class SequencedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        self.calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})
        index = len(self.calls) - 1
        if index >= len(self._responses):
            raise AssertionError(f"No scripted response available for call {index}.")

        return CompletionResult(
            text=self._responses[index],
            model="mock-model-v1",
            provider="mock",
            latency_ms=9.0 + index,
            tokens_in=20,
            tokens_out=8,
            cost_estimate_usd=0.0,
            cache_status="bypass",
            raw=None,
        )


def test_build_workflow_graph_compiles(tmp_path: Path) -> None:
    compiled_graph = build_workflow_graph(
        persist_dir=tmp_path / "unused-store",
        llm=SequencedLLM(["out_of_scope"]),
    )

    assert hasattr(compiled_graph, "invoke")


def test_run_workflow_answers_for_question_classifications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_dir = tmp_path / "chroma"
    _patch_retriever_embeddings(monkeypatch)
    _build_store(
        persist_dir,
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
            Chunk(
                text="Support hours are weekdays from 9am to 5pm.",
                metadata=DocumentMetadata(
                    document_id="support_process",
                    title="Support Process",
                    document_type="process",
                    source_priority=3,
                ),
                chunk_index=0,
                source_path="data/sample_support_process.md",
            ),
        ],
    )
    llm = SequencedLLM(
        [
            "policy_question",
            '{"answer": "Late submissions are allowed for two grace days.", "grounded": true}',
        ]
    )

    result = run_workflow(
        question="What is the late submission policy?",
        persist_dir=persist_dir,
        llm=llm,
        k=1,
        escalation_threshold=0.25,
    )

    assert result.outcome == "answered"
    assert result.text == "Late submissions are allowed for two grace days."
    assert result.classification == "policy_question"
    assert result.retrieved_doc_ids == ["program_policy"]
    assert [entry["node"] for entry in result.trace] == ["classify", "retrieve", "answer"]
    assert result.trace[0]["route_to"] == "retrieve"
    assert result.trace[1]["route_to"] == "answer"
    assert result.trace[1]["routing_reason"] == "sufficient_score"
    assert result.trace[1]["escalation_threshold"] == 0.25
    assert result.trace[1]["retrieved_count"] == 1
    assert len(llm.calls) == 2


def test_run_workflow_refuses_for_out_of_scope_questions(tmp_path: Path) -> None:
    question = "How do I file my taxes?"
    llm = SequencedLLM(["out_of_scope"])

    result = run_workflow(
        question=question,
        persist_dir=tmp_path / "unused-store",
        llm=llm,
    )

    assert result.outcome == "refused"
    assert result.classification == "out_of_scope"
    assert result.retrieved_doc_ids == []
    assert result.text == load_prompt("refuse_out_of_scope", version="v1").format(question=question)
    assert [entry["node"] for entry in result.trace] == ["classify", "refuse"]
    assert result.trace[0]["route_to"] == "refuse"
    assert len(llm.calls) == 1


def test_run_workflow_escalates_when_retrieval_finds_no_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_dir = tmp_path / "empty-chroma"
    _patch_retriever_embeddings(monkeypatch)
    _build_store(persist_dir, chunks=[])
    question = "What is the late submission policy for internship deferrals?"
    llm = SequencedLLM(["policy_question"])

    result = run_workflow(
        question=question,
        persist_dir=persist_dir,
        llm=llm,
    )

    assert result.outcome == "escalated"
    assert result.classification == "policy_question"
    assert result.retrieved_doc_ids == []
    assert result.text == load_prompt("escalate_low_confidence", version="v1").format(
        question=question
    )
    assert [entry["node"] for entry in result.trace] == ["classify", "retrieve", "escalate"]
    assert result.trace[1]["route_to"] == "escalate"
    assert result.trace[1]["routing_reason"] == "no_documents"
    assert result.trace[1]["escalation_threshold"] == 0.3
    assert result.trace[1]["max_score"] is None
    assert len(llm.calls) == 1


def _patch_retriever_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module,
        "get_embeddings",
        lambda model_name=retriever_module.DEFAULT_EMBEDDING_MODEL: FakeEmbeddings(),
    )


def _build_store(persist_dir: Path, chunks: list[Chunk]) -> None:
    store = build_vector_store(
        chunks=chunks,
        persist_directory=persist_dir,
        embeddings=FakeEmbeddings(),
    )
    _shutdown_store(store)


def _shutdown_store(store: Any) -> None:
    store._client._system.stop()
    store._client.clear_system_cache()
