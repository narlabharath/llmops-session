from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import src.evals.runner as runner_module
from src.evals import GoldenRow, JudgeVerdict, run_evals
from src.rag.types import Chunk, DocumentMetadata, RetrievedDocument
from src.workflow import WorkflowResult
from tests.conftest_workflow import ScriptedLLMClient


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self._vectors = vectors

    def embed_query(self, text: str) -> Sequence[float]:
        return self._vectors[text]


def test_run_evals_composes_behavior_groundedness_and_trace_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        _make_row(
            row_id="GQ001",
            query="Policy pass question",
            reference_answer="Policy answer",
        ),
        _make_row(
            row_id="GQ002",
            query="Ungrounded answer question",
        ),
        _make_row(
            row_id="GQ003",
            query="Escalated question",
        ),
    ]
    workflow_results = {
        "Policy pass question": WorkflowResult(
            outcome="answered",
            text="Policy answer",
            classification="policy_question",
            retrieved_doc_ids=["doc-1"],
            trace=[
                {"node": "classify", "latency_ms": 1.0, "cost_estimate_usd": 0.01},
                {"node": "retrieve", "latency_ms": 2.5},
                {"node": "answer", "latency_ms": 3.5, "cost_estimate_usd": 0.04},
            ],
        ),
        "Ungrounded answer question": WorkflowResult(
            outcome="answered",
            text="Hallucinated answer",
            classification="policy_question",
            retrieved_doc_ids=["doc-2"],
            trace=[
                {"node": "classify", "latency_ms": 1.5, "cost_estimate_usd": 0.02},
                {"node": "retrieve", "latency_ms": 1.0},
                {"node": "answer", "latency_ms": 4.0, "cost_estimate_usd": 0.08},
            ],
        ),
        "Escalated question": WorkflowResult(
            outcome="escalated",
            text="Please contact staff for help.",
            classification="policy_question",
            retrieved_doc_ids=["doc-3"],
            trace=[
                {"node": "classify", "latency_ms": 0.5, "cost_estimate_usd": 0.01},
                {"node": "retrieve", "latency_ms": 1.25},
                {"node": "escalate", "latency_ms": 0.75},
            ],
        ),
    }
    retrieved_docs_by_query = {
        "Policy pass question": [
            _make_retrieved_document(
                document_id="doc-1",
                text="Policy answer from source.",
                score=0.93,
                rank=0,
            )
        ],
        "Ungrounded answer question": [
            _make_retrieved_document(
                document_id="doc-2",
                text="Only source-supported answer.",
                score=0.81,
                rank=0,
            )
        ],
    }
    workflow_calls: list[dict[str, object]] = []
    retrieve_calls: list[dict[str, object]] = []
    embedding_call_count = 0
    workflow_llm = ScriptedLLMClient(rules=[])
    judge_llm = ScriptedLLMClient(
        rules=[
            (
                "Policy pass question",
                json.dumps(
                    {
                        "grounded": True,
                        "confidence": 0.97,
                        "reason": "Every factual claim is supported by the retrieved context.",
                    }
                ),
            ),
            (
                "Ungrounded answer question",
                json.dumps(
                    {
                        "grounded": False,
                        "confidence": 0.18,
                        "reason": "Unsupported claim is not supported by the retrieved context.",
                    }
                ),
            ),
        ]
    )
    embeddings = FakeEmbeddings({"Policy answer": [1.0, 2.0, 3.0]})

    def fake_run_workflow(
        question: str,
        persist_dir: Path,
        llm: object | None = None,
        k: int = 5,
        escalation_threshold: float = 0.3,
    ) -> WorkflowResult:
        workflow_calls.append(
            {
                "question": question,
                "persist_dir": Path(persist_dir),
                "llm": llm,
                "k": k,
                "escalation_threshold": escalation_threshold,
            }
        )
        return workflow_results[question]

    def fake_retrieve(
        persist_dir: Path,
        query: str,
        k: int = 5,
        embedding_model: str | None = None,
    ) -> list[RetrievedDocument]:
        del embedding_model
        retrieve_calls.append({"persist_dir": Path(persist_dir), "query": query, "k": k})
        return retrieved_docs_by_query[query]

    def fake_get_embeddings() -> FakeEmbeddings:
        nonlocal embedding_call_count
        embedding_call_count += 1
        return embeddings

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(runner_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(runner_module, "get_embeddings", fake_get_embeddings)

    results = run_evals(
        rows,
        tmp_path,
        workflow_llm=workflow_llm,
        judge_llm=judge_llm,
        k=4,
        escalation_threshold=0.25,
    )

    assert len(results) == 3
    assert results[0].passed is True
    assert results[0].behavior_match is True
    assert results[0].groundedness == JudgeVerdict(
        grounded=True,
        confidence=0.97,
        reason="Every factual claim is supported by the retrieved context.",
        model="scripted",
    )
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[0].latency_ms == pytest.approx(7.0)
    assert results[0].cost_estimate_usd == pytest.approx(0.05)
    assert results[0].failures == []

    assert results[1].passed is False
    assert results[1].behavior_match is True
    assert results[1].groundedness == JudgeVerdict(
        grounded=False,
        confidence=0.18,
        reason="Unsupported claim is not supported by the retrieved context.",
        model="scripted",
    )
    assert results[1].similarity_score is None
    assert results[1].latency_ms == pytest.approx(6.5)
    assert results[1].cost_estimate_usd == pytest.approx(0.10)
    assert results[1].failures == [
        "answer not grounded: Unsupported claim is not supported by the retrieved context."
    ]

    assert results[2].passed is False
    assert results[2].behavior_match is False
    assert results[2].groundedness is None
    assert results[2].similarity_score is None
    assert results[2].latency_ms == pytest.approx(2.5)
    assert results[2].cost_estimate_usd == pytest.approx(0.01)
    assert results[2].failures == ["expected answer, got escalated"]

    assert workflow_calls == [
        {
            "question": "Policy pass question",
            "persist_dir": tmp_path,
            "llm": workflow_llm,
            "k": 4,
            "escalation_threshold": 0.25,
        },
        {
            "question": "Ungrounded answer question",
            "persist_dir": tmp_path,
            "llm": workflow_llm,
            "k": 4,
            "escalation_threshold": 0.25,
        },
        {
            "question": "Escalated question",
            "persist_dir": tmp_path,
            "llm": workflow_llm,
            "k": 4,
            "escalation_threshold": 0.25,
        },
    ]
    assert retrieve_calls == [
        {"persist_dir": tmp_path, "query": "Policy pass question", "k": 4},
        {"persist_dir": tmp_path, "query": "Ungrounded answer question", "k": 4},
    ]
    assert embedding_call_count == 1
    assert "Policy answer from source." in str(judge_llm.calls[0]["prompt"])
    assert "Only source-supported answer." in str(judge_llm.calls[1]["prompt"])


def test_run_evals_reuses_one_llm_client_when_only_workflow_llm_is_passed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _make_row(row_id="GQ010", query="Shared client question")
    shared_llm = ScriptedLLMClient(
        rules=[
            (
                "Shared client question",
                json.dumps(
                    {
                        "grounded": True,
                        "confidence": 0.88,
                        "reason": "The answer stays within the retrieved context.",
                    }
                ),
            )
        ]
    )
    captured_llms: list[object | None] = []

    def fake_run_workflow(
        question: str,
        persist_dir: Path,
        llm: object | None = None,
        k: int = 5,
        escalation_threshold: float = 0.3,
    ) -> WorkflowResult:
        del question, persist_dir, k, escalation_threshold
        captured_llms.append(llm)
        return WorkflowResult(
            outcome="answered",
            text="Shared client answer",
            classification="policy_question",
            retrieved_doc_ids=["doc-shared"],
            trace=[{"node": "answer", "latency_ms": 2.0, "cost_estimate_usd": 0.03}],
        )

    def fake_retrieve(
        persist_dir: Path,
        query: str,
        k: int = 5,
        embedding_model: str | None = None,
    ) -> list[RetrievedDocument]:
        del persist_dir, query, k, embedding_model
        return [
            _make_retrieved_document(
                document_id="doc-shared",
                text="Shared client answer from the source.",
                score=0.72,
                rank=0,
            )
        ]

    monkeypatch.setattr(runner_module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(runner_module, "retrieve", fake_retrieve)

    results = run_evals([row], tmp_path, workflow_llm=shared_llm)

    assert captured_llms == [shared_llm]
    assert len(shared_llm.calls) == 1
    assert results[0].passed is True
    assert results[0].groundedness == JudgeVerdict(
        grounded=True,
        confidence=0.88,
        reason="The answer stays within the retrieved context.",
        model="scripted",
    )


def _make_row(
    *,
    row_id: str,
    query: str,
    reference_answer: str | None = None,
) -> GoldenRow:
    return GoldenRow(
        id=row_id,
        query=query,
        category="policy",
        expected_behavior="answer",
        reference_answer=reference_answer,
        required_context=None,
        risk_level="low",
        should_retrieve=True,
        should_refuse=False,
        should_escalate=False,
        notes=None,
    )


def _make_retrieved_document(
    *,
    document_id: str,
    text: str,
    score: float,
    rank: int,
) -> RetrievedDocument:
    return RetrievedDocument(
        chunk=Chunk(
            text=text,
            metadata=DocumentMetadata(document_id=document_id, source_priority=1),
            chunk_index=0,
            source_path=f"{document_id}.md",
        ),
        score=score,
        rank=rank,
    )
