from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import src.rag.retriever as retriever_module
from src.llm import CompletionResult, LLMClient, load_prompt
from src.rag import Chunk, DocumentMetadata, RetrievedDocument, build_vector_store
from src.workflow.nodes import generate_answer, retrieve_documents


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


class StubLLM:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        self.calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})
        return CompletionResult(
            text=self._response_text,
            model="mock-model-v1",
            provider="mock",
            latency_ms=18.0,
            tokens_in=42,
            tokens_out=12,
            cost_estimate_usd=0.0,
            cache_status="bypass",
            raw=None,
        )


def test_retrieve_documents_uses_public_rag_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_dir = tmp_path / "chroma"
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
        persist_directory=persist_dir,
        embeddings=FakeEmbeddings(),
    )
    _shutdown_store(store)
    monkeypatch.setattr(
        retriever_module,
        "get_embeddings",
        lambda model_name=retriever_module.DEFAULT_EMBEDDING_MODEL: FakeEmbeddings(),
    )

    updates = retrieve_documents(
        {"question": "What is the late submission policy?"},
        persist_dir=persist_dir,
        k=2,
    )

    retrieved_docs = updates["retrieved_docs"]
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) == 2
    assert retrieved_docs[0].chunk.metadata.document_id == "program_policy"
    assert retrieved_docs[1].chunk.metadata.document_id == "support_process"

    trace = updates["trace"]
    assert isinstance(trace, list)
    assert trace[-1]["node"] == "retrieve"
    assert trace[-1]["input"] == "What is the late submission policy?"
    assert trace[-1]["output"] == ["program_policy", "support_process"]
    assert trace[-1]["retrieved_doc_ids"] == ["program_policy", "support_process"]
    assert trace[-1]["retrieved_count"] == 2
    assert isinstance(trace[-1]["max_score"], float)
    assert isinstance(trace[-1]["latency_ms"], float)


def test_generate_answer_formats_all_retrieved_docs_and_parses_json_response() -> None:
    llm = StubLLM(
        '{"answer": "Late submissions are allowed for two grace days. See Program Policy.", '
        '"grounded": true, "source_section": "Late Submission Policy", "confidence": 0.93}'
    )
    state = {
        "question": "What is the late submission policy?",
        "retrieved_docs": [
            _make_retrieved_document(
                document_id="program_policy",
                text="Late submission policy with two grace days.",
                source_priority=1,
                rank=0,
                score=0.95,
                source_path="data/sample_program_policy.md",
            ),
            _make_retrieved_document(
                document_id="schedule",
                text="The weekly review session runs every Friday at 7pm.",
                source_priority=2,
                rank=1,
                score=0.61,
                source_path="data/sample_schedule.md",
            ),
        ],
    }

    updates = generate_answer(state, llm=llm)  # type: ignore[arg-type]

    assert updates["answer"] == "Late submissions are allowed for two grace days. See Program Policy."
    assert llm.calls[0]["system"] == load_prompt("program_assistant_system", version="v1")
    assert llm.calls[0]["kwargs"] == {"prompt_version": "v1"}

    prompt = str(llm.calls[0]["prompt"])
    assert "Late submission policy with two grace days." in prompt
    assert "The weekly review session runs every Friday at 7pm." in prompt
    assert "[source: program_policy, priority: 1, score: 0.950, rank: 0]" in prompt
    assert "[source: schedule, priority: 2, score: 0.610, rank: 1]" in prompt

    trace = updates["trace"]
    assert isinstance(trace, list)
    assert trace[-1] == {
        "node": "answer",
        "input": "What is the late submission policy?",
        "output": "Late submissions are allowed for two grace days. See Program Policy.",
        "retrieved_doc_ids": ["program_policy", "schedule"],
        "latency_ms": 18.0,
        "prompt_tokens": 42,
        "completion_tokens": 12,
        "cost_estimate_usd": 0.0,
        "cache_status": "bypass",
        "prompt_version": "v1",
    }


def test_generate_answer_with_mock_provider_uses_raw_text_when_response_is_not_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        updates = generate_answer(
            {
                "question": "What is the late submission policy?",
                "retrieved_docs": [
                    _make_retrieved_document(
                        document_id="program_policy",
                        text="Late submission policy with two grace days.",
                        source_priority=1,
                        rank=0,
                        score=0.95,
                        source_path="data/sample_program_policy.md",
                    )
                ],
            },
            llm=LLMClient(provider="mock"),
        )

    assert str(updates["answer"]).startswith("[mock:")
    assert "Answer generation returned non-JSON content" in caplog.text


def _make_retrieved_document(
    document_id: str,
    text: str,
    source_priority: int,
    rank: int,
    score: float,
    source_path: str,
) -> RetrievedDocument:
    return RetrievedDocument(
        chunk=Chunk(
            text=text,
            metadata=DocumentMetadata(
                document_id=document_id,
                source_priority=source_priority,
            ),
            chunk_index=0,
            source_path=source_path,
        ),
        score=score,
        rank=rank,
    )


def _shutdown_store(store: Any) -> None:
    store._client._system.stop()
    store._client.clear_system_cache()
