from __future__ import annotations

from pathlib import Path
from typing import Any

from src.rag import (
    Chunk,
    DocumentMetadata,
    build_vector_store,
    load_vector_store,
    query_store,
)


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


def test_build_and_load_vector_store_round_trips_chunk_metadata(tmp_path: Path) -> None:
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
        persist_directory=tmp_path,
        embeddings=FakeEmbeddings(),
    )
    _shutdown_store(store)

    loaded_store = load_vector_store(tmp_path, embeddings=FakeEmbeddings())
    retrieved_documents = query_store(loaded_store, "late policy", k=2)

    assert len(retrieved_documents) == 2
    assert retrieved_documents[0].chunk.metadata.document_id == "program_policy"
    assert retrieved_documents[0].chunk.metadata.source_priority == 1
    assert retrieved_documents[0].chunk.source_path == "data/sample_program_policy.md"
    assert retrieved_documents[1].chunk.metadata.document_id == "support_process"

    _shutdown_store(loaded_store)


def test_query_store_returns_ranked_similarity_scores(tmp_path: Path) -> None:
    store = build_vector_store(
        chunks=[
            Chunk(
                text="Late submission policy with two grace days.",
                metadata=DocumentMetadata(document_id="program_policy", source_priority=1),
                chunk_index=0,
                source_path="data/sample_program_policy.md",
            ),
            Chunk(
                text="Support hours are weekdays from 9am to 5pm.",
                metadata=DocumentMetadata(document_id="support_process", source_priority=3),
                chunk_index=0,
                source_path="data/sample_support_process.md",
            ),
        ],
        persist_directory=tmp_path,
        embeddings=FakeEmbeddings(),
    )
    retrieved_documents = query_store(store, "late policy", k=2)

    assert [document.rank for document in retrieved_documents] == [0, 1]
    assert all(0.0 <= document.score <= 1.0 for document in retrieved_documents)
    assert retrieved_documents[0].score > retrieved_documents[1].score
    assert retrieved_documents[0].chunk.metadata.document_id == "program_policy"

    _shutdown_store(store)


def _shutdown_store(store: Any) -> None:
    store._client._system.stop()
    store._client.clear_system_cache()
