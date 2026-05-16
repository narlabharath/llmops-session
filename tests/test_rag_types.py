from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from src.rag.types import Chunk, DocumentMetadata, IngestionResult, LoadedDocument, RetrievedDocument


def test_rag_types_are_frozen_dataclasses() -> None:
    metadata = DocumentMetadata(
        document_id="program-policy",
        title="Program Policy",
        document_type="policy",
        version="v1",
        effective_date="2026-01-15",
        owner="Course Staff",
        source_priority=1,
    )
    loaded_document = LoadedDocument(
        path=Path("data/sample_program_policy.md"),
        metadata=metadata,
        content="Late submissions require approval.",
    )
    chunk = Chunk(
        text="Late submissions require approval.",
        metadata=metadata,
        chunk_index=0,
        source_path="data/sample_program_policy.md",
    )
    retrieved = RetrievedDocument(chunk=chunk, score=0.9, rank=0)
    ingestion = IngestionResult(
        documents_loaded=5,
        chunks_created=8,
        chunks_indexed=8,
        vector_store_path=Path("data/chroma"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    for value in (metadata, loaded_document, chunk, retrieved, ingestion):
        assert is_dataclass(value)

    assert metadata.document_id == "program-policy"
    assert loaded_document.path == Path("data/sample_program_policy.md")
    assert chunk.chunk_index == 0
    assert retrieved.rank == 0
    assert ingestion.chunks_indexed == 8

    with pytest.raises(FrozenInstanceError):
        metadata.source_priority = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        loaded_document.content = "Updated content"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        chunk.chunk_index = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        retrieved.score = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ingestion.embedding_model = "different-model"  # type: ignore[misc]
