from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from src.rag import (
    Chunk,
    DocumentMetadata,
    LoadedDocument,
    chunk_corpus,
    chunk_document,
    load_corpus,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sample_documents() -> list[LoadedDocument]:
    return load_corpus(DATA_DIR)


def test_chunk_document_produces_chunks_for_each_sample_doc(
    sample_documents: list[LoadedDocument],
) -> None:
    for document in sample_documents:
        chunks = chunk_document(document)

        assert chunks
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_document_preserves_document_metadata(
    sample_documents: list[LoadedDocument],
) -> None:
    document = next(doc for doc in sample_documents if doc.metadata.document_id == "program_policy")

    chunks = chunk_document(document, chunk_size=400, chunk_overlap=50)

    assert chunks
    assert all(chunk.metadata == document.metadata for chunk in chunks)
    assert all(chunk.source_path == str(document.path) for chunk in chunks)


def test_smaller_chunk_size_produces_more_chunks(
    sample_documents: list[LoadedDocument],
) -> None:
    document = next(doc for doc in sample_documents if doc.metadata.document_id == "program_policy")

    default_chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)
    smaller_chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)

    assert len(smaller_chunks) > len(default_chunks)


def test_chunk_index_is_monotonic_per_document(
    sample_documents: list[LoadedDocument],
) -> None:
    chunks = chunk_corpus(sample_documents, chunk_size=300, chunk_overlap=50)
    indices_by_source_path: dict[str, list[int]] = defaultdict(list)

    for chunk in chunks:
        indices_by_source_path[chunk.source_path].append(chunk.chunk_index)

    assert indices_by_source_path
    for indices in indices_by_source_path.values():
        assert indices == list(range(len(indices)))


def test_chunk_document_splits_two_thousand_chars_into_about_five_chunks() -> None:
    document = LoadedDocument(
        path=Path("synthetic.md"),
        metadata=DocumentMetadata(document_id="synthetic"),
        content=("abcdefghij " * 185).strip(),
    )

    chunks = chunk_document(document, chunk_size=500, chunk_overlap=50)

    assert 4 <= len(chunks) <= 6
    assert all(len(chunk.text) <= 500 for chunk in chunks)
