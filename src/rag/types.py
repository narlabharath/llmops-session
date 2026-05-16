"""Shared dataclasses for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE_PRIORITY = 99


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str | None = None
    title: str | None = None
    document_type: str | None = None
    version: str | None = None
    effective_date: str | None = None
    owner: str | None = None
    source_priority: int = DEFAULT_SOURCE_PRIORITY


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    metadata: DocumentMetadata
    content: str


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: DocumentMetadata
    chunk_index: int
    source_path: str


@dataclass(frozen=True)
class RetrievedDocument:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class IngestionResult:
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    vector_store_path: Path
    embedding_model: str


__all__ = [
    "DEFAULT_SOURCE_PRIORITY",
    "DocumentMetadata",
    "LoadedDocument",
    "Chunk",
    "RetrievedDocument",
    "IngestionResult",
]
