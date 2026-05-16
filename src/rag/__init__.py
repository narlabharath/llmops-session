"""RAG module public API."""

from __future__ import annotations

from .chunking import chunk_corpus, chunk_document
from .embeddings import get_embeddings
from .loader import load_corpus, load_document
from .retriever import ingest, retrieve
from .store import build_vector_store, load_vector_store, query_store
from .types import Chunk, DocumentMetadata, IngestionResult, LoadedDocument, RetrievedDocument

__all__ = [
    "DocumentMetadata",
    "LoadedDocument",
    "Chunk",
    "RetrievedDocument",
    "IngestionResult",
    "load_document",
    "load_corpus",
    "chunk_document",
    "chunk_corpus",
    "get_embeddings",
    "build_vector_store",
    "load_vector_store",
    "query_store",
    "ingest",
    "retrieve",
]
