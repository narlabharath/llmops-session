"""RAG module public API."""

from __future__ import annotations

from .loader import load_corpus, load_document
from .types import Chunk, DocumentMetadata, IngestionResult, LoadedDocument, RetrievedDocument

__all__ = [
    "DocumentMetadata",
    "LoadedDocument",
    "Chunk",
    "RetrievedDocument",
    "IngestionResult",
    "load_document",
    "load_corpus",
]
