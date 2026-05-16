"""Vector store utilities for the RAG module."""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .embeddings import get_embeddings
from .types import Chunk, DEFAULT_SOURCE_PRIORITY, DocumentMetadata, RetrievedDocument

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rag"


def build_vector_store(
    chunks: list[Chunk],
    persist_directory: Path,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Build and persist a Chroma vector store for the provided chunks."""

    persist_directory.mkdir(parents=True, exist_ok=True)
    chroma_class = _get_chroma_class()
    resolved_embeddings = embeddings or get_embeddings()

    store = chroma_class(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=resolved_embeddings,
    )
    store.reset_collection()

    documents = [_chunk_to_langchain_document(chunk) for chunk in chunks]
    if documents:
        store.add_documents(documents)

    logger.info(
        "Built vector store at %s with %d chunks.",
        persist_directory,
        len(chunks),
    )
    return store


def load_vector_store(
    persist_directory: Path,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Load a previously persisted Chroma vector store."""

    if not persist_directory.exists() or not any(persist_directory.iterdir()):
        raise FileNotFoundError(f"No persisted vector store found at {persist_directory}.")

    chroma_class = _get_chroma_class()
    resolved_embeddings = embeddings or get_embeddings()
    store = chroma_class(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=resolved_embeddings,
    )

    logger.info("Loaded vector store from %s.", persist_directory)
    return store


def query_store(store: Chroma, query: str, k: int = 5) -> list[RetrievedDocument]:
    """Query a Chroma store and return ranked retrieved documents."""

    search_results = store.similarity_search_with_score(query, k=k)
    retrieved_documents = [
        _build_retrieved_document(document=document, distance=distance, rank=rank)
        for rank, (document, distance) in enumerate(search_results)
    ]
    logger.info("Retrieved %d chunks for query %r.", len(retrieved_documents), query)
    return retrieved_documents


def _get_chroma_class() -> type[Chroma]:
    from langchain_chroma import Chroma

    return Chroma


def _chunk_to_langchain_document(chunk: Chunk) -> Document:
    from langchain_core.documents import Document

    return Document(
        page_content=chunk.text,
        metadata=_serialize_chunk_metadata(chunk),
    )


def _serialize_chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    metadata = {
        key: value
        for key, value in asdict(chunk.metadata).items()
        if value is not None
    }
    metadata["chunk_index"] = chunk.chunk_index
    metadata["source_path"] = chunk.source_path
    return metadata


def _build_retrieved_document(
    document: Document,
    distance: float,
    rank: int,
) -> RetrievedDocument:
    metadata = document.metadata
    chunk = Chunk(
        text=document.page_content,
        metadata=_deserialize_document_metadata(metadata),
        chunk_index=_parse_int(metadata.get("chunk_index"), default=0),
        source_path=_parse_str(metadata.get("source_path"), default=""),
    )
    # Chroma reports distances where lower is more similar, so convert to a
    # bounded similarity score for the public API.
    score = 1.0 / (1.0 + max(distance, 0.0))
    return RetrievedDocument(chunk=chunk, score=score, rank=rank)


def _deserialize_document_metadata(raw_metadata: dict[str, Any]) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=_parse_optional_str(raw_metadata.get("document_id")),
        title=_parse_optional_str(raw_metadata.get("title")),
        document_type=_parse_optional_str(raw_metadata.get("document_type")),
        version=_parse_optional_str(raw_metadata.get("version")),
        effective_date=_parse_optional_str(raw_metadata.get("effective_date")),
        owner=_parse_optional_str(raw_metadata.get("owner")),
        source_priority=_parse_int(
            raw_metadata.get("source_priority"),
            default=DEFAULT_SOURCE_PRIORITY,
        ),
    )


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _parse_int(value: Any, default: int) -> int:
    try:
        if value is None:
            raise TypeError("Missing integer metadata value.")
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Falling back to default integer metadata value %d.", default)
        return default


__all__ = ["build_vector_store", "load_vector_store", "query_store"]
