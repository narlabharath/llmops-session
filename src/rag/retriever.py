"""High-level retrieval APIs for the RAG module."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from .chunking import chunk_corpus
from .embeddings import DEFAULT_EMBEDDING_MODEL, get_embeddings
from .loader import load_corpus
from .store import build_vector_store, load_vector_store, query_store
from .types import IngestionResult, RetrievedDocument

logger = logging.getLogger(__name__)


def ingest(
    corpus_dir: Path,
    persist_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> IngestionResult:
    """Load, chunk, embed, and persist a corpus to a local vector store."""

    documents = load_corpus(Path(corpus_dir))
    chunks = chunk_corpus(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = get_embeddings(model_name=embedding_model)
    store = build_vector_store(
        chunks=chunks,
        persist_directory=Path(persist_dir),
        embeddings=embeddings,
    )
    try:
        chunks_indexed = len(chunks)
        logger.info(
            "Ingested %d documents into %d chunks at %s using %s.",
            len(documents),
            chunks_indexed,
            persist_dir,
            embedding_model,
        )
        return IngestionResult(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_indexed=chunks_indexed,
            vector_store_path=Path(persist_dir),
            embedding_model=embedding_model,
        )
    finally:
        _shutdown_store(store)


def retrieve(
    persist_dir: Path,
    query: str,
    k: int = 5,
    embedding_model: str | None = None,
) -> list[RetrievedDocument]:
    """Load a persisted vector store and retrieve ranked source documents for a query."""

    model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
    embeddings = get_embeddings(model_name=model_name)
    store = load_vector_store(Path(persist_dir), embeddings=embeddings)
    try:
        candidate_count = max(k * 5, k)
        retrieved_chunks = query_store(store, query, k=candidate_count)
        retrieved_documents = _collapse_retrieved_chunks(retrieved_chunks, k=k)
        logger.info(
            "Retrieved %d documents from %s with model %s.",
            len(retrieved_documents),
            persist_dir,
            model_name,
        )
        return retrieved_documents
    finally:
        _shutdown_store(store)


def _shutdown_store(store: object) -> None:
    client = getattr(store, "_client", None)
    if client is None:
        return

    system = getattr(client, "_system", None)
    if system is not None:
        system.stop()

    clear_system_cache = getattr(client, "clear_system_cache", None)
    if callable(clear_system_cache):
        clear_system_cache()


def _collapse_retrieved_chunks(
    retrieved_chunks: list[RetrievedDocument],
    k: int,
) -> list[RetrievedDocument]:
    grouped_hits: dict[str, list[RetrievedDocument]] = {}
    for retrieved_chunk in retrieved_chunks:
        document_key = (
            retrieved_chunk.chunk.metadata.document_id
            or retrieved_chunk.chunk.source_path
            or f"rank-{retrieved_chunk.rank}"
        )
        grouped_hits.setdefault(document_key, []).append(retrieved_chunk)

    ranked_documents: list[tuple[float, int, int, RetrievedDocument]] = []
    for hits in grouped_hits.values():
        representative_chunk = max(hits, key=lambda hit: hit.score)
        combined_score = _combine_similarity_scores(hit.score for hit in hits)
        best_rank = min(hit.rank for hit in hits)
        ranked_documents.append(
            (
                combined_score,
                representative_chunk.chunk.metadata.source_priority,
                best_rank,
                representative_chunk,
            )
        )

    ranked_documents.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        RetrievedDocument(
            chunk=representative.chunk,
            score=combined_score,
            rank=rank,
        )
        for rank, (combined_score, _, _, representative) in enumerate(ranked_documents[:k])
    ]


def _combine_similarity_scores(scores: Iterable[float]) -> float:
    combined_miss_probability = 1.0
    for raw_score in scores:
        bounded_score = min(max(float(raw_score), 0.0), 1.0)
        combined_miss_probability *= 1.0 - bounded_score
    return 1.0 - combined_miss_probability


__all__ = ["ingest", "retrieve"]
