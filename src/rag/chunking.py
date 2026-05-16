"""Chunking utilities for the RAG module."""

from __future__ import annotations

import logging

from .types import Chunk, LoadedDocument

logger = logging.getLogger(__name__)


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a loaded document into chunk dataclasses."""

    splitter = _build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_texts = splitter.split_text(doc.content)
    chunks = [
        Chunk(
            text=chunk_text,
            metadata=doc.metadata,
            chunk_index=chunk_index,
            source_path=str(doc.path),
        )
        for chunk_index, chunk_text in enumerate(chunk_texts)
    ]
    logger.info("Chunked %s into %d chunks.", doc.path, len(chunks))
    return chunks


def chunk_corpus(
    docs: list[LoadedDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a corpus of loaded documents into chunk dataclasses."""

    chunks: list[Chunk] = []
    for document in docs:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    logger.info("Chunked %d documents into %d chunks.", len(docs), len(chunks))
    return chunks


def _build_text_splitter(chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


__all__ = ["chunk_document", "chunk_corpus"]
