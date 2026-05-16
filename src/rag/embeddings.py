"""Embedding model factories for the RAG module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_cache: dict[str, Embeddings] = {}


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    """Return a cached LangChain embeddings instance for the requested model."""

    cached_embeddings = _cache.get(model_name)
    if cached_embeddings is not None:
        logger.debug("Reusing cached embeddings model %s.", model_name)
        return cached_embeddings

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    _cache[model_name] = embeddings
    logger.info("Loaded embeddings model %s.", model_name)
    return embeddings


__all__ = ["get_embeddings"]
