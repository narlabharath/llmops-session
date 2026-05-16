from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

import pytest

import src.rag.embeddings as embeddings_module
from src.rag import get_embeddings


@pytest.fixture(autouse=True)
def clear_embeddings_cache() -> None:
    embeddings_module._cache.clear()
    yield
    embeddings_module._cache.clear()


def test_importing_embeddings_module_is_lazy() -> None:
    sys.modules.pop("src.rag.embeddings", None)
    sys.modules.pop("langchain_huggingface", None)

    imported_module = importlib.import_module("src.rag.embeddings")

    assert imported_module.__name__ == "src.rag.embeddings"
    assert "langchain_huggingface" not in sys.modules


def test_get_embeddings_returns_object_with_embed_query(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("langchain_huggingface")

    class FakeEmbeddings:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed_query(self, query: str) -> list[float]:
            return [float(len(query))]

    fake_module.HuggingFaceEmbeddings = FakeEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_module)

    embeddings = get_embeddings()

    assert hasattr(embeddings, "embed_query")
    assert embeddings.model_name == embeddings_module.DEFAULT_EMBEDDING_MODEL
    assert embeddings.embed_query("policy") == [6.0]


def test_get_embeddings_reuses_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("langchain_huggingface")

    class FakeEmbeddings:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed_query(self, query: str) -> list[float]:
            return [1.0]

    fake_module.HuggingFaceEmbeddings = FakeEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_module)

    first_embeddings = get_embeddings()
    second_embeddings = get_embeddings()

    assert first_embeddings is second_embeddings


@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") != "1", reason="Set RUN_SLOW_TESTS=1 to run embedding download tests.")
def test_get_embeddings_can_embed_query() -> None:
    embeddings = get_embeddings()
    vector = embeddings.embed_query("late submission policy")

    assert len(vector) == 384
