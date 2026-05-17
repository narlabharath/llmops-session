from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.evals import GoldenRow, score_behavior, score_similarity
from src.rag import get_embeddings


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self._vectors = vectors

    def embed_query(self, text: str) -> Sequence[float]:
        return self._vectors[text]


def test_score_behavior_matches_answered_outcome() -> None:
    row = _make_row()

    passed, failures = score_behavior(row, "answered")

    assert passed is True
    assert failures == []


def test_score_behavior_rejects_answered_when_refusal_is_expected() -> None:
    row = _make_row(should_refuse=True)

    passed, failures = score_behavior(row, "answered")

    assert passed is False
    assert failures == ["expected refusal, got answered"]


def test_score_behavior_rejects_refused_when_escalation_is_expected() -> None:
    row = _make_row(should_escalate=True)

    passed, failures = score_behavior(row, "refused")

    assert passed is False
    assert failures == ["expected escalation, got refused"]


def test_score_behavior_rejects_escalated_when_answer_is_expected() -> None:
    row = _make_row()

    passed, failures = score_behavior(row, "escalated")

    assert passed is False
    assert failures == ["expected answer, got escalated"]


def test_score_similarity_returns_none_without_reference_answer() -> None:
    embeddings = FakeEmbeddings({"workflow": [1.0, 0.0]})

    similarity = score_similarity("workflow", None, embeddings)

    assert similarity is None


def test_score_similarity_returns_one_for_identical_vectors() -> None:
    embeddings = FakeEmbeddings(
        {
            "same answer": [1.0, 2.0, 3.0],
        }
    )

    similarity = score_similarity("same answer", "same answer", embeddings)

    assert similarity is not None
    assert similarity == pytest.approx(1.0)


def test_score_similarity_returns_low_score_for_unrelated_vectors() -> None:
    embeddings = FakeEmbeddings(
        {
            "workflow": [1.0, 0.0, 0.0],
            "reference": [0.0, 1.0, 0.0],
        }
    )

    similarity = score_similarity("workflow", "reference", embeddings)

    assert similarity is not None
    assert similarity < 0.5


@pytest.mark.slow
def test_score_similarity_with_real_embeddings_for_identical_strings() -> None:
    embeddings = get_embeddings()

    similarity = score_similarity(
        "Late submissions can be accepted with a note within 48 hours.",
        "Late submissions can be accepted with a note within 48 hours.",
        embeddings,
    )

    assert similarity is not None
    assert similarity == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def _make_row(
    *,
    should_refuse: bool = False,
    should_escalate: bool = False,
) -> GoldenRow:
    return GoldenRow(
        id="GQ999",
        query="Test query",
        category="policy",
        expected_behavior="answer",
        reference_answer=None,
        required_context=None,
        risk_level="low",
        should_retrieve=True,
        should_refuse=should_refuse,
        should_escalate=should_escalate,
        notes=None,
    )
