"""Scoring helpers for the evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import GoldenRow


class SupportsQueryEmbedding(Protocol):
    """Minimal embedding protocol used by similarity scoring."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Return a vector embedding for a single text string."""


def score_behavior(row: GoldenRow, workflow_outcome: str) -> tuple[bool, list[str]]:
    """Score whether the workflow outcome matches the row's behavioral flags."""

    normalized_outcome = workflow_outcome.strip().lower() or "unknown"
    expected_outcomes = _expected_outcomes(row)

    if normalized_outcome in expected_outcomes:
        return True, []

    failures = [
        f"expected {expected_label}, got {normalized_outcome}"
        for expected_label in _expected_labels(row)
    ]
    return False, failures


def score_similarity(
    workflow_text: str,
    reference_answer: str | None,
    embeddings: SupportsQueryEmbedding,
) -> float | None:
    """Return cosine similarity between the workflow text and the reference answer."""

    if reference_answer is None or not reference_answer.strip():
        return None

    import numpy as np

    workflow_vector = np.asarray(embeddings.embed_query(workflow_text), dtype=float)
    reference_vector = np.asarray(embeddings.embed_query(reference_answer), dtype=float)

    workflow_norm = float(np.linalg.norm(workflow_vector))
    reference_norm = float(np.linalg.norm(reference_vector))
    if workflow_norm == 0.0 or reference_norm == 0.0:
        return 0.0

    similarity = float(
        np.dot(workflow_vector, reference_vector) / (workflow_norm * reference_norm)
    )
    return similarity


def _expected_outcomes(row: GoldenRow) -> set[str]:
    outcomes: set[str] = set()
    if row.should_refuse:
        outcomes.add("refused")
    if row.should_escalate:
        outcomes.add("escalated")
    if not outcomes:
        outcomes.add("answered")
    return outcomes


def _expected_labels(row: GoldenRow) -> list[str]:
    labels: list[str] = []
    if row.should_refuse:
        labels.append("refusal")
    if row.should_escalate:
        labels.append("escalation")
    if not labels:
        labels.append("answer")
    return labels


__all__ = ["score_behavior", "score_similarity"]
