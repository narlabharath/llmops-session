"""Shared workflow dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .state import TraceEntry


@dataclass(frozen=True)
class WorkflowResult:
    """Normalized workflow output returned to callers."""

    outcome: Literal["answered", "refused", "escalated"]
    text: str
    classification: str
    retrieved_doc_ids: list[str] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)


__all__ = ["WorkflowResult"]
