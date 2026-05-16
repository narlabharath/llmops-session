"""State schema for workflow orchestration."""

from __future__ import annotations

from typing import NotRequired, Required, TypedDict

from src.rag.types import RetrievedDocument

type TraceEntry = dict[str, object]


class WorkflowState(TypedDict, total=False):
    """State shared across workflow nodes."""

    question: Required[str]
    classification: NotRequired[str]
    retrieved_docs: NotRequired[list[RetrievedDocument]]
    answer: NotRequired[str]
    refusal_reason: NotRequired[str]
    escalation_reason: NotRequired[str]
    trace: NotRequired[list[TraceEntry]]


__all__ = ["TraceEntry", "WorkflowState"]
