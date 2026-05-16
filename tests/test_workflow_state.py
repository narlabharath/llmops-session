from __future__ import annotations

from typing import NotRequired, Required, get_origin, get_type_hints

from src.workflow.state import WorkflowState
from src.workflow.types import WorkflowResult


def test_workflow_state_tracks_required_and_optional_keys() -> None:
    state: WorkflowState = {"question": "When is the assignment due?"}
    hints = get_type_hints(WorkflowState, include_extras=True)

    assert state["question"] == "When is the assignment due?"
    assert get_origin(hints["question"]) is Required
    assert get_origin(hints["classification"]) is NotRequired
    assert get_origin(hints["retrieved_docs"]) is NotRequired
    assert get_origin(hints["answer"]) is NotRequired
    assert get_origin(hints["refusal_reason"]) is NotRequired
    assert get_origin(hints["escalation_reason"]) is NotRequired
    assert get_origin(hints["trace"]) is NotRequired


def test_workflow_result_defaults_to_empty_collections() -> None:
    result = WorkflowResult(
        outcome="answered",
        text="Assignment 2 is due Friday.",
        classification="schedule_question",
    )

    assert result.retrieved_doc_ids == []
    assert result.trace == []
