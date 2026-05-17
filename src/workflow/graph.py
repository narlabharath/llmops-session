"""Workflow graph assembly."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal, cast

from src.llm import LLMClient
from src.rag.types import RetrievedDocument

from .nodes import classify_question, escalate, generate_answer, refuse, retrieve_documents
from .state import WorkflowState
from .types import WorkflowResult

logger = logging.getLogger(__name__)

_QUESTION_CLASSIFICATIONS: Final[set[str]] = {
    "policy_question",
    "schedule_question",
    "assignment_question",
}
_CLASSIFY_NODE: Final[str] = "classify_node"
_RETRIEVE_NODE: Final[str] = "retrieve_node"
_ANSWER_NODE: Final[str] = "answer_node"
_REFUSE_NODE: Final[str] = "refuse_node"
_ESCALATE_NODE: Final[str] = "escalate_node"


def build_workflow_graph(
    persist_dir: Path,
    llm: LLMClient | None = None,
    k: int = 5,
    escalation_threshold: float = 0.3,
) -> CompiledStateGraph:
    """Build and compile the LangGraph workflow."""

    from langgraph.graph import END, START, StateGraph

    resolved_persist_dir = Path(persist_dir)

    graph = StateGraph(WorkflowState)
    graph.add_node(_CLASSIFY_NODE, lambda state: classify_question(state, llm=llm))
    graph.add_node(
        _RETRIEVE_NODE,
        lambda state: retrieve_documents(
            state,
            persist_dir=resolved_persist_dir,
            k=k,
        ),
    )
    graph.add_node(_ANSWER_NODE, lambda state: generate_answer(state, llm=llm))
    graph.add_node(_REFUSE_NODE, refuse)
    graph.add_node(_ESCALATE_NODE, escalate)

    graph.add_edge(START, _CLASSIFY_NODE)
    graph.add_conditional_edges(
        _CLASSIFY_NODE,
        _route_after_classify,
        {
            "retrieve": _RETRIEVE_NODE,
            "refuse": _REFUSE_NODE,
        },
    )
    graph.add_conditional_edges(
        _RETRIEVE_NODE,
        lambda state: _route_after_retrieve(
            state,
            escalation_threshold=escalation_threshold,
        ),
        {
            "answer": _ANSWER_NODE,
            "escalate": _ESCALATE_NODE,
        },
    )
    graph.add_edge(_ANSWER_NODE, END)
    graph.add_edge(_REFUSE_NODE, END)
    graph.add_edge(_ESCALATE_NODE, END)

    return graph.compile()


def run_workflow(
    question: str,
    persist_dir: Path,
    llm: LLMClient | None = None,
    k: int = 5,
    escalation_threshold: float = 0.3,
) -> WorkflowResult:
    """Run the workflow graph for a single participant question."""

    compiled_graph = build_workflow_graph(
        persist_dir=Path(persist_dir),
        llm=llm,
        k=k,
        escalation_threshold=escalation_threshold,
    )
    final_state = cast(
        WorkflowState,
        compiled_graph.invoke({"question": question, "trace": []}),
    )
    classification = str(final_state.get("classification", ""))
    retrieved_docs = final_state.get("retrieved_docs", [])
    trace = list(final_state.get("trace", []))

    if "answer" in final_state:
        outcome: Literal["answered", "refused", "escalated"] = "answered"
        text = str(final_state["answer"])
    elif "refusal_reason" in final_state:
        outcome = "refused"
        text = str(final_state["refusal_reason"])
    elif "escalation_reason" in final_state:
        outcome = "escalated"
        text = str(final_state["escalation_reason"])
    else:
        raise ValueError("Workflow completed without a terminal output in final state.")

    return WorkflowResult(
        outcome=outcome,
        text=text,
        classification=classification,
        retrieved_doc_ids=_get_retrieved_doc_ids(retrieved_docs),
        trace=trace,
    )


def _route_after_classify(state: WorkflowState) -> Literal["retrieve", "refuse"]:
    classification = str(state.get("classification", ""))
    next_node: Literal["retrieve", "refuse"]
    if classification in _QUESTION_CLASSIFICATIONS:
        next_node = "retrieve"
    else:
        next_node = "refuse"

    _annotate_latest_trace_entry(state, route_to=next_node)
    return next_node


def _route_after_retrieve(
    state: WorkflowState,
    escalation_threshold: float,
) -> Literal["answer", "escalate"]:
    retrieved_docs = state.get("retrieved_docs", [])
    max_score = max((document.score for document in retrieved_docs), default=None)

    if not retrieved_docs:
        next_node: Literal["answer", "escalate"] = "escalate"
        routing_reason = "no_documents"
    elif max_score is None or max_score < escalation_threshold:
        next_node = "escalate"
        routing_reason = "low_score"
    else:
        next_node = "answer"
        routing_reason = "sufficient_score"

    _annotate_latest_trace_entry(
        state,
        route_to=next_node,
        routing_reason=routing_reason,
        escalation_threshold=escalation_threshold,
        max_score=max_score,
    )
    return next_node


def _annotate_latest_trace_entry(state: WorkflowState, **fields: object) -> None:
    trace = state.get("trace")
    if not trace:
        logger.warning("Workflow routing ran without an existing trace entry to annotate.")
        return

    trace[-1].update(fields)


def _get_retrieved_doc_ids(retrieved_docs: list[RetrievedDocument]) -> list[str]:
    return [_get_retrieved_doc_id(document) for document in retrieved_docs]


def _get_retrieved_doc_id(document: RetrievedDocument) -> str:
    return (
        document.chunk.metadata.document_id
        or document.chunk.source_path
        or f"rank-{document.rank}"
    )


__all__ = ["build_workflow_graph", "run_workflow"]
