"""Workflow node implementations."""

from __future__ import annotations

import logging
from typing import Final

from src.llm import LLMClient, load_prompt

from .state import TraceEntry, WorkflowState

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM_PROMPT: Final[str] = (
    "You classify TalentSprint program-assistant questions. "
    "Reply with exactly one category label."
)
_PROMPT_VERSION: Final[str] = "v1"
_VALID_CLASSIFICATIONS: Final[set[str]] = {
    "policy_question",
    "schedule_question",
    "assignment_question",
    "out_of_scope",
    "injection_attempt",
    "private_request",
}
_FALLBACK_CLASSIFICATION: Final[str] = "out_of_scope"


def classify_question(state: WorkflowState, llm: LLMClient | None = None) -> dict[str, object]:
    """Classify the participant question into a workflow routing category."""

    question = state["question"]
    template = load_prompt("classify_question", version=_PROMPT_VERSION)
    prompt = template.format(question=question)
    llm_client = llm or LLMClient(provider="mock", prompt_version=_PROMPT_VERSION)
    result = llm_client.complete(prompt, system=_CLASSIFIER_SYSTEM_PROMPT)
    classification = _parse_classification(result.text)
    trace_entry: TraceEntry = {
        "node": "classify",
        "input": question,
        "output": classification,
        "latency_ms": result.latency_ms,
        "prompt_version": _PROMPT_VERSION,
    }
    trace = [*state.get("trace", []), trace_entry]
    return {"classification": classification, "trace": trace}


def _parse_classification(raw_output: str) -> str:
    normalized = raw_output.strip().lower()
    if normalized in _VALID_CLASSIFICATIONS:
        return normalized

    logger.warning(
        "Unexpected classification returned by LLM; falling back to %s. raw_output=%r",
        _FALLBACK_CLASSIFICATION,
        raw_output,
    )
    return _FALLBACK_CLASSIFICATION


__all__ = ["classify_question"]
