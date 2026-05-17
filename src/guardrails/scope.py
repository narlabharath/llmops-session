"""LLM-based scope guardrail for out-of-scope detection."""

from __future__ import annotations

import json
import logging
from typing import Final

from src.llm import LLMClient, load_prompt

from .types import GuardrailDecision

LOGGER = logging.getLogger(__name__)

_GUARDRAIL_NAME: Final[str] = "scope.llm_judge"
_PROMPT_NAME: Final[str] = "scope_check"
_PROMPT_VERSION: Final[str] = "v1"
_DEFAULT_PROGRAM_SUMMARY: Final[str] = (
    "The TalentSprint AI engineering program teaches learners how to build "
    "practical LLM-powered applications using Python, prompt design, "
    "retrieval-augmented generation, evaluation, and guardrails. "
    "Students work in a shared teaching repo with notebooks, tests, and "
    "lightweight app components to practice implementation, debugging, and "
    "iteration. "
    "The assistant should help with course concepts, assignments, code, and "
    "project work that directly relates to the program curriculum. "
    "Requests unrelated to the program's AI engineering focus should be "
    "treated as out of scope."
)


def check_scope(
    question: str,
    llm: LLMClient | None = None,
    program_summary: str | None = None,
) -> GuardrailDecision:
    """Judge whether a question is within the program's scope."""

    prompt_template = load_prompt(_PROMPT_NAME, version=_PROMPT_VERSION)
    prompt = prompt_template.format(
        question=question,
        program_summary=_DEFAULT_PROGRAM_SUMMARY
        if program_summary is None
        else program_summary,
    )
    llm_client = llm or LLMClient(provider="mock", prompt_version=_PROMPT_VERSION)
    result = llm_client.complete(prompt, prompt_version=_PROMPT_VERSION)

    try:
        payload = json.loads(result.text)
        return _parse_scope_decision(payload, result.model)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _invalid_scope_decision(result.text, result.model)


def _parse_scope_decision(payload: object, model: str) -> GuardrailDecision:
    if not isinstance(payload, dict):
        raise TypeError("Scope payload must be a JSON object.")

    in_scope = payload["in_scope"]
    confidence = payload["confidence"]
    reason = payload["reason"]

    if not isinstance(in_scope, bool):
        raise TypeError("Scope in_scope flag must be a bool.")
    if isinstance(confidence, bool):
        raise TypeError("Scope confidence must be numeric, not bool.")

    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("Scope confidence must be between 0.0 and 1.0.")

    reason_text = str(reason).strip()
    if not reason_text:
        raise ValueError("Scope reason must be non-empty.")

    if in_scope:
        return GuardrailDecision(
            passed=True,
            guardrail=_GUARDRAIL_NAME,
            severity="info",
            reason=reason_text,
            matched_pattern=None,
            metadata={
                "in_scope": True,
                "confidence": confidence_value,
                "model": model,
            },
        )

    severity = "block" if confidence_value > 0.6 else "warn"
    return GuardrailDecision(
        passed=False,
        guardrail=_GUARDRAIL_NAME,
        severity=severity,
        reason=reason_text,
        matched_pattern=None,
        metadata={
            "in_scope": False,
            "confidence": confidence_value,
            "model": model,
        },
    )


def _invalid_scope_decision(raw_output: str, model: str) -> GuardrailDecision:
    preview = raw_output[:80]
    LOGGER.warning(
        "Scope judge returned invalid JSON; using fail-open warning. raw_output=%r",
        raw_output,
    )
    return GuardrailDecision(
        passed=True,
        guardrail=_GUARDRAIL_NAME,
        severity="warn",
        reason="scope judge returned invalid JSON",
        matched_pattern=None,
        metadata={
            "in_scope": None,
            "confidence": None,
            "model": model,
            "raw_output_preview": preview,
        },
    )


__all__ = ["check_scope"]
