"""Groundedness judge helpers for the evaluation harness."""

from __future__ import annotations

import json
import logging
from typing import Final

from src.llm import LLMClient, load_prompt

from .types import JudgeVerdict

logger = logging.getLogger(__name__)

_PROMPT_NAME: Final[str] = "judge_groundedness"
_PROMPT_VERSION: Final[str] = "v1"


def judge_groundedness(
    question: str,
    answer: str,
    retrieved_context: str,
    judge_llm: LLMClient | None = None,
) -> JudgeVerdict:
    """Judge whether an answer is fully supported by the retrieved context."""

    prompt_template = load_prompt(_PROMPT_NAME, version=_PROMPT_VERSION)
    prompt = prompt_template.format(
        question=question,
        answer=answer,
        retrieved_context=retrieved_context,
    )
    llm_client = judge_llm or LLMClient(provider="mock", prompt_version=_PROMPT_VERSION)
    result = llm_client.complete(prompt, prompt_version=_PROMPT_VERSION)

    try:
        verdict_payload = json.loads(result.text)
        return _parse_verdict(verdict_payload, result.model)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _invalid_json_verdict(result.text, result.model)


def _parse_verdict(payload: object, model: str) -> JudgeVerdict:
    if not isinstance(payload, dict):
        raise TypeError("Judge payload must be a JSON object.")

    grounded = payload["grounded"]
    confidence = payload["confidence"]
    reason = payload["reason"]

    if not isinstance(grounded, bool):
        raise TypeError("Judge grounded flag must be a bool.")

    confidence_value = float(confidence)
    reason_text = str(reason).strip()
    if not reason_text:
        raise ValueError("Judge reason must be non-empty.")

    return JudgeVerdict(
        grounded=grounded,
        confidence=confidence_value,
        reason=reason_text,
        model=model,
    )


def _invalid_json_verdict(raw_output: str, model: str) -> JudgeVerdict:
    preview = raw_output[:80]
    logger.warning(
        "Groundedness judge returned invalid JSON; using fallback verdict. raw_output=%r",
        raw_output,
    )
    return JudgeVerdict(
        grounded=False,
        confidence=0.0,
        reason=f"judge returned invalid JSON: {preview}",
        model=model,
    )


__all__ = ["judge_groundedness"]
