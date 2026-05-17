"""Pure-Python output guardrails that run before an answer is shown."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final, Literal

from .types import GuardrailDecision, GuardrailReport

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT_OVERLAP_CHARS: Final[int] = 50


@dataclass(frozen=True)
class _PIIRule:
    pattern: re.Pattern[str]
    pii_type: str
    reason: str


_PII_RULES = (
    _PIIRule(
        pattern=re.compile(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        pii_type="email_address",
        reason="output appears to contain an email address",
    ),
    _PIIRule(
        pattern=re.compile(r"(?<!\d)(?:\+91[-.\s]?)?[6-9]\d{4}[-.\s]?\d{5}(?!\d)"),
        pii_type="phone_number_india",
        reason="output appears to contain an Indian phone number",
    ),
    _PIIRule(
        pattern=re.compile(r"(?<!\w)\+\d{1,3}(?:[-.\s]?\d){8,14}(?!\w)"),
        pii_type="phone_number_international",
        reason="output appears to contain an international phone number",
    ),
    _PIIRule(
        pattern=re.compile(r"(?<!\d)\d{10,16}(?!\d)"),
        pii_type="account_like_number",
        reason="output appears to contain a long personal identifier or account number",
    ),
    _PIIRule(
        pattern=re.compile(
            r"\b[A-Z][a-z]+ [A-Z][a-z]+'s "
            r"(?:email|phone(?: number)?|contact details|grade|grades|marks|"
            r"attendance|address|aadhaar|account(?: number)?)\b"
        ),
        pii_type="name_with_personal_marker",
        reason="output appears to pair a person's full name with personal information",
    ),
    _PIIRule(
        pattern=re.compile(
            r"\b(?:email|phone(?: number)?|contact details|grade|grades|marks|"
            r"attendance|address|aadhaar|account(?: number)?)\s*(?::|for|of)\s*"
            r"[A-Z][a-z]+ [A-Z][a-z]+\b"
        ),
        pii_type="personal_marker_with_name",
        reason="output appears to disclose personal information alongside a full name",
    ),
)


def check_pii(text: str) -> GuardrailDecision:
    """Detect obvious PII leaks in model output."""

    for rule in _PII_RULES:
        match = rule.pattern.search(text)
        if match is None:
            continue

        matched_text = match.group(0)
        LOGGER.warning("Blocking output due to PII match: %s", rule.pii_type)
        return GuardrailDecision(
            passed=False,
            guardrail="output.pii",
            severity="block",
            reason=rule.reason,
            matched_pattern=rule.pattern.pattern,
            metadata={
                "pii_type": rule.pii_type,
                "matched_text": matched_text,
                "match_span": [match.start(), match.end()],
            },
        )

    return GuardrailDecision(
        passed=True,
        guardrail="output.pii",
        severity="info",
        reason="no obvious PII detected in output",
        matched_pattern=None,
        metadata={},
    )


def check_length(text: str, max_chars: int = 2000) -> GuardrailDecision:
    """Warn on oversized output and block extremely large output."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    char_count = len(text)
    block_threshold = max_chars * 2

    if char_count >= block_threshold:
        LOGGER.warning(
            "Blocking oversized output: %d chars exceeds hard limit %d.",
            char_count,
            block_threshold,
        )
        return GuardrailDecision(
            passed=False,
            guardrail="output.length",
            severity="block",
            reason=f"output length {char_count} exceeds hard limit {block_threshold}",
            matched_pattern=None,
            metadata={
                "char_count": char_count,
                "max_chars": max_chars,
                "block_threshold": block_threshold,
            },
        )

    if char_count >= max_chars:
        LOGGER.warning(
            "Warning on oversized output: %d chars exceeds soft limit %d.",
            char_count,
            max_chars,
        )
        return GuardrailDecision(
            passed=False,
            guardrail="output.length",
            severity="warn",
            reason=f"output length {char_count} exceeds soft limit {max_chars}",
            matched_pattern=None,
            metadata={
                "char_count": char_count,
                "max_chars": max_chars,
                "block_threshold": block_threshold,
            },
        )

    return GuardrailDecision(
        passed=True,
        guardrail="output.length",
        severity="info",
        reason="output length is within limits",
        matched_pattern=None,
        metadata={
            "char_count": char_count,
            "max_chars": max_chars,
            "block_threshold": block_threshold,
        },
    )


def check_system_prompt_leak(
    text: str,
    system_prompt: str | None = None,
) -> GuardrailDecision:
    """Detect whether output appears to reproduce a system prompt."""

    if system_prompt is None:
        return GuardrailDecision(
            passed=True,
            guardrail="output.system_prompt_leak",
            severity="info",
            reason="no system prompt was provided for leak detection",
            matched_pattern=None,
            metadata={"overlap_chars": 0},
        )

    normalized_output = _normalize_for_overlap(text)
    normalized_prompt = _normalize_for_overlap(system_prompt)

    if len(normalized_prompt) < _SYSTEM_PROMPT_OVERLAP_CHARS:
        return GuardrailDecision(
            passed=True,
            guardrail="output.system_prompt_leak",
            severity="info",
            reason="system prompt too short for overlap-based leak detection",
            matched_pattern=None,
            metadata={
                "overlap_chars": 0,
                "required_overlap_chars": _SYSTEM_PROMPT_OVERLAP_CHARS,
            },
        )

    overlap = _find_prompt_overlap(normalized_output, normalized_prompt)
    if overlap is None:
        return GuardrailDecision(
            passed=True,
            guardrail="output.system_prompt_leak",
            severity="info",
            reason="no system prompt leak detected",
            matched_pattern=None,
            metadata={"overlap_chars": 0},
        )

    LOGGER.warning(
        "Blocking output because it overlaps the system prompt by %d chars.",
        len(overlap),
    )
    return GuardrailDecision(
        passed=False,
        guardrail="output.system_prompt_leak",
        severity="block",
        reason="output appears to reproduce the system prompt",
        matched_pattern=overlap,
        metadata={
            "overlap_chars": len(overlap),
            "required_overlap_chars": _SYSTEM_PROMPT_OVERLAP_CHARS,
        },
    )


def check_output(text: str, system_prompt: str | None = None) -> GuardrailReport:
    """Run the LC-33 output guardrails and return a composed report."""

    decisions = [
        check_pii(text),
        check_length(text),
        check_system_prompt_leak(text, system_prompt=system_prompt),
    ]

    blocked_by = [
        decision.guardrail
        for decision in decisions
        if not decision.passed and decision.severity == "block"
    ]

    if blocked_by:
        overall: Literal["pass", "warn", "block"] = "block"
    elif any(not decision.passed and decision.severity == "warn" for decision in decisions):
        overall = "warn"
    else:
        overall = "pass"

    return GuardrailReport(decisions=decisions, overall=overall, blocked_by=blocked_by)


def _normalize_for_overlap(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_prompt_overlap(output_text: str, system_prompt: str) -> str | None:
    for start in range(0, len(system_prompt) - _SYSTEM_PROMPT_OVERLAP_CHARS + 1):
        snippet = system_prompt[start : start + _SYSTEM_PROMPT_OVERLAP_CHARS]
        if snippet in output_text:
            return snippet

    return None


__all__ = [
    "check_length",
    "check_output",
    "check_pii",
    "check_system_prompt_leak",
]
