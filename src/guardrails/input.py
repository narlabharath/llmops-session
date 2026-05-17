"""Pure-Python input guardrails that run before any LLM call."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from .types import GuardrailDecision, GuardrailReport

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InjectionRule:
    pattern: re.Pattern[str]
    severity: Literal["info", "block"]
    reason: str
    intent: str


_INJECTION_RULES = (
    # Direct overrides attempt to replace the model's operating instructions.
    _InjectionRule(
        pattern=re.compile(
            r"\bignore (?:all )?(?:prior|previous|above) "
            r"(?:instructions|prompts|directions|rules)\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input attempts to override prior instructions",
        intent="instruction override",
    ),
    # Variants using "disregard" aim at the same control-flow break.
    _InjectionRule(
        pattern=re.compile(
            r"\bdisregard (?:all )?(?:prior|previous|above)"
            r"(?: (?:instructions|prompts|directions|rules))?\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input asks the assistant to disregard earlier instructions",
        intent="instruction override",
    ),
    # Mode-switch prompts try to move the assistant into an unsafe persona.
    _InjectionRule(
        pattern=re.compile(
            r"\byou are now (?:in )?(?:unrestricted|jailbroken|dan|developer) mode\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input attempts to switch the assistant into an unrestricted mode",
        intent="mode switch",
    ),
    # Printing hidden instructions is another direct exfiltration pattern.
    _InjectionRule(
        pattern=re.compile(
            r"\bprint (?:your|the) (?:system|hidden|secret) "
            r"(?:prompt|instructions)\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input asks for hidden instructions to be printed",
        intent="system prompt exfiltration",
    ),
    # "Reveal/show/dump" variants target the same hidden context.
    _InjectionRule(
        pattern=re.compile(
            r"\b(?:reveal|show|display|dump) (?:your|the) "
            r"(?:system|hidden|secret) (?:prompt|instructions)\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input asks for hidden instructions to be revealed",
        intent="system prompt exfiltration",
    ),
    # Explicit requests for the system prompt are high-signal exfiltration attempts.
    _InjectionRule(
        pattern=re.compile(r"\bsystem prompt\b", re.IGNORECASE),
        severity="block",
        reason="input references the system prompt directly",
        intent="system prompt exfiltration",
    ),
    # Bypass language targets safety layers explicitly.
    _InjectionRule(
        pattern=re.compile(
            r"\bbypass (?:the )?(?:safety|guardrails|filters|restrictions)\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input explicitly asks to bypass safety controls",
        intent="safety bypass",
    ),
    # Requests to stop following earlier rules are direct prompt-injection attempts.
    _InjectionRule(
        pattern=re.compile(
            r"\bdo not follow (?:the )?(?:previous|prior|above) "
            r"(?:instructions|rules|directions)\b",
            re.IGNORECASE,
        ),
        severity="block",
        reason="input asks the assistant to stop following prior rules",
        intent="instruction override",
    ),
    # Roleplay can be benign, so the "pretend" pattern is logged as informational only.
    _InjectionRule(
        pattern=re.compile(r"\bpretend (?:that )?(?:the |you are )\b", re.IGNORECASE),
        severity="info",
        reason="input uses borderline roleplay phrasing that can correlate with jailbreak attempts",
        intent="borderline roleplay",
    ),
)


def check_injection_patterns(text: str) -> GuardrailDecision:
    """Detect regex-based prompt injection patterns in user input."""

    informational_match: tuple[_InjectionRule, re.Match[str]] | None = None

    for rule in _INJECTION_RULES:
        match = rule.pattern.search(text)
        if match is None:
            continue

        if rule.severity == "block":
            LOGGER.warning("Blocking input due to injection pattern: %s", rule.intent)
            return GuardrailDecision(
                passed=False,
                guardrail="input.injection_regex",
                severity="block",
                reason=rule.reason,
                matched_pattern=rule.pattern.pattern,
                metadata={
                    "intent": rule.intent,
                    "matched_text": match.group(0),
                    "match_span": [match.start(), match.end()],
                },
            )

        informational_match = (rule, match)

    if informational_match is not None:
        rule, match = informational_match
        LOGGER.info("Allowing borderline injection phrasing: %s", rule.intent)
        return GuardrailDecision(
            passed=True,
            guardrail="input.injection_regex",
            severity="info",
            reason=rule.reason,
            matched_pattern=rule.pattern.pattern,
            metadata={
                "intent": rule.intent,
                "matched_text": match.group(0),
                "match_span": [match.start(), match.end()],
            },
        )

    return GuardrailDecision(
        passed=True,
        guardrail="input.injection_regex",
        severity="info",
        reason="no known prompt-injection patterns detected",
        matched_pattern=None,
        metadata={},
    )


def check_length(text: str, max_chars: int = 4000) -> GuardrailDecision:
    """Warn on oversized input and block extremely large input."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    char_count = len(text)
    block_threshold = max_chars * 2

    if char_count >= block_threshold:
        LOGGER.warning(
            "Blocking oversized input: %d chars exceeds hard limit %d.",
            char_count,
            block_threshold,
        )
        return GuardrailDecision(
            passed=False,
            guardrail="input.length",
            severity="block",
            reason=f"input length {char_count} exceeds hard limit {block_threshold}",
            matched_pattern=None,
            metadata={
                "char_count": char_count,
                "max_chars": max_chars,
                "block_threshold": block_threshold,
            },
        )

    if char_count > max_chars:
        LOGGER.warning(
            "Warning on oversized input: %d chars exceeds soft limit %d.",
            char_count,
            max_chars,
        )
        return GuardrailDecision(
            passed=False,
            guardrail="input.length",
            severity="warn",
            reason=f"input length {char_count} exceeds soft limit {max_chars}",
            matched_pattern=None,
            metadata={
                "char_count": char_count,
                "max_chars": max_chars,
                "block_threshold": block_threshold,
            },
        )

    return GuardrailDecision(
        passed=True,
        guardrail="input.length",
        severity="info",
        reason="input length is within limits",
        matched_pattern=None,
        metadata={
            "char_count": char_count,
            "max_chars": max_chars,
            "block_threshold": block_threshold,
        },
    )


def check_excessive_repetition(text: str) -> GuardrailDecision:
    """Detect obvious repetition that looks like DoS-style spam input."""

    tokens = [match.group(0).lower() for match in re.finditer(r"\b[\w']+\b", text)]
    if not tokens:
        return GuardrailDecision(
            passed=True,
            guardrail="input.excessive_repetition",
            severity="info",
            reason="no excessive repetition detected",
            matched_pattern=None,
            metadata={"repeat_count": 0},
        )

    current_token = tokens[0]
    current_run = 1
    longest_token = current_token
    longest_run = 1

    for token in tokens[1:]:
        if token == current_token:
            current_run += 1
        else:
            current_token = token
            current_run = 1

        if current_run > longest_run:
            longest_run = current_run
            longest_token = current_token

    if longest_run > 50:
        LOGGER.warning(
            "Warning on repeated input token %r occurring %d times consecutively.",
            longest_token,
            longest_run,
        )
        return GuardrailDecision(
            passed=False,
            guardrail="input.excessive_repetition",
            severity="warn",
            reason=f"input repeats the token {longest_token!r} {longest_run} times consecutively",
            matched_pattern=longest_token,
            metadata={"repeat_token": longest_token, "repeat_count": longest_run},
        )

    return GuardrailDecision(
        passed=True,
        guardrail="input.excessive_repetition",
        severity="info",
        reason="no excessive repetition detected",
        matched_pattern=None,
        metadata={"repeat_token": longest_token, "repeat_count": longest_run},
    )


def check_input(text: str) -> GuardrailReport:
    """Run the LC-31 input guardrails and return a composed report."""

    decisions = [
        check_injection_patterns(text),
        check_length(text),
        check_excessive_repetition(text),
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


__all__ = [
    "check_excessive_repetition",
    "check_injection_patterns",
    "check_input",
    "check_length",
]
