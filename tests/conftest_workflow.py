from __future__ import annotations

from dataclasses import dataclass, field

from src.llm import CompletionResult


@dataclass
class ScriptedLLMClient:
    """Test double that returns the first response whose trigger matches the prompt."""

    rules: list[tuple[str, str]]
    prompt_version: str = "v1"
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: object,
    ) -> CompletionResult:
        self.calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})

        for trigger, response in self.rules:
            if trigger in prompt or (system and trigger in system):
                return CompletionResult(
                    text=response,
                    model="scripted",
                    provider="scripted",
                    latency_ms=1.0,
                    tokens_in=0,
                    tokens_out=0,
                    cost_estimate_usd=0.0,
                    cache_status="bypass",
                    raw={"matched_trigger": trigger, "prompt_version": self.prompt_version},
                )

        raise AssertionError(
            "No scripted rule matched. "
            f"Prompt was: {prompt[:200]!r}; system was: {(system or '')[:200]!r}"
        )
