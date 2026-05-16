"""Anthropic provider implementation using the Messages API."""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic


class AnthropicProvider:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self._client = Anthropic()

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            request["system"] = system
        request.update(kwargs)

        response = self._client.messages.create(**request)
        text = response.content[0].text if response.content else ""
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()
        else:
            raw = {"repr": repr(response)}

        return {
            "text": text,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
            "raw": raw,
            "model": getattr(response, "model", self.model),
        }
