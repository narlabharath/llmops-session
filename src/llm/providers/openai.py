"""OpenAI provider implementation using the Chat Completions API."""

from __future__ import annotations

import importlib
import os
from typing import Any


class OpenAIProvider:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            openai = importlib.import_module("openai")
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                raise ImportError("openai package not installed; pip install openai") from exc
            raise
        self._client_factory = openai.OpenAI
        self._client: Any | None = None

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            request["messages"].insert(0, {"role": "system", "content": system})
        request.update(kwargs)

        client = self._client or self._client_factory()
        self._client = client
        response = client.chat.completions.create(**request)
        text = response.choices[0].message.content if response.choices else ""
        usage = getattr(response, "usage", None)
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()
        else:
            raw = {"repr": repr(response)}

        return {
            "text": text or "",
            "tokens_in": getattr(usage, "prompt_tokens", 0) or 0,
            "tokens_out": getattr(usage, "completion_tokens", 0) or 0,
            "raw": raw,
            "model": getattr(response, "model", self.model),
        }
