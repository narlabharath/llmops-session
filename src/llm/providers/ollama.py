"""Ollama provider implementation using the local generate API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    changed = True
    while changed:
        changed = False
        for suffix in ("/api/generate", "/api", "/v1"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].rstrip("/")
                changed = True
                break
    return normalized or "http://localhost:11434"


class OllamaProvider:
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        configured_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = _normalize_base_url(configured_base_url)
        self._generate_url = f"{self.base_url}/api/generate"

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        extra_options = kwargs.pop("options", None)
        if isinstance(extra_options, dict):
            options.update(extra_options)

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system is not None:
            payload["system"] = system
        payload.update(kwargs)
        payload["stream"] = False

        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self._generate_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ConnectionError(
                f"Ollama server not reachable at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        return {
            "text": str(raw.get("response", "") or ""),
            "tokens_in": int(raw.get("prompt_eval_count", 0) or 0),
            "tokens_out": int(raw.get("eval_count", 0) or 0),
            "raw": raw,
            "model": str(raw.get("model") or self.model),
        }
