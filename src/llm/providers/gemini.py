"""Gemini provider implementation using the google-generativeai SDK."""

from __future__ import annotations

import importlib
import os
from typing import Any

from ..provider_registry import get_provider_spec


class GeminiProvider:
    def __init__(self, model: str | None = None) -> None:
        spec = get_provider_spec("gemini")
        self.model = model or os.getenv(spec.model_env_var, spec.default_model)
        try:
            genai = importlib.import_module("google.generativeai")
        except ModuleNotFoundError as exc:
            if exc.name in {"google", "google.generativeai"}:
                raise ImportError(
                    "google-generativeai package not installed; pip install google-generativeai"
                ) from exc
            raise

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self._model_factory = genai.GenerativeModel

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        extra_generation_config = kwargs.pop("generation_config", None)
        if isinstance(extra_generation_config, dict):
            generation_config.update(extra_generation_config)

        model_kwargs: dict[str, Any] = {"model_name": self.model}
        if system is not None:
            model_kwargs["system_instruction"] = system

        response = self._model_factory(**model_kwargs).generate_content(
            prompt,
            generation_config=generation_config,
            **kwargs,
        )
        usage = getattr(response, "usage_metadata", None)
        tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
        tokens_out = int(
            getattr(usage, "candidates_token_count", 0) or max(total_tokens - tokens_in, 0)
        )
        if hasattr(response, "to_dict"):
            raw = response.to_dict()
        elif hasattr(response, "to_json_dict"):
            raw = response.to_json_dict()
        else:
            raw = {"repr": repr(response)}

        return {
            "text": getattr(response, "text", "") or "",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "raw": raw,
            "model": getattr(response, "model_version", self.model),
        }
