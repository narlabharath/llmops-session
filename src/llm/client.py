"""Pluggable LLM client with optional in-memory response caching."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        """Best-effort stdlib fallback when python-dotenv is unavailable."""

        candidates = [Path.cwd(), *Path.cwd().parents]
        for folder in candidates:
            env_path = folder / ".env"
            if not env_path.exists():
                continue
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
            return True
        return False

from .cache import LLMCache
from .provider_registry import (
    CACHE_DEFAULTS,
    MODEL_DEFAULTS,
    MODEL_ENV_VARS,
    SUPPORTED_PROVIDERS,
    get_provider_spec,
)

_SUPPORTED_PROVIDERS = SUPPORTED_PROVIDERS
_CACHE_DEFAULTS = CACHE_DEFAULTS
_MODEL_ENV_VARS = MODEL_ENV_VARS
_MODEL_DEFAULTS = MODEL_DEFAULTS


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_estimate_usd: float
    cache_status: Literal["hit", "miss", "bypass"]
    raw: dict[str, Any] | None = None


class LLMClient:
    """Pluggable LLM client.

    Selects a provider implementation based on the LLM_PROVIDER env var
    (or the `provider=` constructor arg if explicitly passed).

    Caching is default-on for hosted providers and default-off for mock.
    Pass `cache=False` to disable, `cache=True` to force on,
    or `cache=<LLMCache instance>` to inject a specific cache.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        cache: bool | LLMCache | None = None,
        prompt_version: str = "v1",
    ) -> None:
        if not os.getenv("LLM_DISABLE_DOTENV"):
            load_dotenv()

        provider_name = provider or os.getenv("LLM_PROVIDER", "anthropic")
        provider_spec = get_provider_spec(provider_name)

        self._provider_name = provider_spec.name
        self._model = model or os.getenv(
            provider_spec.model_env_var,
            provider_spec.default_model,
        )
        self._prompt_version = prompt_version
        self._cache_default_enabled = provider_spec.cache_default
        self._cache_enabled = self._cache_default_enabled
        self._cache: LLMCache | None = None

        if isinstance(cache, LLMCache):
            self._cache = cache
            self._cache_enabled = True
        elif cache is True:
            self._cache = LLMCache()
            self._cache_enabled = True
        elif cache is False:
            self._cache_enabled = False
        elif self._cache_default_enabled:
            self._cache = LLMCache()

        self._provider = self._build_provider(self._provider_name, self._model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        cache: bool | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        prompt_version = str(kwargs.pop("prompt_version", self._prompt_version))
        cache_enabled = self._cache_enabled if cache is None else cache
        if not cache_enabled:
            if self._cache is not None:
                self._cache.record_bypass()
            return self._call_provider(prompt, system, "bypass", **kwargs)

        if self._cache is None:
            self._cache = LLMCache()

        key = self._cache.make_key(
            prompt=prompt,
            system=system,
            model=self._model,
            prompt_version=prompt_version,
            kwargs=kwargs,
        )

        lookup_started = time.perf_counter()
        cached = self._cache.get(key)
        lookup_latency_ms = (time.perf_counter() - lookup_started) * 1000
        if cached is not None:
            return replace(cached, cache_status="hit", latency_ms=lookup_latency_ms)

        result = self._call_provider(prompt, system, "miss", **kwargs)
        self._cache.set(key, result)
        return result

    def _build_provider(self, provider: str, model: str) -> Any:
        if provider == "anthropic":
            from .providers.anthropic import AnthropicProvider

            return AnthropicProvider(model=model)
        if provider == "openai":
            from .providers.openai import OpenAIProvider

            return OpenAIProvider(model=model)
        if provider == "gemini":
            from .providers.gemini import GeminiProvider

            return GeminiProvider(model=model)
        if provider == "ollama":
            from .providers.ollama import OllamaProvider

            return OllamaProvider(model=model)
        from .providers.mock import MockProvider

        return MockProvider(model=model)

    def _call_provider(
        self,
        prompt: str,
        system: str | None,
        cache_status: Literal["miss", "bypass"],
        **kwargs: Any,
    ) -> CompletionResult:
        started = time.perf_counter()
        payload = self._provider.complete(prompt=prompt, system=system, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        model_name = str(payload.get("model") or self._model)
        tokens_in = int(payload.get("tokens_in", 0) or 0)
        tokens_out = int(payload.get("tokens_out", 0) or 0)
        cost_estimate = self._estimate_cost(model_name, tokens_in, tokens_out)
        return CompletionResult(
            text=str(payload.get("text", "")),
            model=model_name,
            provider=self._provider_name,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_estimate_usd=cost_estimate,
            cache_status=cache_status,
            raw=payload.get("raw"),
        )

    def _estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        if self._provider_name == "anthropic" and "haiku-4-5" in model.lower():
            return (tokens_in / 1_000_000) * 1.0 + (tokens_out / 1_000_000) * 5.0

        # TODO: real pricing for non-Haiku Anthropic models and other providers.
        return 0.0
