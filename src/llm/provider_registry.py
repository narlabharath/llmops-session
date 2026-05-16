"""Internal provider metadata for the llm module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model_env_var: str
    default_model: str
    cache_default: bool
    mode: Literal["hosted", "local", "mock"]
    required_env_vars: tuple[str, ...]


_PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        name="anthropic",
        model_env_var="ANTHROPIC_MODEL",
        default_model="claude-haiku-4-5",
        cache_default=True,
        mode="hosted",
        required_env_vars=("ANTHROPIC_API_KEY",),
    ),
    "openai": ProviderSpec(
        name="openai",
        model_env_var="OPENAI_MODEL",
        default_model="gpt-4o-mini",
        cache_default=True,
        mode="hosted",
        required_env_vars=("OPENAI_API_KEY",),
    ),
    "gemini": ProviderSpec(
        name="gemini",
        model_env_var="GEMINI_MODEL",
        default_model="gemini-2.5-flash",
        cache_default=True,
        mode="hosted",
        required_env_vars=("GOOGLE_API_KEY",),
    ),
    "ollama": ProviderSpec(
        name="ollama",
        model_env_var="OLLAMA_MODEL",
        default_model="llama3.1:8b",
        cache_default=True,
        mode="local",
        required_env_vars=(),
    ),
    "mock": ProviderSpec(
        name="mock",
        model_env_var="MOCK_MODEL",
        default_model="mock-model-v1",
        cache_default=False,
        mode="mock",
        required_env_vars=(),
    ),
}

SUPPORTED_PROVIDERS = frozenset(_PROVIDER_SPECS)
MODEL_ENV_VARS = {name: spec.model_env_var for name, spec in _PROVIDER_SPECS.items()}
MODEL_DEFAULTS = {name: spec.default_model for name, spec in _PROVIDER_SPECS.items()}
CACHE_DEFAULTS = {name: spec.cache_default for name, spec in _PROVIDER_SPECS.items()}


def get_provider_spec(name: str) -> ProviderSpec:
    normalized_name = name.strip().lower()
    spec = _PROVIDER_SPECS.get(normalized_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider '{normalized_name}'. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    return spec
