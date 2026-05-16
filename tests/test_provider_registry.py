from __future__ import annotations

import pytest

from src.llm.provider_registry import SUPPORTED_PROVIDERS, ProviderSpec, get_provider_spec


def test_supported_providers_cover_live_provider_matrix() -> None:
    assert SUPPORTED_PROVIDERS == {"anthropic", "openai", "gemini", "ollama", "mock"}


@pytest.mark.parametrize(
    ("provider_name", "model_env_var", "default_model", "cache_default", "mode", "required_env_vars"),
    [
        ("anthropic", "ANTHROPIC_MODEL", "claude-haiku-4-5", True, "hosted", ("ANTHROPIC_API_KEY",)),
        ("openai", "OPENAI_MODEL", "gpt-4o-mini", True, "hosted", ("OPENAI_API_KEY",)),
        ("gemini", "GEMINI_MODEL", "gemini-2.5-flash", True, "hosted", ("GOOGLE_API_KEY",)),
        ("ollama", "OLLAMA_MODEL", "llama3.1:8b", True, "local", ()),
        ("mock", "MOCK_MODEL", "mock-model-v1", False, "mock", ()),
    ],
)
def test_provider_specs_match_current_defaults(
    provider_name: str,
    model_env_var: str,
    default_model: str,
    cache_default: bool,
    mode: str,
    required_env_vars: tuple[str, ...],
) -> None:
    spec = get_provider_spec(provider_name)

    assert isinstance(spec, ProviderSpec)
    assert spec.name == provider_name
    assert spec.model_env_var == model_env_var
    assert spec.default_model == default_model
    assert spec.cache_default is cache_default
    assert spec.mode == mode
    assert spec.required_env_vars == required_env_vars


def test_get_provider_spec_normalizes_input() -> None:
    assert get_provider_spec("  MoCk ").name == "mock"


def test_get_provider_spec_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider_spec("banana")
