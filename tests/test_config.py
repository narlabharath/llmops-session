"""Verifies .env.example declares every env var that LLMClient + providers read."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.llm.provider_registry import MODEL_ENV_VARS, SUPPORTED_PROVIDERS, get_provider_spec

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


@pytest.fixture(scope="module")
def env_example_text() -> str:
    return ENV_EXAMPLE.read_text(encoding="utf-8")


def test_env_example_lists_llm_provider(env_example_text: str) -> None:
    assert "LLM_PROVIDER" in env_example_text


def test_env_example_lists_every_provider_model_var(env_example_text: str) -> None:
    for env_var in MODEL_ENV_VARS.values():
        assert env_var in env_example_text


def test_env_example_lists_required_provider_env_vars(env_example_text: str) -> None:
    required_env_vars = {
        env_var
        for provider in SUPPORTED_PROVIDERS
        for env_var in get_provider_spec(provider).required_env_vars
    }
    for env_var in sorted(required_env_vars):
        assert env_var in env_example_text


def test_env_example_lists_ollama_base_url(env_example_text: str) -> None:
    assert "OLLAMA_BASE_URL=http://localhost:11434" in env_example_text


def test_env_example_supported_providers_match_client(env_example_text: str) -> None:
    match = re.search(r"^# One of:\s*(.+)$", env_example_text, flags=re.MULTILINE)
    assert match is not None
    declared = {item.strip() for item in match.group(1).split("|")}
    assert declared == SUPPORTED_PROVIDERS
