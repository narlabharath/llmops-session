"""Smoke and provider-matrix tests for setup_check.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.setup_check as setup_check
from src.llm.providers import ollama as ollama_provider_module

REPO = Path(__file__).resolve().parent.parent


def run_setup_check(**env_overrides: str | None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SETUP_CHECK_DISABLE_DOTENV": "1"}
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    return subprocess.run(
        [sys.executable, "scripts/setup_check.py"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_setup_check_passes_with_mock_provider() -> None:
    result = run_setup_check(LLM_PROVIDER="mock")

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "13/13 OK" in result.stdout


@pytest.mark.parametrize(
    ("provider", "missing_env_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
    ],
)
def test_setup_check_fails_early_when_hosted_provider_key_is_missing(
    provider: str,
    missing_env_var: str,
) -> None:
    result = run_setup_check(LLM_PROVIDER=provider, **{missing_env_var: None})

    assert result.returncode != 0
    assert "Provider-specific environment" in result.stdout
    assert missing_env_var in result.stdout


def test_check_provider_env_vars_allows_keyless_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    result = setup_check.check_provider_env_vars()

    assert result.status == "ok"
    assert result.note == "mock provider requires no API key"


def test_check_provider_env_vars_reports_ollama_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

    result = setup_check.check_provider_env_vars()

    assert result.status == "ok"
    assert result.note == "Ollama will use http://127.0.0.1:11434/v1"


def test_check_one_token_completion_surfaces_ollama_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

    def fake_urlopen(_http_request: object, timeout: int = 30) -> object:
        raise ollama_provider_module.error.URLError("connection refused")

    monkeypatch.setattr(ollama_provider_module.request, "urlopen", fake_urlopen)

    result = setup_check.check_one_token_completion()

    assert result.status == "fail"
    assert result.description == "One-token completion"
    assert result.note is not None
    assert "http://127.0.0.1:11434" in result.note
    assert "ollama serve" in result.note
