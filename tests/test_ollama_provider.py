from __future__ import annotations

import json

import pytest

from src.llm.providers import ollama as ollama_provider_module
from src.llm.providers.ollama import OllamaProvider


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def patch_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object] | None = None,
    *,
    side_effect: Exception | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    fake_payload = payload or {
        "response": "stubbed response",
        "prompt_eval_count": 9,
        "eval_count": 5,
        "model": "llama3.1:8b",
    }

    def fake_urlopen(http_request: object, timeout: int = 30) -> FakeHTTPResponse:
        captured["url"] = http_request.full_url
        captured["method"] = http_request.get_method()
        captured["headers"] = {key.lower(): value for key, value in http_request.header_items()}
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        captured["timeout"] = timeout
        if side_effect is not None:
            raise side_effect
        return FakeHTTPResponse(fake_payload)

    monkeypatch.setattr(ollama_provider_module.request, "urlopen", fake_urlopen)
    return captured


def test_ollama_provider_constructor_uses_explicit_overrides() -> None:
    provider = OllamaProvider(model="llama3:test", base_url="http://localhost:11434/v1/")

    assert provider.model == "llama3:test"
    assert provider.base_url == "http://localhost:11434"


def test_ollama_provider_constructor_uses_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama-env")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/api")

    provider = OllamaProvider()

    assert provider.model == "llama-env"
    assert provider.base_url == "http://localhost:11434"


def test_ollama_provider_constructor_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    provider = OllamaProvider()

    assert provider.model == "llama3.1:8b"
    assert provider.base_url == "http://localhost:11434"


def test_ollama_provider_complete_returns_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = patch_urlopen(
        monkeypatch,
        payload={
            "response": "hello from ollama",
            "prompt_eval_count": 12,
            "eval_count": 6,
            "model": "llama3:test",
        },
    )
    provider = OllamaProvider(model="llama3:test", base_url="http://localhost:11434/v1")

    result = provider.complete(
        "Hello",
        system="Be brief",
        max_tokens=77,
        temperature=0.2,
        format="json",
        options={"seed": 7},
    )

    assert result == {
        "text": "hello from ollama",
        "tokens_in": 12,
        "tokens_out": 6,
        "raw": {
            "response": "hello from ollama",
            "prompt_eval_count": 12,
            "eval_count": 6,
            "model": "llama3:test",
        },
        "model": "llama3:test",
    }
    assert captured == {
        "url": "http://localhost:11434/api/generate",
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": {
            "model": "llama3:test",
            "prompt": "Hello",
            "stream": False,
            "options": {
                "num_predict": 77,
                "temperature": 0.2,
                "seed": 7,
            },
            "system": "Be brief",
            "format": "json",
        },
        "timeout": 30,
    }


def test_ollama_provider_raises_clear_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_urlopen(
        monkeypatch,
        side_effect=ollama_provider_module.error.URLError("connection refused"),
    )
    provider = OllamaProvider(base_url="http://localhost:11434")

    with pytest.raises(
        ConnectionError,
        match=r"Ollama server not reachable at http://localhost:11434",
    ):
        provider.complete("Hello")
