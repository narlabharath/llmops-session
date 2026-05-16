from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.providers import openai as openai_provider_module
from src.llm.providers.openai import OpenAIProvider


class FakeResponse:
    def __init__(self, text: str = "stubbed response") -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        self.model = "gpt-test"

    def model_dump(self) -> dict[str, str]:
        return {"id": "resp_123"}


class FakeOpenAIClient:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse()
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


def patch_openai_sdk(
    monkeypatch: pytest.MonkeyPatch,
    client: FakeOpenAIClient | None = None,
) -> FakeOpenAIClient:
    fake_client = client or FakeOpenAIClient()
    original_import_module = openai_provider_module.importlib.import_module
    fake_module = SimpleNamespace(OpenAI=lambda: fake_client)

    def fake_import_module(name: str) -> object:
        if name == "openai":
            return fake_module
        return original_import_module(name)

    monkeypatch.setattr(openai_provider_module.importlib, "import_module", fake_import_module)
    return fake_client


def test_openai_provider_constructor_uses_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_openai_sdk(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-env")

    provider = OpenAIProvider(model="gpt-explicit")

    assert provider.model == "gpt-explicit"


def test_openai_provider_constructor_uses_openai_model_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_openai_sdk(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-env")

    provider = OpenAIProvider()

    assert provider.model == "gpt-env"


def test_openai_provider_constructor_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_openai_sdk(monkeypatch)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = OpenAIProvider()

    assert provider.model == "gpt-4o-mini"


def test_openai_provider_raises_clear_import_error_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = openai_provider_module.importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'", name="openai")
        return original_import_module(name)

    monkeypatch.setattr(openai_provider_module.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="openai package not installed; pip install openai"):
        OpenAIProvider()


def test_openai_provider_complete_returns_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = patch_openai_sdk(monkeypatch, client=FakeOpenAIClient(FakeResponse("hello from openai")))
    provider = OpenAIProvider(model="gpt-explicit")

    result = provider.complete("Hello", system="Be brief", top_p=0.5)

    assert result == {
        "text": "hello from openai",
        "tokens_in": 11,
        "tokens_out": 7,
        "raw": {"id": "resp_123"},
        "model": "gpt-test",
    }
    assert fake_client.calls == [
        {
            "model": "gpt-explicit",
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.5,
        }
    ]
