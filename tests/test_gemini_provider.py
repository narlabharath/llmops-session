from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.providers import gemini as gemini_provider_module
from src.llm.providers.gemini import GeminiProvider


class FakeResponse:
    def __init__(self, text: str = "stubbed response") -> None:
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=13,
            candidates_token_count=5,
            total_token_count=18,
        )
        self.model_version = "gemini-test"

    def to_dict(self) -> dict[str, str]:
        return {"id": "gemini_123"}


class FakeGenerativeModel:
    def __init__(
        self,
        calls: list[dict[str, object]],
        response: FakeResponse,
        *,
        model_name: str,
        system_instruction: str | None = None,
    ) -> None:
        self._calls = calls
        self._response = response
        self._model_name = model_name
        self._system_instruction = system_instruction

    def generate_content(self, prompt: str, **kwargs: object) -> FakeResponse:
        self._calls.append(
            {
                "model_name": self._model_name,
                "system_instruction": self._system_instruction,
                "prompt": prompt,
                **kwargs,
            }
        )
        return self._response


class FakeGenAIModule:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse()
        self.configure_calls: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def configure(self, **kwargs: object) -> None:
        self.configure_calls.append(kwargs)

    def GenerativeModel(
        self,
        *,
        model_name: str,
        system_instruction: str | None = None,
    ) -> FakeGenerativeModel:
        return FakeGenerativeModel(
            self.calls,
            self.response,
            model_name=model_name,
            system_instruction=system_instruction,
        )


def patch_gemini_sdk(
    monkeypatch: pytest.MonkeyPatch,
    module: FakeGenAIModule | None = None,
) -> FakeGenAIModule:
    fake_module = module or FakeGenAIModule()
    original_import_module = gemini_provider_module.importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "google.generativeai":
            return fake_module
        return original_import_module(name)

    monkeypatch.setattr(gemini_provider_module.importlib, "import_module", fake_import_module)
    return fake_module


def test_gemini_provider_constructor_uses_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gemini_sdk(monkeypatch)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env")

    provider = GeminiProvider(model="gemini-explicit")

    assert provider.model == "gemini-explicit"


def test_gemini_provider_constructor_uses_gemini_model_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_gemini_sdk(monkeypatch)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env")

    provider = GeminiProvider()

    assert provider.model == "gemini-env"


def test_gemini_provider_constructor_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gemini_sdk(monkeypatch)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    provider = GeminiProvider()

    assert provider.model == "gemini-2.5-flash"


def test_gemini_provider_raises_clear_import_error_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = gemini_provider_module.importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "google.generativeai":
            raise ModuleNotFoundError(
                "No module named 'google.generativeai'",
                name="google.generativeai",
            )
        return original_import_module(name)

    monkeypatch.setattr(gemini_provider_module.importlib, "import_module", fake_import_module)

    with pytest.raises(
        ImportError,
        match="google-generativeai package not installed; pip install google-generativeai",
    ):
        GeminiProvider()


def test_gemini_provider_complete_returns_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = patch_gemini_sdk(
        monkeypatch,
        module=FakeGenAIModule(FakeResponse("hello from gemini")),
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    provider = GeminiProvider(model="gemini-explicit")

    result = provider.complete("Hello", system="Be brief", safety_settings={"mode": "safe"})

    assert result == {
        "text": "hello from gemini",
        "tokens_in": 13,
        "tokens_out": 5,
        "raw": {"id": "gemini_123"},
        "model": "gemini-test",
    }
    assert fake_module.configure_calls == [{"api_key": "test-key"}]
    assert fake_module.calls == [
        {
            "model_name": "gemini-explicit",
            "system_instruction": "Be brief",
            "prompt": "Hello",
            "generation_config": {
                "max_output_tokens": 1024,
                "temperature": 0.7,
            },
            "safety_settings": {"mode": "safe"},
        }
    ]
