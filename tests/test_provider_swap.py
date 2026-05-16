from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Callable

import pytest

import src.llm.client as client_module
from src.llm import CompletionResult, LLMClient
from src.llm.providers import anthropic as anthropic_provider_module
from src.llm.providers import gemini as gemini_provider_module
from src.llm.providers import ollama as ollama_provider_module
from src.llm.providers import openai as openai_provider_module


@pytest.fixture(autouse=True)
def disable_dotenv_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "load_dotenv", lambda: False)


class FakeAnthropicResponse:
    def __init__(self, text: str = "hello from anthropic") -> None:
        self.content = [SimpleNamespace(text=text)]
        self.usage = SimpleNamespace(input_tokens=8, output_tokens=3)
        self.model = "claude-test"

    def model_dump(self) -> dict[str, str]:
        return {"id": "anthropic_123"}


class FakeAnthropicClient:
    def __init__(self, response: FakeAnthropicResponse | None = None) -> None:
        self.response = response or FakeAnthropicResponse()
        self.calls: list[dict[str, object]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: object) -> FakeAnthropicResponse:
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIResponse:
    def __init__(self, text: str = "hello from openai") -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        self.model = "gpt-test"

    def model_dump(self) -> dict[str, str]:
        return {"id": "openai_123"}


class FakeOpenAIClient:
    def __init__(self, response: FakeOpenAIResponse | None = None) -> None:
        self.response = response or FakeOpenAIResponse()
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        return self.response


class FakeGeminiResponse:
    def __init__(self, text: str = "hello from gemini") -> None:
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
        response: FakeGeminiResponse,
        *,
        model_name: str,
        system_instruction: str | None = None,
    ) -> None:
        self._calls = calls
        self._response = response
        self._model_name = model_name
        self._system_instruction = system_instruction

    def generate_content(self, prompt: str, **kwargs: object) -> FakeGeminiResponse:
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
    def __init__(self, response: FakeGeminiResponse | None = None) -> None:
        self.response = response or FakeGeminiResponse()
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


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def patch_anthropic_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeAnthropicClient()
    monkeypatch.setattr(anthropic_provider_module, "Anthropic", lambda: fake_client)


def patch_openai_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeOpenAIClient()
    original_import_module = openai_provider_module.importlib.import_module
    fake_module = SimpleNamespace(OpenAI=lambda: fake_client)

    def fake_import_module(name: str) -> object:
        if name == "openai":
            return fake_module
        return original_import_module(name)

    monkeypatch.setattr(openai_provider_module.importlib, "import_module", fake_import_module)


def patch_gemini_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = FakeGenAIModule()
    original_import_module = gemini_provider_module.importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "google.generativeai":
            return fake_module
        return original_import_module(name)

    monkeypatch.setattr(gemini_provider_module.importlib, "import_module", fake_import_module)


def patch_ollama_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "response": "hello from ollama",
        "prompt_eval_count": 9,
        "eval_count": 5,
        "model": "llama-test",
    }

    def fake_urlopen(http_request: object, timeout: int = 30) -> FakeHTTPResponse:
        del http_request, timeout
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(ollama_provider_module.request, "urlopen", fake_urlopen)


def assert_completion_shape(result: CompletionResult, provider: str) -> None:
    assert isinstance(result, CompletionResult)
    assert result.provider == provider
    assert result.cache_status == "bypass"
    assert isinstance(result.text, str)
    assert result.text
    assert isinstance(result.model, str)
    assert result.model
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    assert isinstance(result.tokens_in, int)
    assert result.tokens_in >= 0
    assert isinstance(result.tokens_out, int)
    assert result.tokens_out >= 0
    assert isinstance(result.cost_estimate_usd, float)
    assert isinstance(result.raw, dict)


@pytest.mark.parametrize(
    ("provider", "patch_backend", "expected_text", "expected_model"),
    [
        ("anthropic", patch_anthropic_backend, "hello from anthropic", "claude-test"),
        ("openai", patch_openai_backend, "hello from openai", "gpt-test"),
        ("gemini", patch_gemini_backend, "hello from gemini", "gemini-test"),
        ("ollama", patch_ollama_backend, "hello from ollama", "llama-test"),
        ("mock", None, "[mock:", "mock-model-v1"),
    ],
)
def test_provider_swap_returns_consistent_completion_results(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    patch_backend: Callable[[pytest.MonkeyPatch], None] | None,
    expected_text: str,
    expected_model: str,
) -> None:
    if patch_backend is not None:
        patch_backend(monkeypatch)

    result = LLMClient(provider=provider).complete("hello", cache=False)

    assert_completion_shape(result, provider)
    assert expected_text in result.text
    assert result.model == expected_model


def test_llm_client_reads_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    result = LLMClient().complete("hello", cache=False)

    assert_completion_shape(result, "mock")
    assert result.model == "mock-model-v1"


def test_llm_client_defaults_to_anthropic_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    patch_anthropic_backend(monkeypatch)

    client = LLMClient()
    result = client.complete("hello", cache=False)

    assert client.provider_name == "anthropic"
    assert_completion_shape(result, "anthropic")
    assert result.model == "claude-test"


def test_llm_client_rejects_invalid_provider_name() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="banana")
