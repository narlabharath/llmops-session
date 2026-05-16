"""One-shot environment health check for llmops-session.

Run from llmops-session/:
    python scripts/setup_check.py

Exits 0 on success, 1 on first failure. Designed for human eyes
and for CI gates (T-008 Gate A self-check uses this).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(dotenv_path: Path | str | None = None, override: bool = False) -> bool:
        path = Path(dotenv_path) if dotenv_path is not None else ROOT / ".env"
        if not path.exists():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if override or key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip()
        return True

try:
    from rich.console import Console
except ModuleNotFoundError:
    Console = None

from src.llm.client import CompletionResult, LLMClient
from src.llm.provider_registry import (
    MODEL_DEFAULTS,
    MODEL_ENV_VARS,
    SUPPORTED_PROVIDERS,
    get_provider_spec,
)

SAMPLE_DOCS = [
    "sample_program_policy.md",
    "sample_faq.md",
    "sample_schedule.md",
    "sample_assignment_guidelines.md",
    "sample_support_process.md",
]
REQUIRED_FRONTMATTER_KEYS = (
    "document_id",
    "title",
    "document_type",
    "version",
    "effective_date",
    "owner",
    "source_priority",
)
EXPECTED_GOLDEN_COLUMNS = [
    "id",
    "query",
    "category",
    "expected_behavior",
    "reference_answer",
    "required_context",
    "risk_level",
    "should_retrieve",
    "should_refuse",
    "should_escalate",
    "notes",
]
VALID_CATEGORIES = {
    "informational",
    "policy",
    "procedural",
    "schedule",
    "assignment",
    "ambiguous",
    "sensitive_private",
    "out_of_scope",
    "prompt_injection",
    "feedback_regression",
}
CRITICAL_CATEGORIES = {"prompt_injection", "sensitive_private", "out_of_scope", "ambiguous"}
RED_TEAM_REQUIRED_CATEGORIES = {
    "prompt_injection",
    "private_data_request",
    "policy_bypass",
    "harmful_or_abusive",
    "out_of_scope_professional_advice",
    "unauthorized_role_assumption",
    "confidential_system_information",
}
DOTENV_DISABLE_ENV_VAR = "SETUP_CHECK_DISABLE_DOTENV"


@dataclass(frozen=True)
class CheckResult:
    status: Literal["ok", "warn", "fail"]
    description: str
    note: str | None = None


class Reporter:
    def __init__(self) -> None:
        self._console = Console() if Console is not None else None

    def line(self, index: int, total: int, result: CheckResult) -> None:
        if self._console is None:
            symbols = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
            line = f"[{index}/{total}] {symbols[result.status]} {result.description}"
            if result.note:
                line += f" - {result.note}"
            print(line)
            return

        symbols = {"ok": "✓", "warn": "⚠", "fail": "✗"}
        colors = {"ok": "green", "warn": "yellow", "fail": "red"}
        line = f"[{index}/{total}] [{colors[result.status]}]{symbols[result.status]}[/] {result.description}"
        if result.note:
            line += f" — {result.note}"
        self._console.print(line)

    def summary(self, total: int, success: bool) -> None:
        message = (
            f"{total}/{total} OK - ready to run notebooks"
            if success
            else f"1/{total} checks failed - see above"
        )
        if self._console is None:
            print(message)
            return

        style = "green" if success else "red"
        self._console.print(f"[{style}]{message}[/]")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def non_empty(value: Any) -> bool:
    text = str(value).strip()
    return text not in {"", "nan", "None"}


@lru_cache(maxsize=1)
def load_sample_documents() -> list[tuple[Path, dict[str, Any], str]]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyyaml is required to parse sample documents") from exc

    documents: list[tuple[Path, dict[str, Any], str]] = []
    for filename in SAMPLE_DOCS:
        path = ROOT / "data" / filename
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError(f"{filename}: missing opening frontmatter delimiter")
        end_index = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
        if end_index is None:
            raise ValueError(f"{filename}: missing closing frontmatter delimiter")

        frontmatter_text = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1 :]).strip()
        if not body:
            raise ValueError(f"{filename}: empty markdown body")

        metadata = yaml.safe_load(frontmatter_text)
        if not isinstance(metadata, dict):
            raise ValueError(f"{filename}: frontmatter did not parse to a mapping")

        documents.append((path, metadata, body))

    return documents


@lru_cache(maxsize=1)
def load_golden_dataframe() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required to load golden_queries.csv") from exc

    return pd.read_csv(ROOT / "data" / "golden_queries.csv")


def get_selected_provider() -> str:
    return os.getenv("LLM_PROVIDER", "").strip().lower()


def should_load_dotenv() -> bool:
    # Allow tests and clean-room diagnostics to ignore repo-local secrets.
    return not truthy(os.getenv(DOTENV_DISABLE_ENV_VAR, ""))


def get_configured_ollama_base_url() -> str:
    value = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    return value or "http://localhost:11434"


def check_python_version() -> CheckResult:
    if sys.version_info >= (3, 10):
        return CheckResult("ok", f"Python version is {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return CheckResult(
        "fail",
        "Python version requirement",
        f"Python >= 3.10 required; found {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def check_env_file() -> CheckResult:
    env_path = ROOT / ".env"
    if env_path.exists():
        return CheckResult("ok", ".env file found")
    return CheckResult("warn", ".env file not found", "mock provider can still run without it")


def check_llm_provider() -> CheckResult:
    provider = get_selected_provider()
    if not provider:
        return CheckResult(
            "fail",
            "LLM provider selection",
            f"set LLM_PROVIDER to one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}",
        )
    if provider not in SUPPORTED_PROVIDERS:
        return CheckResult(
            "fail",
            "LLM provider selection",
            f"unsupported provider '{provider}'; expected one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}",
        )
    return CheckResult("ok", f"LLM provider is '{provider}'")


def check_provider_env_vars() -> CheckResult:
    provider = get_selected_provider()
    if provider not in SUPPORTED_PROVIDERS:
        return CheckResult(
            "fail",
            "Provider model configuration",
            f"set LLM_PROVIDER to one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}",
        )

    provider_spec = get_provider_spec(provider)
    model_var = provider_spec.model_env_var
    model_value = os.getenv(model_var) or provider_spec.default_model
    if not model_value:
        return CheckResult("fail", "Provider model configuration", f"set {model_var} in .env")

    missing_env_vars = [
        env_var for env_var in provider_spec.required_env_vars if not os.getenv(env_var, "").strip()
    ]
    if missing_env_vars:
        return CheckResult(
            "fail",
            "Provider-specific environment",
            (
                f"set {', '.join(missing_env_vars)} in .env or the current shell "
                f"for provider '{provider}'"
            ),
        )

    if provider_spec.mode == "mock":
        return CheckResult(
            "ok",
            f"Provider environment is ready for '{provider}'",
            "mock provider requires no API key",
        )
    if provider_spec.mode == "local":
        return CheckResult(
            "ok",
            f"Provider environment is ready for '{provider}'",
            f"Ollama will use {get_configured_ollama_base_url()}",
        )

    return CheckResult("ok", f"Provider environment is ready for '{provider}'")


def check_sample_docs_exist() -> CheckResult:
    missing = [name for name in SAMPLE_DOCS if not (ROOT / "data" / name).exists()]
    if missing:
        return CheckResult("fail", "Sample documents exist", f"missing: {', '.join(missing)}")
    return CheckResult("ok", "All 5 sample documents exist")


def check_sample_docs_parse() -> CheckResult:
    try:
        documents = load_sample_documents()
    except (RuntimeError, ValueError) as exc:
        return CheckResult("fail", "Sample documents parse", str(exc))

    for path, metadata, _body in documents:
        for key in REQUIRED_FRONTMATTER_KEYS:
            if key not in metadata:
                return CheckResult("fail", "Sample documents parse", f"{path.name}: missing frontmatter key '{key}'")
    return CheckResult("ok", "All 5 sample documents parse as frontmatter + body")


def check_source_priority_unique() -> CheckResult:
    try:
        documents = load_sample_documents()
    except (RuntimeError, ValueError) as exc:
        return CheckResult("fail", "source_priority uniqueness", str(exc))

    priorities: list[int] = []
    for path, metadata, _body in documents:
        try:
            priorities.append(int(metadata["source_priority"]))
        except (KeyError, TypeError, ValueError):
            return CheckResult("fail", "source_priority uniqueness", f"{path.name}: source_priority must be an integer")

    if set(priorities) != {1, 2, 3, 4, 5}:
        return CheckResult("fail", "source_priority uniqueness", f"expected priorities 1..5; found {priorities}")
    return CheckResult("ok", "source_priority values are unique integers 1..5")


def check_golden_csv_loads() -> CheckResult:
    try:
        dataframe = load_golden_dataframe()
    except (RuntimeError, Exception) as exc:
        return CheckResult("fail", "golden_queries.csv loads", str(exc))

    actual_columns = list(dataframe.columns)
    if actual_columns != EXPECTED_GOLDEN_COLUMNS:
        missing = [column for column in EXPECTED_GOLDEN_COLUMNS if column not in actual_columns]
        unexpected = [column for column in actual_columns if column not in EXPECTED_GOLDEN_COLUMNS]
        detail = f"missing={missing or 'none'}; unexpected={unexpected or 'none'}"
        return CheckResult("fail", "golden_queries.csv schema", detail)
    return CheckResult("ok", "golden_queries.csv loads with the expected 11-column schema")


def check_golden_csv_categories() -> CheckResult:
    try:
        dataframe = load_golden_dataframe()
    except (RuntimeError, Exception) as exc:
        return CheckResult("fail", "golden_queries.csv categories", str(exc))

    invalid_rows: list[str] = []
    for row in dataframe.itertuples(index=False):
        if row.category not in VALID_CATEGORIES:
            invalid_rows.append(f"{row.id}:{row.category}")
    if invalid_rows:
        return CheckResult("fail", "golden_queries.csv categories", f"invalid rows: {', '.join(invalid_rows)}")
    return CheckResult("ok", "golden_queries.csv categories are valid")


def check_golden_csv_boolean_consistency() -> CheckResult:
    try:
        dataframe = load_golden_dataframe()
    except (RuntimeError, Exception) as exc:
        return CheckResult("fail", "golden_queries.csv boolean consistency", str(exc))

    issues: list[str] = []
    for row in dataframe.itertuples(index=False):
        expected_behavior = str(row.expected_behavior).strip().lower()
        if truthy(row.should_refuse) and "refuse" not in expected_behavior:
            issues.append(f"{row.id}: should_refuse=true but expected_behavior='{row.expected_behavior}'")
        if truthy(row.should_retrieve) and not non_empty(row.required_context):
            issues.append(f"{row.id}: should_retrieve=true but required_context is empty")

    if issues:
        return CheckResult("fail", "golden_queries.csv boolean consistency", "; ".join(issues))
    return CheckResult("ok", "golden_queries.csv boolean consistency checks passed")


def check_red_team_yaml() -> CheckResult:
    path = ROOT / "security" / "red_team_cases.yaml"
    if not path.exists():
        return CheckResult("warn", "red-team YAML not yet authored", "main session task T-004")

    try:
        import yaml
    except ModuleNotFoundError as exc:
        return CheckResult("fail", "red-team YAML", f"pyyaml is required to parse {path.name}: {exc}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult("fail", "red-team YAML", f"failed to parse {path.name}: {exc}")

    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        return CheckResult("fail", "red-team YAML", "expected a top-level 'cases:' list")

    seen = {
        case.get("category")
        for case in cases
        if isinstance(case, dict) and case.get("category")
    }
    missing = sorted(RED_TEAM_REQUIRED_CATEGORIES - seen)
    if missing:
        return CheckResult("fail", "red-team YAML", f"missing categories: {', '.join(missing)}")
    return CheckResult("ok", "red-team YAML covers all required categories")


def check_llm_client_instantiates() -> CheckResult:
    try:
        LLMClient()
    except Exception as exc:
        return CheckResult("fail", "LLMClient instantiation", str(exc))
    return CheckResult("ok", "LLMClient instantiates for the configured provider")


def check_one_token_completion() -> CheckResult:
    try:
        result = LLMClient().complete("ping", max_tokens=5)
    except Exception as exc:
        return CheckResult("fail", "One-token completion", str(exc))

    if not isinstance(result, CompletionResult):
        return CheckResult("fail", "One-token completion", f"expected CompletionResult; got {type(result).__name__}")
    return CheckResult("ok", "One-token completion returned a CompletionResult")


def main() -> int:
    if should_load_dotenv():
        load_dotenv(dotenv_path=ROOT / ".env", override=False)
    reporter = Reporter()
    checks = [
        check_python_version,
        check_env_file,
        check_llm_provider,
        check_provider_env_vars,
        check_sample_docs_exist,
        check_sample_docs_parse,
        check_source_priority_unique,
        check_golden_csv_loads,
        check_golden_csv_categories,
        check_golden_csv_boolean_consistency,
        check_red_team_yaml,
        check_llm_client_instantiates,
        check_one_token_completion,
    ]

    total = len(checks)
    for index, check in enumerate(checks, start=1):
        result = check()
        reporter.line(index, total, result)
        if result.status == "fail":
            reporter.summary(total, success=False)
            return 1

    reporter.summary(total, success=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
