"""Externalized prompt templates per D-019.

Loads .md prompt files from src/llm/prompts/ keyed by name + version.
Caller substitutes variables via str.format(**kwargs).

Naming: src/llm/prompts/<name>.<version>.md
Example: src/llm/prompts/classify_question.v1.md
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, version: str = "v1") -> str:
    """Return the raw prompt text for {name}.{version}.md."""

    path = _PROMPTS_DIR / f"{name}.{version}.md"
    if not path.exists():
        available = [p.name for p in _PROMPTS_DIR.glob("*.md")] if _PROMPTS_DIR.exists() else []
        raise FileNotFoundError(f"No prompt at {path}. Available: {available}")
    return path.read_text(encoding="utf-8")


def list_prompts() -> list[str]:
    """Return all available prompt files."""

    if not _PROMPTS_DIR.exists():
        return []
    return sorted(p.name for p in _PROMPTS_DIR.glob("*.md"))


__all__ = ["load_prompt", "list_prompts"]
