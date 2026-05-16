from __future__ import annotations

from pathlib import Path

import pytest

import src.llm.prompts as prompt_module
from src.llm import list_prompts, load_prompt


def test_load_prompt_missing_lists_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "existing.v1.md").write_text("Prompt body", encoding="utf-8")
    monkeypatch.setattr(prompt_module, "_PROMPTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt("nonexistent")

    message = str(excinfo.value)
    assert "nonexistent.v1.md" in message
    assert "existing.v1.md" in message


def test_load_prompt_round_trips_prompt_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = "Classify this question."
    (tmp_path / "classify_question.v1.md").write_text(expected, encoding="utf-8")
    monkeypatch.setattr(prompt_module, "_PROMPTS_DIR", tmp_path)

    assert load_prompt("classify_question") == expected


def test_list_prompts_returns_sorted_prompt_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "ground_answer.v1.md").write_text("Answer prompt", encoding="utf-8")
    (tmp_path / "classify_question.v1.md").write_text("Classify prompt", encoding="utf-8")
    monkeypatch.setattr(prompt_module, "_PROMPTS_DIR", tmp_path)

    assert list_prompts() == ["classify_question.v1.md", "ground_answer.v1.md"]
