from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.rag import DocumentMetadata, load_corpus, load_document

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize(
    ("filename", "document_id", "source_priority"),
    [
        ("sample_program_policy.md", "program_policy", 1),
        ("sample_assignment_guidelines.md", "assignment_guidelines", 2),
        ("sample_support_process.md", "support_process", 3),
        ("sample_schedule.md", "schedule", 4),
        ("sample_faq.md", "faq", 5),
    ],
)
def test_load_document_parses_sample_frontmatter(
    filename: str,
    document_id: str,
    source_priority: int,
) -> None:
    document = load_document(DATA_DIR / filename)

    assert document.metadata.document_id == document_id
    assert document.metadata.source_priority == source_priority
    assert document.content.startswith("# Sample")
    assert not document.content.startswith("---")


def test_load_corpus_returns_documents_sorted_by_source_priority() -> None:
    documents = load_corpus(DATA_DIR)

    assert len(documents) == 5
    assert [document.metadata.document_id for document in documents] == [
        "program_policy",
        "assignment_guidelines",
        "support_process",
        "schedule",
        "faq",
    ]
    assert [document.metadata.source_priority for document in documents] == [1, 2, 3, 4, 5]


def test_load_document_without_frontmatter_uses_default_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "plain.md"
    content = "# Plain Document\n\nBody text without frontmatter."
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.rag.loader"):
        document = load_document(path)

    assert document.metadata == DocumentMetadata()
    assert document.content == content
    assert "has no YAML frontmatter" in caplog.text


def test_load_document_with_malformed_yaml_uses_default_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "malformed.md"
    path.write_text(
        "---\n"
        "document_id: broken\n"
        "source_priority: [missing bracket\n"
        "---\n\n"
        "# Broken Document\n\n"
        "Body text.\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="src.rag.loader"):
        document = load_document(path)

    assert document.metadata == DocumentMetadata()
    assert document.content.startswith("# Broken Document")
    assert "malformed YAML frontmatter" in caplog.text


def test_load_document_with_invalid_source_priority_uses_sentinel(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "invalid-priority.md"
    path.write_text(
        "---\n"
        "document_id: custom_doc\n"
        "title: Custom Doc\n"
        "source_priority: high\n"
        "---\n\n"
        "# Custom Doc\n\n"
        "Body text.\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="src.rag.loader"):
        document = load_document(path)

    assert document.metadata.document_id == "custom_doc"
    assert document.metadata.title == "Custom Doc"
    assert document.metadata.source_priority == DocumentMetadata().source_priority
    assert document.content.startswith("# Custom Doc")
    assert "non-integer source_priority" in caplog.text
