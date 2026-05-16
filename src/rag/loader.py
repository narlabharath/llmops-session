"""Document loading utilities for the RAG module."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .types import DEFAULT_SOURCE_PRIORITY, DocumentMetadata, LoadedDocument

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(
    r"\A---\r?\n(?P<frontmatter>.*?)(?:\r?\n)---(?:\r?\n)?(?P<body>.*)\Z",
    re.DOTALL,
)


def load_document(path: Path) -> LoadedDocument:
    """Load one markdown file and parse its YAML frontmatter."""

    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    raw_text = "\n".join(document.page_content for document in documents)
    frontmatter_text, body = _split_frontmatter(raw_text, path)
    metadata = _parse_metadata(frontmatter_text, path)
    return LoadedDocument(path=path, metadata=metadata, content=body)


def load_corpus(directory: Path, pattern: str = "sample_*.md") -> list[LoadedDocument]:
    """Load all matching markdown files from a directory."""

    documents = [load_document(path) for path in Path(directory).glob(pattern) if path.is_file()]
    return sorted(documents, key=lambda document: (document.metadata.source_priority, document.path.name))


def _split_frontmatter(raw_text: str, path: Path) -> tuple[str | None, str]:
    if not raw_text.startswith("---"):
        logger.warning("Document %s has no YAML frontmatter; using default metadata.", path)
        return None, raw_text

    match = _FRONTMATTER_PATTERN.match(raw_text)
    if match is None:
        logger.warning(
            "Document %s starts with a frontmatter marker but has no closing delimiter; using default metadata.",
            path,
        )
        return None, raw_text

    body = match.group("body").lstrip("\r\n")
    return match.group("frontmatter"), body


def _parse_metadata(frontmatter_text: str | None, path: Path) -> DocumentMetadata:
    if frontmatter_text is None:
        return DocumentMetadata()

    try:
        parsed_frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Document %s has malformed YAML frontmatter; using default metadata. %s", path, exc)
        return DocumentMetadata()

    if not isinstance(parsed_frontmatter, Mapping):
        logger.warning("Document %s frontmatter is not a mapping; using default metadata.", path)
        return DocumentMetadata()

    return DocumentMetadata(
        document_id=_as_optional_string(parsed_frontmatter.get("document_id")),
        title=_as_optional_string(parsed_frontmatter.get("title")),
        document_type=_as_optional_string(parsed_frontmatter.get("document_type")),
        version=_as_optional_string(parsed_frontmatter.get("version")),
        effective_date=_as_optional_string(parsed_frontmatter.get("effective_date")),
        owner=_as_optional_string(parsed_frontmatter.get("owner")),
        source_priority=_parse_source_priority(parsed_frontmatter.get("source_priority"), path),
    )


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_source_priority(value: Any, path: Path) -> int:
    if value is None:
        return DEFAULT_SOURCE_PRIORITY
    if isinstance(value, bool):
        logger.warning(
            "Document %s has non-integer source_priority %r; using default priority %d.",
            path,
            value,
            DEFAULT_SOURCE_PRIORITY,
        )
        return DEFAULT_SOURCE_PRIORITY

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Document %s has non-integer source_priority %r; using default priority %d.",
            path,
            value,
            DEFAULT_SOURCE_PRIORITY,
        )
        return DEFAULT_SOURCE_PRIORITY


__all__ = ["load_document", "load_corpus"]
