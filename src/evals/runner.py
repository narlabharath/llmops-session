"""Golden dataset loader and evaluation runner helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from src.llm import LLMClient
from src.rag import get_embeddings, retrieve
from src.rag.types import RetrievedDocument
from src.workflow import run_workflow

from .judge import judge_groundedness
from .scoring import score_behavior, score_similarity
from .types import GoldenRow, RowResult

LOGGER = logging.getLogger(__name__)

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


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes"}


def _normalize_optional_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def _normalize_required_text(value: object, column_name: str) -> str:
    text = _normalize_optional_text(value)
    if text is None:
        raise ValueError(f"golden CSV row is missing a value for required column '{column_name}'")

    return text


def load_golden_rows(csv_path: Path) -> list[GoldenRow]:
    """Load golden query rows from CSV into typed dataclasses."""

    dataframe = pd.read_csv(csv_path)
    missing_columns = [column for column in EXPECTED_GOLDEN_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        LOGGER.warning("golden CSV %s missing expected columns: %s", csv_path, missing_display)
        raise ValueError(f"golden CSV missing expected columns: {missing_display}")

    rows = [
        GoldenRow(
            id=_normalize_required_text(record["id"], "id"),
            query=_normalize_required_text(record["query"], "query"),
            category=_normalize_required_text(record["category"], "category"),
            expected_behavior=_normalize_required_text(record["expected_behavior"], "expected_behavior"),
            reference_answer=_normalize_optional_text(record["reference_answer"]),
            required_context=_normalize_optional_text(record["required_context"]),
            risk_level=_normalize_required_text(record["risk_level"], "risk_level"),
            should_retrieve=_parse_bool(record["should_retrieve"]),
            should_refuse=_parse_bool(record["should_refuse"]),
            should_escalate=_parse_bool(record["should_escalate"]),
            notes=_normalize_optional_text(record["notes"]),
        )
        for record in dataframe.to_dict(orient="records")
    ]
    rows.sort(key=lambda row: row.id)
    return rows


def run_evals(
    golden_rows: list[GoldenRow],
    persist_dir: Path,
    workflow_llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    k: int = 5,
    escalation_threshold: float = 0.3,
) -> list[RowResult]:
    """Run the workflow and eval scorers for each golden row."""

    resolved_workflow_llm, resolved_judge_llm = _resolve_llm_clients(
        workflow_llm=workflow_llm,
        judge_llm=judge_llm,
    )
    resolved_persist_dir = Path(persist_dir)
    embeddings = None
    row_results: list[RowResult] = []

    for row in golden_rows:
        workflow_result = run_workflow(
            row.query,
            resolved_persist_dir,
            llm=resolved_workflow_llm,
            k=k,
            escalation_threshold=escalation_threshold,
        )
        behavior_match, failures = score_behavior(row, workflow_result.outcome)

        groundedness = None
        if workflow_result.outcome == "answered":
            retrieved_context = _reconstruct_retrieved_context(
                question=row.query,
                persist_dir=resolved_persist_dir,
                k=k,
                retrieved_doc_ids=workflow_result.retrieved_doc_ids,
            )
            groundedness = judge_groundedness(
                row.query,
                workflow_result.text,
                retrieved_context,
                judge_llm=resolved_judge_llm,
            )
            if not groundedness.grounded:
                failures = [*failures, f"answer not grounded: {groundedness.reason}"]

        similarity_score = None
        if row.reference_answer:
            if embeddings is None:
                embeddings = get_embeddings()
            similarity_score = score_similarity(
                workflow_result.text,
                row.reference_answer,
                embeddings,
            )

        latency_ms = _sum_trace_metric(workflow_result.trace, "latency_ms")
        cost_estimate_usd = _sum_trace_metric(workflow_result.trace, "cost_estimate_usd")
        passed = behavior_match and (groundedness is None or groundedness.grounded)
        row_result = RowResult(
            row=row,
            workflow_outcome=workflow_result.outcome,
            workflow_text=workflow_result.text,
            behavior_match=behavior_match,
            groundedness=groundedness,
            similarity_score=similarity_score,
            passed=passed,
            failures=failures,
            latency_ms=latency_ms,
            cost_estimate_usd=cost_estimate_usd,
        )
        row_results.append(row_result)

        LOGGER.info(
            "row %s: outcome=%s passed=%s",
            row.id,
            workflow_result.outcome,
            passed,
        )
        if not passed:
            LOGGER.warning("row %s failed checks: %s", row.id, failures)

    return row_results


def _resolve_llm_clients(
    workflow_llm: LLMClient | None,
    judge_llm: LLMClient | None,
) -> tuple[LLMClient, LLMClient]:
    if workflow_llm is None and judge_llm is None:
        shared_llm = LLMClient()
        return shared_llm, shared_llm
    if workflow_llm is None:
        assert judge_llm is not None
        return judge_llm, judge_llm
    if judge_llm is None:
        return workflow_llm, workflow_llm
    return workflow_llm, judge_llm


def _reconstruct_retrieved_context(
    question: str,
    persist_dir: Path,
    k: int,
    retrieved_doc_ids: list[str],
) -> str:
    retrieved_docs = retrieve(
        persist_dir=Path(persist_dir),
        query=question,
        k=max(k, len(retrieved_doc_ids)),
    )
    ordered_docs = _order_retrieved_docs(retrieved_docs, retrieved_doc_ids)
    return _format_retrieved_context(ordered_docs)


def _order_retrieved_docs(
    retrieved_docs: list[RetrievedDocument],
    retrieved_doc_ids: list[str],
) -> list[RetrievedDocument]:
    if not retrieved_doc_ids:
        return retrieved_docs

    docs_by_id = {_get_retrieved_doc_id(document): document for document in retrieved_docs}
    ordered_docs: list[RetrievedDocument] = []
    seen_ids: set[str] = set()

    for doc_id in retrieved_doc_ids:
        document = docs_by_id.get(doc_id)
        if document is None:
            continue
        ordered_docs.append(document)
        seen_ids.add(doc_id)

    ordered_docs.extend(
        document
        for document in retrieved_docs
        if _get_retrieved_doc_id(document) not in seen_ids
    )
    return ordered_docs


def _format_retrieved_context(retrieved_docs: list[RetrievedDocument]) -> str:
    if not retrieved_docs:
        return "[source: none, priority: 99, score: 0.000, rank: 0]\n<no retrieved documents>"

    rendered_documents: list[str] = []
    for document in retrieved_docs:
        rendered_documents.append(
            (
                f"[source: {_get_retrieved_doc_id(document)}, "
                f"priority: {document.chunk.metadata.source_priority}, "
                f"score: {document.score:.3f}, rank: {document.rank}]\n"
                f"{document.chunk.text}"
            )
        )
    return "\n\n".join(rendered_documents)


def _get_retrieved_doc_id(document: RetrievedDocument) -> str:
    return (
        document.chunk.metadata.document_id
        or document.chunk.source_path
        or f"rank-{document.rank}"
    )


def _sum_trace_metric(trace: list[dict[str, object]], field: str) -> float:
    total = 0.0
    for entry in trace:
        value = entry.get(field)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


__all__ = ["EXPECTED_GOLDEN_COLUMNS", "load_golden_rows", "run_evals"]
