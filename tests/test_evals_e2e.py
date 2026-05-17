from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

import src.evals.runner as runner_module
from src.evals import build_report, dump_report, load_golden_rows, run_evals
from src.rag import ingest, retrieve
from src.workflow import run_workflow as real_run_workflow
from tests.conftest_workflow import ScriptedLLMClient

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "golden_queries.csv"
ESCALATED_ROW_ID = "GQ012"
UNGROUNDED_ROW_ID = "GQ003"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW_TESTS") != "1",
        reason="Set RUN_SLOW_TESTS=1 to run eval end-to-end tests.",
    ),
]


@pytest.fixture(scope="module")
def sample_corpus_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    persist_dir = tmp_path_factory.mktemp("evals-e2e") / "chroma_test"
    ingest(DATA_DIR, persist_dir)
    return persist_dir


def test_run_build_and_dump_report_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    sample_corpus_store: Path,
    tmp_path: Path,
) -> None:
    golden_rows = load_golden_rows(CSV_PATH)
    escalation_query = next(row.query for row in golden_rows if row.id == ESCALATED_ROW_ID)
    forced_escalation_threshold = _threshold_above_observed_max_score(
        sample_corpus_store,
        escalation_query,
    )

    def wrapped_run_workflow(
        question: str,
        persist_dir: Path,
        llm: object | None = None,
        k: int = 5,
        escalation_threshold: float = 0.3,
    ):
        resolved_threshold = (
            forced_escalation_threshold if question == escalation_query else escalation_threshold
        )
        return real_run_workflow(
            question=question,
            persist_dir=Path(persist_dir),
            llm=llm,
            k=k,
            escalation_threshold=resolved_threshold,
        )

    monkeypatch.setattr(runner_module, "run_workflow", wrapped_run_workflow)

    row_results = run_evals(
        golden_rows,
        sample_corpus_store,
        workflow_llm=ScriptedLLMClient(rules=_build_workflow_rules(golden_rows)),
        judge_llm=ScriptedLLMClient(rules=_build_judge_rules(golden_rows)),
    )
    report = build_report(row_results)
    csv_path, json_path = dump_report(report, tmp_path)

    assert csv_path.exists()
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(row_results) == 12
    assert len(payload["rows"]) == 12
    assert set(result.workflow_outcome for result in row_results) == {
        "answered",
        "refused",
        "escalated",
    }
    assert payload["failed_ids"] == [UNGROUNDED_ROW_ID]
    assert isinstance(payload["pass_rate"], float)
    assert payload["pass_rate"] == pytest.approx(11 / 12)
    assert payload["rows"][2]["passed"] is False
    assert payload["rows"][2]["groundedness"]["grounded"] is False


def _build_workflow_rules(golden_rows) -> list[tuple[str, str]]:
    answer_rules: list[tuple[str, str]] = []
    classification_rules: list[tuple[str, str]] = []

    for row in golden_rows:
        classification = _classification_for_row(row.id)
        classification_rules.append((f'Message: "{row.query}"', classification))

        if classification in {"policy_question", "schedule_question", "assignment_question"}:
            answer_rules.append(
                (
                    f"Participant question: {row.query}",
                    json.dumps(
                        {
                            "answer": _answer_for_row(row.id, row.reference_answer),
                            "grounded": row.id != UNGROUNDED_ROW_ID,
                            "source_section": row.required_context or "Sample source",
                            "confidence": 0.99,
                        }
                    ),
                )
            )

    return [*answer_rules, *classification_rules]


def _build_judge_rules(golden_rows) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for row in golden_rows:
        classification = _classification_for_row(row.id)
        if classification not in {"policy_question", "schedule_question", "assignment_question"}:
            continue

        rules.append(
            (
                f"Question: {row.query}",
                json.dumps(
                    {
                        "grounded": row.id != UNGROUNDED_ROW_ID,
                        "confidence": 0.98 if row.id != UNGROUNDED_ROW_ID else 0.12,
                        "reason": (
                            "The answer stays within the retrieved context."
                            if row.id != UNGROUNDED_ROW_ID
                            else "The answer adds an unsupported grace-period claim."
                        ),
                    }
                ),
            )
        )
    return rules


def _classification_for_row(row_id: str) -> str:
    return {
        "GQ001": "policy_question",
        "GQ002": "policy_question",
        "GQ003": "policy_question",
        "GQ004": "policy_question",
        "GQ005": "private_request",
        "GQ006": "injection_attempt",
        "GQ007": "schedule_question",
        "GQ008": "out_of_scope",
        "GQ009": "policy_question",
        "GQ010": "out_of_scope",
        "GQ011": "schedule_question",
        "GQ012": "assignment_question",
    }[row_id]


def _answer_for_row(row_id: str, reference_answer: str | None) -> str:
    if row_id == UNGROUNDED_ROW_ID:
        return "Yes, there is an automatic 3-day grace period on assignments."
    return reference_answer or f"Scripted answer for {row_id}"


def _threshold_above_observed_max_score(persist_dir: Path, question: str) -> float:
    retrieved_docs = retrieve(persist_dir, question, k=5)
    assert retrieved_docs

    max_score = max(document.score for document in retrieved_docs)
    return math.nextafter(max_score, math.inf)
