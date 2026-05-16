from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.rag import ingest, retrieve

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run end-to-end embedding tests.",
)
def test_ingest_and_retrieve_end_to_end(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"

    result = ingest(DATA_DIR, persist_dir)

    assert result.documents_loaded == 5
    assert result.chunks_created >= 5
    assert result.chunks_indexed == result.chunks_created
    assert result.vector_store_path == persist_dir

    late_submission_results = retrieve(persist_dir, "late submission policy", k=3)
    assert late_submission_results
    assert late_submission_results[0].chunk.metadata.document_id == "program_policy"

    final_exam_results = retrieve(persist_dir, "final exam schedule", k=3)
    assert final_exam_results
    assert final_exam_results[0].chunk.metadata.document_id in {"schedule", "assignment_guidelines"}
