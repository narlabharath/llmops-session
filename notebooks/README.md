# Notebooks

## 01_core_llm_workflow.ipynb

Demonstrates the core LLM workflow against a real Anthropic backend.

NB 03 (when authored) will consume `src.workflow.run_workflow` to demonstrate the classify->route->answer flow.

### Run

From inside `llmops-session/`:

```bash
pip install -r requirements.txt
# Set ANTHROPIC_API_KEY in .env (see .env.example)
jupyter nbconvert --execute --to notebook --inplace notebooks/01_core_llm_workflow.ipynb
```

Or open in Jupyter Lab and run cells top-to-bottom.

### Cells

- Cells 3–7 — provider abstraction
- Cells 8–10 — non-determinism + system messages
- Cells 11–13 — cache hit/miss/version-bump
- Cells 14–15 — clean failure surfacing
- Cells 16–22 — prompting (few-shot, JSON, grounding, prompt versioning), anchored to the program-assistant running use case
- Cells 23–25 — closing arc: all levers applied to one participant question, bridge to NB 02 (RAG)

### Cells that need an API key

Cells 4, 9, 10, 12, 13, 15, 17, 19, 21, 24 need a valid `ANTHROPIC_API_KEY`. Without one they print `[skipped — no key]` and the notebook continues.

---

## 02_rag_pipeline.ipynb

Demonstrates the RAG (Retrieval-Augmented Generation) pipeline end-to-end against the sample program corpus. Shows five retrieval failure modes and their mitigations: hallucination without grounding, wrong-doc-on-top, missing context, conflicting sources, and stale ingestion.

Consumes `src.rag.ingest` and `src.rag.retrieve` (built in Batch 04).

### Run

From inside `llmops-session/`:

```bash
pip install -r requirements.txt
# Set ANTHROPIC_API_KEY in .env (see .env.example) for cells that call Anthropic
jupyter nbconvert --execute --to notebook --inplace notebooks/02_rag_pipeline.ipynb
```

Or open in Jupyter Lab and run cells top-to-bottom.

### Cells

- Cell 0 — title + narrative intro
- Cell 1–4 — setup, corpus ingestion (5 docs, 42 chunks into `data/chroma_nb02/`)
- Cells 5–7 — baseline retrieval: happy-path top-5 with scores and source priority
- Cells 8–11 — Failure 1: parametric knowledge vs. retrieved policy (hallucination mitigation)
- Cells 12–15 — Failure 2: wrong doc on top; fix via source-priority re-ranking
- Cells 16–18 — Failure 3: missing context; grounded-or-refuse helper with score threshold
- Cells 19–21 — Failure 4a: conflicting sources; priority-filter resolution
- Cells 22–24 — Failure 4b: chunk-size sensitivity (chunk_size=2000 vs 100 vs 500)
- Cells 25–26 — Failure 5: stale ingestion; old vs. new vector store comparison
- Cells 27–28 — Closing arc: full best-practice pipeline on one participant question; bridge to NB 03

### Cells that need an API key

Cell 9 (ungrounded LLM call) and Cell 28 (closing recap generation) need a valid `ANTHROPIC_API_KEY`. Without one they print `[skipped — no ANTHROPIC_API_KEY]` and the notebook continues. All other cells use `LLMClient(provider="mock")` or show retrieval output only.

