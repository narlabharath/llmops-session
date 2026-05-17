# Notebooks

## 01_core_llm_workflow.ipynb

Demonstrates the core LLM workflow against a real Anthropic backend.

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

---

## 03_workflow_orchestration.ipynb

Demonstrates the workflow orchestration layer end-to-end on top of the sample program corpus: classify, route, retrieve, answer, refuse, escalate, and observe workflow-level cache behavior.

Consumes `src.workflow.run_workflow` and `src.workflow.build_workflow_graph`, building on the retrieval setup from NB 02.

### Run

From inside `llmops-session/`:

```bash
pip install -r requirements.txt
# Optionally set ANTHROPIC_API_KEY in .env to use Anthropic in cell 8
jupyter nbconvert --execute --to notebook --inplace notebooks/03_workflow_orchestration.ipynb
```

Or open in Jupyter Lab and run cells top-to-bottom.

### Cells

- Cells 0–6 — setup, corpus ingestion into `data/chroma_nb03/`, workflow graph inspection
- Cells 7–10 — happy path and trace inspection
- Cells 11–14 — refusal routing for out-of-scope, private-request, and injection attempts
- Cells 15–18 — escalation on low-confidence retrieval and threshold contrast
- Cells 19–22 — workflow cache hit/miss and prompt-version invalidation
- Cells 23–26 — side-by-side recap of answered/refused/escalated outcomes and bridge to NB 04

### Cells that need an API key

Cell 8 uses Anthropic when `ANTHROPIC_API_KEY` is present; without one it falls back to `LLMClient(provider="mock")` and the notebook still executes. Cells 12, 16, 18, 20, 22, and 24 use notebook-local deterministic helpers, so they do not require a key.

---

## 04_evaluation.ipynb

Demonstrates the evaluation harness end-to-end against the program-assistant workflow: load the golden CSV, run the eval pass, build and dump reports, inspect groundedness judging, and compare a deliberate regression against the baseline.

Consumes `src.evals.run_evals`, `src.evals.build_report`, `src.evals.print_report`, `src.evals.dump_report`, and `src.evals.judge_groundedness`, building on `src.workflow.run_workflow` and the corpus setup from earlier notebooks.

### Run

From inside `llmops-session/`:

```bash
pip install -r requirements.txt
# Optionally set ANTHROPIC_API_KEY in .env to use Anthropic in the eval and judge cells
jupyter nbconvert --execute --to notebook --inplace notebooks/04_evaluation.ipynb
```

Or open in Jupyter Lab and run cells top-to-bottom.

### Cells

- Cells 0–5 — setup, golden CSV loading, and notebook scaffolding
- Cells 6–9 — corpus ingestion, eval pass, and per-row result table
- Cells 10–13 — report aggregation and artifact dump for CI-style consumption
- Cells 14–17 — groundedness verdict inspection and overconfident-answer judge demo
- Cells 18–21 — deliberate regression run and baseline-vs-regression diff
- Cells 22–25 — ship recap, NB 05 bridge, and API-key note

### Cells that need an API key

Cell 17 requires a valid `ANTHROPIC_API_KEY` for the live groundedness judge demo; without one it prints `[skipped — no key]`. Cells 8 and 19 use Anthropic when a key is present, but fall back to `LLMClient(provider="mock")` so the notebook still executes without one.
