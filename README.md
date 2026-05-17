# LLM Ops — From Model Capability to Product Reliability

A 3-hour teaching session built around one running use case: the **Learning Program Support Assistant**, a document-grounded assistant that answers participant questions about a training program — and refuses or escalates when it shouldn't.

This repo contains the full session: a 16-section HTML theory microsite, six concept notebooks, a Streamlit MVP that composes all the layers, and the reusable Python modules behind them.

## Quick start

```bash
# 1. Create a virtual environment and install deps
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY (skip for mock provider)

# 3. Sanity check
python scripts/setup_check.py

# 4. Open the theory microsite
# double-click web/index.html — works on file://, no server needed

# 5. Run a notebook (any of 01..06)
jupyter notebook notebooks/01_core_llm_workflow.ipynb

# 6. Run the MVP app
streamlit run src/app/streamlit_app.py
# opens at http://localhost:8501
```

## Run the MVP

From `llmops-session/`, launch the one-screen Streamlit MVP with:

```bash
streamlit run src/app/streamlit_app.py
```

The MVP demonstrates how the course's six `src/` layers land as one product surface: a single screen with question, answer, trace, and history panels, backed by the existing `src/llm`, `src/rag`, `src/workflow`, `src/guardrails`, `src/observability`, and `src/evals` architecture described in `dev-docs/PLAN.md` Section S-13. The app itself stays a thin composition shell rather than introducing new business logic.

## What's inside

```
web/         16-section HTML theory microsite (single file, file:// works)
notebooks/   six concept notebooks (one per LLM Ops capability)
             01 core LLM workflow · 02 RAG · 03 workflow orchestration
             04 evaluation · 05 guardrails · 06 observability
src/         reusable Python modules
             llm/  rag/  workflow/  evals/  guardrails/
             observability/  app/  utils/
data/        synthetic program documents + 12-row golden query dataset
security/    red-team adversarial cases
scripts/     setup_check (env health), plus CI runners for evals + guardrails
tests/       ~175 pytest tests covering every module + the MVP
```

## The composition (what the MVP does)

```
user question
    ↓
run_input_guardrails       # block obvious adversarial inputs
    ↓
trace_workflow(run_workflow, ...)   # classify → route → answer / refuse / escalate
    ↓
run_output_guardrails      # block PII / system-prompt leaks
    ↓
display answer + trace
```

Every layer is one module under `src/`, demoed in isolation in one notebook (NB 01..06), then composed in `src/app/streamlit_app.py`. The HTML microsite is the teaching narrative around it.

## License & data

All program documents under `data/` are synthetic. No real participant data is used.
