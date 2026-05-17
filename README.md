# LLM Ops — From Model Capability to Product Reliability

A 3-hour teaching session built around one running use case: the **Learning Program Support Assistant**, a document-grounded assistant that answers participant questions about a training program — and refuses or escalates when it shouldn't.

This repo contains the full session: the HTML theory microsite, six concept notebooks, a Streamlit MVP, the reusable Python modules behind them, and a CI/CD setup.

> **Status:** under active construction. Full setup instructions and run commands land with Section 15 (*Architecture & How to Run*).

## Quick start (preview — will be expanded)

```bash
# 1. Create a virtual environment and install deps
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# edit .env and paste your Anthropic API key into ANTHROPIC_API_KEY=

# 3. Open the theory microsite
# double-click web/index.html — works on file://, no server needed

# 4. Run the first notebook
jupyter notebook notebooks/01_core_llm_workflow.ipynb
```

## What's inside

```
web/         single-file HTML microsite (theory + visual story)
notebooks/   six concept notebooks (one per LLM Ops capability)
src/         reusable Python modules (llm, rag, workflow, evals,
             guardrails, observability, app, utils)
data/        synthetic program documents + golden query dataset
security/    red-team adversarial cases
scripts/     setup_check, run_evals, run_guardrail_tests
tests/       pytest coverage of config + datasets + guardrails
```

`src/observability/` contains the local trace, span, trace-store, and
session-metrics helpers used by the Batch 08 observability demos.

## License & data

All program documents under `data/` are synthetic. No real participant data is used.
