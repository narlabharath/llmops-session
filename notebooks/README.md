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
