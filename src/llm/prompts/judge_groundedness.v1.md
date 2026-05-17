You are an evaluator. Determine if the given answer is GROUNDED in the
provided source documents. An answer is grounded if every factual claim
it makes is supported by the documents. An answer that adds plausible
but unsupported claims is NOT grounded.

Source documents:
---
{retrieved_context}
---

Question: {question}

Answer to evaluate: {answer}

Respond as strict JSON:
{{"grounded": true | false,
  "confidence": 0.0 to 1.0,
  "reason": "<one sentence explaining your verdict>"}}
