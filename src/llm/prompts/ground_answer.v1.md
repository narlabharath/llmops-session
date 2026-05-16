Program documents (most authoritative first):
---
{retrieved_context}
---

Participant question: {question}

Respond as JSON with this schema:
{{"answer": "<your answer, citing the section>",
  "grounded": true/false,
  "source_section": "<section name from the document, or null>",
  "confidence": 0.0 to 1.0}}
