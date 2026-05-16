"""Deterministic mock provider used for tests and offline development."""

from __future__ import annotations

import hashlib


class MockProvider:
    def __init__(self, model: str = "mock-model-v1") -> None:
        self.model = model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> dict:
        """Return a deterministic canned response for the prompt."""

        seed = f"{system or ''}\x1f{prompt}"
        fingerprint = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
        snippet = " ".join(prompt.strip().split()) or "<empty>"
        suffix = "..." if len(snippet) > 48 else ""
        text = f"[mock:{fingerprint}] echo: {snippet[:48]}{suffix}"[:200]
        return {
            "text": text,
            "tokens_in": max(1, len((system or "") + prompt) // 4),
            "tokens_out": max(1, len(text) // 4),
            "raw": {
                "fingerprint": fingerprint,
                "prompt": prompt,
                "system": system,
                "kwargs": kwargs,
            },
            "model": self.model,
        }
