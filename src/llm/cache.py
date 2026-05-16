"""Small in-memory cache for LLMClient completions.

NOT provider-side prompt caching (which is a different concept, such as
Anthropic's `cache_control` API). This cache stores full completions keyed by
fingerprints of the caller input. Default scope: per-process.

Invalidation: bump the `prompt_version` passed to LLMClient, change the model,
or change the prompt/system text. Any of those produces a different key and a
miss.

Optional observability hook: pass `on_cache_event` to receive `"hit"`,
`"miss"`, or `"bypass"` after each cache decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .client import CompletionResult


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


CacheEventCallback = Callable[[str], None]


@dataclass
class LLMCache:
    """In-memory response cache for LLMClient with optional event callbacks."""

    maxsize: int = 1024
    on_cache_event: CacheEventCallback | None = None
    _store: dict[str, "CompletionResult"] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _hits: int = 0
    _misses: int = 0

    def make_key(
        self,
        prompt: str,
        system: str | None,
        model: str,
        prompt_version: str,
        kwargs: dict,
    ) -> str:
        """Stable key from input. SHA-256 hex truncated to 16 chars per component."""

        kwargs_blob = json.dumps(kwargs, sort_keys=True, separators=(",", ":"), default=str)
        parts = (
            _fingerprint(prompt),
            _fingerprint(system or ""),
            _fingerprint(model),
            _fingerprint(prompt_version),
            _fingerprint(kwargs_blob),
        )
        return "\x1f".join(parts)

    def get(self, key: str) -> "CompletionResult" | None:
        value = self._store.get(key)
        if value is None:
            self._misses += 1
            self._emit_event("miss")
            return None
        self._hits += 1
        self._emit_event("hit")
        return value

    def set(self, key: str, value: "CompletionResult") -> None:
        if key not in self._store:
            self._order.append(key)
            if len(self._order) > self.maxsize:
                evicted = self._order.pop(0)
                self._store.pop(evicted, None)
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
        self._hits = 0
        self._misses = 0

    def record_bypass(self) -> None:
        self._emit_event("bypass")

    def stats(self) -> dict[str, int]:
        """Returns {'size': N, 'hits': N, 'misses': N}."""

        return {"size": len(self._store), "hits": self._hits, "misses": self._misses}

    def _emit_event(self, status: str) -> None:
        if self.on_cache_event is not None:
            self.on_cache_event(status)
