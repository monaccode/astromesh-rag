"""Context building, same format as the core's format_knowledge."""

from __future__ import annotations

from .types import Hit


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(t for t in (h.chunk.text.strip() for h in hits) if t)
