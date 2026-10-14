"""Pure-Python cosine index."""

from __future__ import annotations

import math

from ..types import Chunk, Hit


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


class MemoryIndex:
    def __init__(self):
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks, vectors):
        for c, v in zip(chunks, vectors, strict=True):
            self._items[c.id] = (c, _norm(v))

    async def search(self, vector, k, filters=None):
        q = _norm(vector)
        hits = []
        for c, v in self._items.values():
            if filters and any(c.metadata.get(fk) != fv for fk, fv in filters.items()):
                continue
            s = sum(a * b for a, b in zip(q, v, strict=True))
            hits.append(Hit(chunk=c, score=s, source="vector", details={"vector": s}))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    async def delete_document(self, doc_id):
        self._items = {i: cv for i, cv in self._items.items() if cv[0].doc_id != doc_id}
