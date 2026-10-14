"""FAISS flat inner-product index over normalized vectors (cosine)."""

from __future__ import annotations

from .._extras import require
from ..types import Chunk, Hit


class FaissIndex:
    def __init__(self, dimension: int):
        self._faiss = require("faiss", "faiss")
        self._np = require("numpy", "faiss")
        self.dimension = dimension
        self._index = self._faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk | None] = []  # position -> chunk; None = tombstone
        self._pos: dict[str, int] = {}  # chunk id -> position
        self._doc_chunks: dict[str, list[int]] = {}

    def _vec(self, vectors):
        a = self._np.array(vectors, dtype=self._np.float32)
        n = self._np.linalg.norm(a, axis=1, keepdims=True)
        return a / self._np.where(n > 0, n, 1)

    def _tombstone(self, positions):
        for p in positions:
            c = self._chunks[p]
            if c is not None:
                self._pos.pop(c.id, None)
            self._chunks[p] = None

    async def upsert(self, chunks, vectors):
        if not chunks:
            return
        self._tombstone([self._pos[c.id] for c in chunks if c.id in self._pos])
        start = self._index.ntotal
        self._index.add(self._vec(vectors))
        for i, c in enumerate(chunks):
            self._chunks.append(c)
            self._pos[c.id] = start + i
            self._doc_chunks.setdefault(c.doc_id, []).append(start + i)

    async def search(self, vector, k, filters=None):
        if self._index.ntotal == 0:
            return []
        # over-fetch everything: tombstones and filters shrink the result
        scores, idxs = self._index.search(self._vec([vector]), self._index.ntotal)
        hits: list[Hit] = []
        for s, i in zip(scores[0], idxs[0], strict=False):
            c = self._chunks[i] if i >= 0 else None
            if c is None:
                continue
            if filters and any(c.metadata.get(fk) != fv for fk, fv in filters.items()):
                continue
            hits.append(Hit(chunk=c, score=float(s), source="vector", details={"vector": float(s)}))
            if len(hits) >= k:
                break
        return hits

    async def delete_document(self, doc_id):
        self._tombstone(self._doc_chunks.pop(doc_id, []))
