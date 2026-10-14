"""Rank fusion over several hit lists. Sync."""

from __future__ import annotations

from .types import Hit


def _merge(lists: list[list[Hit]], scores: dict[str, float], key: str, k: int) -> list[Hit]:
    chunks, sources, details = {}, {}, {}
    for lst in lists:
        for h in lst:
            chunks.setdefault(h.chunk.id, h.chunk)
            sources.setdefault(h.chunk.id, set()).add(h.source)
            details.setdefault(h.chunk.id, {}).update(h.details)
    out = []
    for cid, s in scores.items():
        d = {**details[cid], key: s}
        out.append(Hit(chunks[cid], s, "+".join(sorted(sources[cid])), d))
    out.sort(key=lambda h: (-h.score, h.chunk.id))
    return out[:k]


class RRF:
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf

    def fuse(self, lists: list[list[Hit]], k: int) -> list[Hit]:
        scores: dict[str, float] = {}
        for lst in lists:
            for rank, h in enumerate(lst, 1):
                scores[h.chunk.id] = scores.get(h.chunk.id, 0.0) + 1 / (self.k_rrf + rank)
        return _merge(lists, scores, "rrf", k)


class WeightedFusion:
    def __init__(self, weights: list[float]):
        self.weights = weights

    def fuse(self, lists: list[list[Hit]], k: int) -> list[Hit]:
        scores: dict[str, float] = {}
        for w, lst in zip(self.weights, lists, strict=True):
            if not lst:
                continue
            lo, hi = min(h.score for h in lst), max(h.score for h in lst)
            for h in lst:
                n = (h.score - lo) / (hi - lo) if hi > lo else 1.0
                scores[h.chunk.id] = scores.get(h.chunk.id, 0.0) + w * n
        return _merge(lists, scores, "fused", k)
