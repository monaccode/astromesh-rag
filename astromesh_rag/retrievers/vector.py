"""Dense retriever: embed the query, search the vector index."""

from __future__ import annotations

from ..types import Hit


class VectorRetriever:
    name = "vector"

    def __init__(
        self, embedder, index, min_score: float | None = None, filters: dict | None = None
    ):
        self.embedder = embedder
        self.index = index
        self.min_score = min_score
        self.filters = filters

    async def retrieve(self, query: str, k: int) -> list[Hit]:
        (vec,) = await self.embedder.embed([query])
        hits = await self.index.search(vec, k, self.filters)
        if self.min_score is None:
            return hits
        return [h for h in hits if h.score >= self.min_score]
