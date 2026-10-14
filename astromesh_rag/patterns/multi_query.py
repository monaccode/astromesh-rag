from __future__ import annotations

import re

from ..pipeline import QueryPipeline
from ..retrievers import VectorRetriever

MULTI_QUERY_PROMPT = (
    "Rewrite the question in {n} different ways that keep its meaning, "
    "one per line, with no numbering or commentary.\nQuestion: {query}"
)


class LLMQueryVariants:
    name = "multi_query"

    def __init__(self, llm, n: int = 3):
        self.llm, self.n = llm, n

    async def transform(self, query: str) -> list[str]:
        out = await self.llm.complete(MULTI_QUERY_PROMPT.format(n=self.n, query=query))
        lines = (re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip() for ln in out.splitlines())
        return [ln for ln in lines if ln][: self.n]


def multi_query(
    embedder,
    index,
    llm,
    *,
    variants: int = 3,
    reranker=None,
    rerank_k=None,
    min_score=None,
    candidates=None,
) -> QueryPipeline:
    return QueryPipeline(
        [VectorRetriever(embedder, index, min_score)],
        transform=LLMQueryVariants(llm, variants),
        reranker=reranker,
        rerank_k=rerank_k,
        candidates=candidates,
    )
