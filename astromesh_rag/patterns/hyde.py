from __future__ import annotations

from ..pipeline import QueryPipeline
from ..retrievers import VectorRetriever

HYDE_PROMPT = (
    "Write a short passage that answers the question, as it might appear in a "
    "reference document.\nQuestion: {query}"
)


class HypotheticalDocument:
    name = "hyde"

    def __init__(self, llm):
        self.llm = llm

    async def transform(self, query: str) -> list[str]:
        return [(await self.llm.complete(HYDE_PROMPT.format(query=query))).strip()]


def hyde(
    embedder, index, llm, *, reranker=None, rerank_k=None, min_score=None, candidates=None
) -> QueryPipeline:
    return QueryPipeline(
        [VectorRetriever(embedder, index, min_score)],
        transform=HypotheticalDocument(llm),
        reranker=reranker,
        rerank_k=rerank_k,
        candidates=candidates,
    )
