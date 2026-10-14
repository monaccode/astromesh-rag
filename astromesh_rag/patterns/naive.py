from __future__ import annotations

from ..pipeline import QueryPipeline
from ..retrievers import VectorRetriever


def naive(
    embedder, index, *, reranker=None, rerank_k=None, min_score=None, candidates=None
) -> QueryPipeline:
    return QueryPipeline(
        [VectorRetriever(embedder, index, min_score)],
        reranker=reranker,
        rerank_k=rerank_k,
        candidates=candidates,
    )
