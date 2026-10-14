from __future__ import annotations

from ..pipeline import QueryPipeline
from ..retrievers import VectorRetriever


def hybrid(
    embedder,
    index,
    bm25,
    *,
    fusion=None,
    reranker=None,
    rerank_k=None,
    min_score=None,
    candidates=None,
) -> QueryPipeline:
    return QueryPipeline(
        [VectorRetriever(embedder, index, min_score), bm25],
        fusion=fusion,
        reranker=reranker,
        rerank_k=rerank_k,
        candidates=candidates,
    )
