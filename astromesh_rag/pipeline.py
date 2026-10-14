"""Ingest and query pipelines. Every query stage appends an entry to the trace."""

from __future__ import annotations

import asyncio
import time

from .context import build_context
from .errors import StageError
from .fusion import RRF
from .types import QueryResult


def _name(obj) -> str:
    if isinstance(obj, str):
        return obj
    return getattr(obj, "name", None) or type(obj).__name__


class IngestPipeline:
    def __init__(self, splitter, embedder, index, keyword_index=None, batch_size: int = 64):
        self.splitter = splitter
        self.embedder = embedder
        self.index = index
        self.keyword_index = keyword_index
        self.batch_size = batch_size

    async def ingest(self, docs) -> int:
        total = 0
        for doc in docs:
            doc_id = doc.resolved_id()
            await self.index.delete_document(doc_id)
            if self.keyword_index is not None:
                await self.keyword_index.delete_document(doc_id)
            chunks = await self.splitter.split(type(doc)(doc.text, doc_id, doc.metadata))
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                vectors = await self.embedder.embed([c.text for c in batch])
                await self.index.upsert(batch, vectors)
            if self.keyword_index is not None and chunks:
                await self.keyword_index.add(chunks)
            total += len(chunks)
        return total


class QueryPipeline:
    def __init__(
        self,
        retrievers,
        *,
        transform=None,
        include_original: bool = True,
        fusion=None,
        reranker=None,
        rerank_k: int | None = None,
        candidates: int | None = None,
    ):
        self.retrievers = retrievers
        self.transform = transform
        self.include_original = include_original
        self.fusion = fusion or RRF()
        self.reranker = reranker
        self.rerank_k = rerank_k
        self.candidates = candidates

    async def query(self, query: str, k: int = 5) -> QueryResult:
        result = QueryResult(query=query)
        if not query.strip():
            return result
        trace = result.trace

        def log(stage, comp, t0, n_in, n_out, error=None):
            e = {"stage": stage, "name": _name(comp), "ms": (time.perf_counter() - t0) * 1000}
            e |= {"in": n_in, "out": n_out}
            if error:
                e["error"] = error
            trace.append(e)

        queries = [query]
        if self.transform is not None:
            t0 = time.perf_counter()
            try:
                extra = await self.transform.transform(query)
            except Exception as exc:
                log("transform", self.transform, t0, 1, 0, str(exc))
                raise StageError("transform", _name(self.transform), exc) from exc
            queries = ([query] if self.include_original else []) + [
                q for q in extra if not (self.include_original and q == query)
            ]
            log("transform", self.transform, t0, 1, len(queries))

        n = self.candidates or k * 2

        async def one(q, r):
            t0 = time.perf_counter()
            return t0, await r.retrieve(q, n)

        pairs = [(q, r) for q in queries for r in self.retrievers]
        outs = await asyncio.gather(*(one(q, r) for q, r in pairs), return_exceptions=True)
        lists, last_exc = [], None
        for (_, r), out in zip(pairs, outs, strict=True):
            if isinstance(out, BaseException):
                if not isinstance(out, Exception):
                    raise out
                last_exc = out
                log("retrieve", r, time.perf_counter(), 1, 0, str(out))
            else:
                t0, hits = out
                log("retrieve", r, t0, 1, len(hits))
                lists.append(hits)
        if not lists:
            names = ",".join(_name(r) for r in self.retrievers)
            raise StageError("retrieve", names, last_exc)

        if len(lists) == 1:
            hits = lists[0]
        else:
            t0 = time.perf_counter()
            hits = self.fusion.fuse(lists, n)
            log("fuse", self.fusion, t0, sum(map(len, lists)), len(hits))

        if self.reranker is not None:
            t0 = time.perf_counter()
            try:
                out = await self.reranker.rerank(query, hits, self.rerank_k or k)
            except Exception as exc:
                log("rerank", self.reranker, t0, len(hits), 0, str(exc))
                raise StageError("rerank", _name(self.reranker), exc) from exc
            log("rerank", self.reranker, t0, len(hits), len(out))
            hits = out

        t0 = time.perf_counter()
        result.hits = hits[:k]
        result.context = build_context(result.hits)
        log("context", "context", t0, len(hits), len(result.hits))
        return result


class Pipeline:
    def __init__(self, ingest: IngestPipeline, query: QueryPipeline):
        self.ingest_pipeline = ingest
        self.query_pipeline = query

    async def ingest(self, docs) -> int:
        return await self.ingest_pipeline.ingest(docs)

    async def query(self, q: str, k: int = 5) -> QueryResult:
        return await self.query_pipeline.query(q, k)
