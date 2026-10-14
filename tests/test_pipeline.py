import asyncio

import pytest

from astromesh_rag.errors import StageError
from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.pipeline import IngestPipeline, Pipeline, QueryPipeline
from astromesh_rag.retrievers import BM25Index, VectorRetriever
from astromesh_rag.splitters import RecursiveSplitter
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Chunk, Document, Hit


def make(hybrid=False):
    emb, idx, bm = HashEmbedder(), MemoryIndex(), BM25Index()
    ing = IngestPipeline(
        RecursiveSplitter(chunk_size=40, overlap=0), emb, idx, bm if hybrid else None
    )
    rets = [VectorRetriever(emb, idx)] + ([bm] if hybrid else [])
    return Pipeline(ing, QueryPipeline(rets)), idx


async def test_two_documents_both_survive():
    p, _ = make()
    await p.ingest([Document("apples are red", id="a"), Document("bananas are yellow", id="b")])
    assert {h.chunk.doc_id for h in (await p.query("are", 10)).hits} == {"a", "b"}


async def test_reingest_with_fewer_chunks_removes_stale():
    p, _ = make()
    await p.ingest(
        [Document("one two three four five six seven eight nine ten eleven twelve", id="a")]
    )
    await p.ingest([Document("short", id="a")])
    assert [h.chunk.id for h in (await p.query("short one two", 10)).hits] == ["a#0"]


async def test_query_empty_index():
    p, _ = make()
    r = await p.query("anything")
    assert r.hits == []
    assert r.context == ""


async def test_query_blank_string():
    p, _ = make()
    assert (await p.query("   ")).hits == []


async def test_trace_records_each_stage():
    p, _ = make(hybrid=True)
    await p.ingest([Document("ERR-4012 billing failure", id="a")])
    stages = [t["stage"] for t in (await p.query("ERR-4012")).trace]
    assert stages.count("retrieve") == 2
    assert "fuse" in stages


class Boom:
    name = "boom"

    async def retrieve(self, q, k):
        raise RuntimeError("down")


async def test_hybrid_survives_one_failing_retriever():
    p, _ = make()
    await p.ingest([Document("apples", id="a")])
    q = QueryPipeline([p.query_pipeline.retrievers[0], Boom()])
    r = await q.query("apples")
    assert r.hits
    assert any(t.get("error") for t in r.trace)


async def test_all_retrievers_failing_raises():
    with pytest.raises(StageError):
        await QueryPipeline([Boom()]).query("x")


async def test_empty_document_clears_previous_chunks():
    p, _ = make()
    await p.ingest([Document("something here", id="a")])
    assert await p.ingest([Document("", id="a")]) == 0
    assert (await p.query("something")).hits == []


async def test_ingest_empty_list_and_blank_query_calls_nothing():
    p, _ = make()
    assert await p.ingest([]) == 0
    r = await p.query(" ")
    assert r.trace == []


class Slow:
    def __init__(self, name, delay, fail=False):
        self.name, self.delay, self.fail = name, delay, fail

    async def retrieve(self, q, k):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("down")
        return [Hit(Chunk(f"{self.name}#0", self.name, "x"), 1.0, self.name)]


async def test_retrieve_ms_is_per_retriever():
    qp = QueryPipeline([Slow("fast", 0.01), Slow("slow", 0.05), Slow("bad", 0.03, fail=True)])
    r = await qp.query("q")
    ms = {t["name"]: t["ms"] for t in r.trace if t["stage"] == "retrieve"}
    assert ms["fast"] < ms["slow"]
    assert ms["bad"] > 0
    assert any(t["name"] == "bad" and "error" in t for t in r.trace)


async def test_reingest_fewer_chunks_hybrid_clears_bm25():
    p, _ = make(hybrid=True)
    await p.ingest([Document("alpha beta gamma delta epsilon zeta eta theta iota kappa", id="a")])
    await p.ingest([Document("short", id="a")])
    bm = p.query_pipeline.retrievers[1]
    assert [h.chunk.id for h in await bm.retrieve("alpha beta short", 10)] == ["a#0"]


class Expand:
    async def transform(self, q):
        return ["bananas"]


async def test_transform_extra_queries_retrieved_and_fused():
    p, _ = make()
    await p.ingest([Document("apples are red", id="a"), Document("bananas yellow", id="b")])
    q = QueryPipeline(p.query_pipeline.retrievers, transform=Expand())
    r = await q.query("apples", 10)
    stages = [t["stage"] for t in r.trace]
    assert stages.count("retrieve") == 2
    assert "fuse" in stages
    assert {h.chunk.doc_id for h in r.hits} == {"a", "b"}


class Reverse:
    async def rerank(self, q, hits, k):
        return list(reversed(hits))[:k]


class BadRerank:
    async def rerank(self, q, hits, k):
        raise RuntimeError("nope")


async def test_reranker_reorders_and_truncates():
    p, _ = make()
    await p.ingest([Document(f"doc number {i}", id=f"d{i}") for i in range(4)])
    base = await QueryPipeline(p.query_pipeline.retrievers).query("doc number", 4)
    r = await QueryPipeline(p.query_pipeline.retrievers, reranker=Reverse()).query("doc number", 2)
    assert [h.chunk.id for h in r.hits] == [h.chunk.id for h in reversed(base.hits)][:2]
    assert len(r.hits) == 2


async def test_raising_reranker_is_stage_error():
    p, _ = make()
    await p.ingest([Document("apples", id="a")])
    with pytest.raises(StageError) as ei:
        await QueryPipeline(p.query_pipeline.retrievers, reranker=BadRerank()).query("apples")
    assert ei.value.stage == "rerank"


class CountingEmbedder(HashEmbedder):
    calls = 0

    async def embed(self, texts):
        type(self).calls += 1
        return await super().embed(texts)


async def test_batch_size_batches_embeds():
    emb = CountingEmbedder()
    CountingEmbedder.calls = 0
    ing = IngestPipeline(RecursiveSplitter(chunk_size=10, overlap=0), emb, MemoryIndex(), None, 2)
    n = await ing.ingest([Document("aaaaaaaaa " * 5, id="a")])
    assert n == 5
    assert CountingEmbedder.calls == 3
