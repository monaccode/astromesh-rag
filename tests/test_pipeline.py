import pytest

from astromesh_rag.errors import StageError
from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.pipeline import IngestPipeline, Pipeline, QueryPipeline
from astromesh_rag.retrievers import BM25Index, VectorRetriever
from astromesh_rag.splitters import RecursiveSplitter
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Document


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
