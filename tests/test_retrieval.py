from astromesh_rag.fusion import RRF, WeightedFusion
from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.retrievers import BM25Index, VectorRetriever, tokenize
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Chunk, Hit


def c(i, t):
    return Chunk(f"d{i}#0", f"d{i}", t)


def test_tokenize_keeps_codes_and_strips_accents():
    toks = tokenize("Error ERR-4012 en facturación")
    assert {"err-4012", "facturacion"} <= set(toks)


async def test_bm25_finds_exact_code():
    bm = BM25Index()
    await bm.add(
        [c(1, "billing failed with ERR-4012"), c(2, "billing works fine"), c(3, "ERR-5000 timeout")]
    )
    hits = await bm.retrieve("what is ERR-4012", 2)
    assert (hits[0].chunk.doc_id, hits[0].source) == ("d1", "bm25")


async def test_bm25_delete_and_persist(tmp_path):
    bm = BM25Index(path=str(tmp_path / "bm25.json"))
    await bm.add([c(1, "alpha"), c(2, "beta")])
    await bm.delete_document("d1")
    bm.save()
    again = BM25Index.load(str(tmp_path / "bm25.json"))
    assert [h.chunk.doc_id for h in await again.retrieve("alpha beta", 5)] == ["d2"]


def test_rrf_rewards_agreement():
    a = [Hit(c(1, "x"), 0.9, "vector"), Hit(c(2, "y"), 0.8, "vector")]
    b = [Hit(c(2, "y"), 7.0, "bm25"), Hit(c(3, "z"), 6.0, "bm25")]
    out = RRF().fuse([a, b], 3)
    assert (out[0].chunk.id, out[0].source) == ("d2#0", "bm25+vector")
    assert "rrf" in out[0].details


def test_weighted_fusion_respects_weights():
    a = [Hit(c(1, "x"), 1.0, "vector"), Hit(c(2, "y"), 0.0, "vector")]
    b = [Hit(c(2, "y"), 1.0, "bm25"), Hit(c(1, "x"), 0.0, "bm25")]
    assert WeightedFusion([0.9, 0.1]).fuse([a, b], 1)[0].chunk.id == "d1#0"


async def test_vector_retriever_threshold():
    idx, emb = MemoryIndex(), HashEmbedder()
    chunks = [c(1, "red apple pie"), c(2, "quantum physics")]
    await idx.upsert(chunks, await emb.embed([c.text for c in chunks]))
    hits = await VectorRetriever(emb, idx, min_score=0.5).retrieve("red apple", 5)
    assert [h.chunk.doc_id for h in hits] == ["d1"]


def test_build_context_strips_and_drops_empty():
    from astromesh_rag.context import build_context

    hits = [Hit(c(1, " a \n"), 1, "v"), Hit(c(2, "  "), 1, "v"), Hit(c(3, "b"), 1, "v")]
    assert build_context(hits) == "a\n\nb"


async def test_bm25_no_match_and_save_without_path():
    import pytest

    bm = BM25Index()
    await bm.add([c(1, "alpha")])
    assert await bm.retrieve("zzz", 3) == []
    with pytest.raises(ValueError, match="path"):
        bm.save()
