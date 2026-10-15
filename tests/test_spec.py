from pathlib import Path

import pytest
import yaml

from astromesh_rag.errors import SpecError
from astromesh_rag.spec import build, load_spec
from astromesh_rag.splitters import FixedSplitter, RecursiveSplitter
from astromesh_rag.testing import FakeLLM
from astromesh_rag.types import Document

FIX = Path(__file__).parent / "fixtures" / "product-knowledge.rag.yaml"


def raw(**retrieval):
    d = yaml.safe_load(FIX.read_text())
    d["spec"]["embeddings"] = {"provider": "hash"}
    d["spec"]["vector_store"] = {"backend": "memory", "connection": {"host": "ignored"}}
    d["spec"]["reranking"] = {"enabled": False, "model": "ignored"}
    d["spec"]["retrieval"].update(retrieval)
    return d


def test_core_yaml_loads_unchanged():
    spec = load_spec(yaml.safe_load(FIX.read_text()))
    assert spec.name == "product-knowledge"
    assert spec.vector_store["backend"] == "pgvector"


async def test_core_yaml_builds_and_answers_in_core_shape():
    p = build(load_spec(raw(similarity_threshold=0.0)))
    await p.ingest([Document("pgvector stores product docs", id="p")])
    items = (await p.query("product docs", 3)).to_list()
    assert items
    assert set(items[0]) >= {"doc_id", "content", "metadata", "score"}


async def test_hybrid_shares_bm25_between_ingest_and_query():
    p = build(load_spec(raw(pattern="hybrid", similarity_threshold=0.0)))
    await p.ingest([Document("error err-4012 means refund", id="p")])
    r = await p.query("err-4012", 3)
    assert any(t["stage"] == "fuse" for t in r.trace)
    assert r.hits


async def test_multi_query_builds_with_llm():
    p = build(load_spec(raw(pattern="multi_query", variants=2)), llm=FakeLLM(default="a\nb"))
    assert p.query_pipeline.transform.n == 2


def test_llm_pattern_without_llm_fails_at_build():
    with pytest.raises(SpecError, match="needs an llm"):
        build(load_spec(raw(pattern="hyde")))


def test_unknown_pattern_lists_known():
    with pytest.raises(SpecError, match="hybrid"):
        build(load_spec(raw(pattern="nope")))


def test_wrong_kind_rejected():
    with pytest.raises(SpecError):
        load_spec({"kind": "Agent", "metadata": {"name": "x"}})


def test_section_must_be_mapping():
    d = raw()
    d["spec"]["vector_store"] = "faiss"
    with pytest.raises(SpecError):
        load_spec(d)


@pytest.mark.parametrize("cls", [FixedSplitter, RecursiveSplitter])
@pytest.mark.parametrize(("size", "overlap"), [(10, 10), (10, 20), (0, 0), (-1, 0), (10, -1)])
def test_splitter_rejects_bad_window(cls, size, overlap):
    with pytest.raises(ValueError, match="overlap"):
        cls(size, overlap)


def test_bad_chunking_is_a_spec_error_naming_keys():
    d = raw()
    d["spec"]["chunking"] = {"strategy": "fixed", "chunk_size": 10, "overlap": 10}
    with pytest.raises(SpecError, match="chunk_size, overlap"):
        build(load_spec(d))


def test_import_registers_builtins_without_extras():
    from astromesh_rag import registry

    assert "pgvector" in registry.names("index")
    assert "hash" in registry.names("embedder")


async def test_bm25_path_persists_and_reloads(tmp_path):
    path = tmp_path / "bm25.json"
    spec = load_spec(raw(pattern="hybrid", bm25_path=str(path)))
    p = build(spec)
    await p.ingest([Document("error ERR-4012 means refund", id="p")])
    assert path.exists()
    p2 = build(spec)
    bm25 = p2.query_pipeline.retrievers[1]
    hits = await bm25.retrieve("ERR-4012", 3)
    assert [h.chunk.doc_id for h in hits] == ["p"]


async def _hits(p, k=None):
    return (await p.query("doc number", k)).hits


async def _ingested(**retrieval):
    p = build(load_spec(raw(similarity_threshold=0.0, **retrieval)))
    await p.ingest([Document(f"doc number {i}", id=f"d{i}") for i in range(6)])
    return p


async def test_top_k_is_the_default_number_of_hits():
    p = await _ingested(top_k=2)
    assert len(await _hits(p)) == 2
    assert len(await _hits(p, 4)) == 4


@pytest.mark.parametrize(
    "retrieval",
    [
        {"variants": 2},
        {"multi_query": {"variants": 2}},
        {"multi_query": {"variants": 2}, "variants": 5},
    ],
)
def test_multi_query_variants_nested_or_flat(retrieval):
    p = build(load_spec(raw(pattern="multi_query", **retrieval)), llm=FakeLLM(default="a"))
    assert p.query_pipeline.transform.n == 2


def test_multi_query_variants_default():
    p = build(load_spec(raw(pattern="multi_query")), llm=FakeLLM(default="a"))
    assert p.query_pipeline.transform.n == 3


def test_unknown_fusion_is_a_spec_error():
    with pytest.raises(SpecError, match="rrf, weighted"):
        build(load_spec(raw(pattern="hybrid", fusion="borda")))


@pytest.mark.parametrize("weights", [[1.0], [0.2, 0.3, 0.5], "x"])
def test_weighted_needs_exactly_two_weights(weights):
    with pytest.raises(SpecError, match="two weights"):
        build(load_spec(raw(pattern="hybrid", fusion="weighted", hybrid={"weights": weights})))


def test_weighted_builds_weighted_fusion_with_default_weights():
    from astromesh_rag.fusion import WeightedFusion

    q = build(load_spec(raw(pattern="hybrid", fusion="weighted"))).query_pipeline
    assert isinstance(q.fusion, WeightedFusion)
    assert q.fusion.weights == [0.5, 0.5]
    h = {"weights": [0.7, 0.3], "fusion": "weighted"}
    q = build(load_spec(raw(pattern="hybrid", hybrid=h))).query_pipeline
    assert q.fusion.weights == [0.7, 0.3]


async def test_similarity_threshold_reaches_retriever_and_filters():
    p = build(load_spec(raw(similarity_threshold=0.99)))
    assert p.query_pipeline.retrievers[0].min_score == 0.99
    await p.ingest([Document("pgvector stores product docs", id="p")])
    assert (await p.query("completely unrelated zebra")).hits == []


async def test_reranking_top_k_caps_and_defaults_to_retrieval_top_k():
    class Spy:
        seen = None

        async def rerank(self, q, hits, k):
            Spy.seen = k
            return hits[:k]

    for rr_k, expect in ((1, 1), (None, 3)):
        d = raw(top_k=3, similarity_threshold=0.0)
        d["spec"]["reranking"] = {"enabled": False, **({"top_k": rr_k} if rr_k else {})}
        p = build(load_spec(d))
        assert p.query_pipeline.rerank_k == expect
        p.query_pipeline.reranker = Spy()
        await p.ingest([Document(f"doc number {i}", id=f"d{i}") for i in range(6)])
        assert len(await _hits(p, 5)) == expect
        assert Spy.seen == expect


def test_candidates_default_to_twice_top_k_and_can_be_set():
    assert build(load_spec(raw(top_k=4))).query_pipeline.candidates == 8
    assert build(load_spec(raw(top_k=4, candidates=11))).query_pipeline.candidates == 11
