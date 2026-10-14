from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.patterns import hyde, multi_query
from astromesh_rag.testing import FakeLLM, HashEmbedder


async def test_multi_query_issues_variants_plus_original():
    llm = FakeLLM({"Rewrite": "1. cargo duplicado\n2. doble cobro\n"})
    q = multi_query(HashEmbedder(), MemoryIndex(), llm, variants=2)
    r = await q.query("me cobraron dos veces")
    assert sum(t["stage"] == "retrieve" for t in r.trace) == 3
    assert len(llm.calls) == 1


async def test_hyde_searches_with_hypothetical_text():
    llm = FakeLLM({"passage": "A duplicate charge is refunded in 5 days."})
    r = await hyde(HashEmbedder(), MemoryIndex(), llm).query("double charge?")
    assert any(t["stage"] == "transform" for t in r.trace)
