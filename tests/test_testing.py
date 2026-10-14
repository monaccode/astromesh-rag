from astromesh_rag.testing import FakeLLM, HashEmbedder


async def test_hash_embedder_is_deterministic_and_normalized():
    a, b = await HashEmbedder().embed(["red apple", "red apple"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1) < 1e-9


async def test_fake_llm_matches_by_substring():
    llm = FakeLLM({"rewrite": "v1\nv2"}, default="none")
    assert await llm.complete("please rewrite this") == "v1\nv2"
    assert await llm.complete("other") == "none"
    assert len(llm.calls) == 2
