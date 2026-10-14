import pytest

pytest.importorskip("sentence_transformers")

from astromesh_rag.embedders import SentenceTransformersEmbedder
from astromesh_rag.rerankers import CrossEncoderReranker
from astromesh_rag.types import Chunk, Hit


async def test_st_embeds_with_dimension():
    e = SentenceTransformersEmbedder()
    vecs = await e.embed(["hello", "world"])
    assert len(vecs) == 2
    assert e.dimension == len(vecs[0]) == 384


async def test_cross_encoder_ranks_relevant_first():
    hits = [
        Hit(Chunk(f"d#{i}", "d", t), 0.0, "vector")
        for i, t in enumerate(["Bananas are yellow", "Paris is the capital of France"])
    ]
    out = await CrossEncoderReranker().rerank("capital of France", hits, 2)
    assert out[0].chunk.text.startswith("Paris")
    assert "rerank" in out[0].details
