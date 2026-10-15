import sys
import threading
import time
import types
import warnings

import pytest

from astromesh_rag.retrievers import BM25Index
from astromesh_rag.types import Chunk


async def test_bm25_save_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "bm25.json"
    idx = BM25Index(path=str(path))
    await idx.add([Chunk("a#0", "a", "hello world")])
    idx.save()
    assert [p.name for p in tmp_path.iterdir()] == ["bm25.json"]
    assert [h.chunk.id for h in await BM25Index.load(str(path)).retrieve("hello", 3)] == ["a#0"]


def test_bm25_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "bm25.json"
    idx = BM25Index(path=str(path))
    idx.save()
    before = path.read_text()
    monkeypatch.setattr("json.dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        idx.save()
    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["bm25.json"]


def _fake_st(monkeypatch, loads, dim_attr):
    class Model:
        def encode(self, texts):
            import numpy as np

            return np.zeros((len(texts), 2))

        def predict(self, pairs):
            return [0.5] * len(pairs)

    setattr(Model, dim_attr, lambda self: 2)

    def make(name):
        loads.append(name)
        time.sleep(0.05)
        return Model()

    mod = types.SimpleNamespace(SentenceTransformer=make, CrossEncoder=make)
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)


def test_embedder_loads_once_under_threads_and_uses_new_dimension_api(monkeypatch):
    pytest.importorskip("numpy")
    from astromesh_rag.embedders import SentenceTransformersEmbedder

    loads = []
    _fake_st(monkeypatch, loads, "get_embedding_dimension")  # no old-name method at all
    emb = SentenceTransformersEmbedder()
    ts = [threading.Thread(target=emb._load) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert loads == ["all-MiniLM-L6-v2"]
    assert emb.dimension == 2


def test_embedder_falls_back_to_old_dimension_api(monkeypatch):
    from astromesh_rag.embedders import SentenceTransformersEmbedder

    _fake_st(monkeypatch, [], "get_sentence_embedding_dimension")
    emb = SentenceTransformersEmbedder()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emb._load()
    assert emb.dimension == 2


async def test_cross_encoder_loads_once_under_concurrency(monkeypatch):
    import asyncio

    from astromesh_rag.rerankers import CrossEncoderReranker
    from astromesh_rag.types import Hit

    loads = []
    _fake_st(monkeypatch, loads, "get_embedding_dimension")
    rr = CrossEncoderReranker()
    hits = [Hit(Chunk("a#0", "a", "x"), 0.1, "vector")]
    await asyncio.gather(*(rr.rerank("q", hits, 1) for _ in range(4)))
    assert len(loads) == 1
