"""Shared setup for the examples and the benchmark: corpus, models from env, printing.

Environment:
    RAG_EMBEDDER          hash (default, offline, meaningless scores) | st (all-MiniLM-L6-v2)
    RAG_BENCH_BASE_URL    OpenAI-compatible endpoint for multi_query / hyde; else FakeLLM
    RAG_BENCH_API_KEY     its key (never printed, never written)
    RAG_BENCH_MODEL       its model name
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.pipeline import IngestPipeline
from astromesh_rag.retrievers import BM25Index
from astromesh_rag.splitters import RecursiveSplitter
from astromesh_rag.testing import FakeLLM, HashEmbedder
from astromesh_rag.types import Document

CORPUS = Path(__file__).parent / "corpus" / "corpus.jsonl"
QUERIES = Path(__file__).parent / "corpus" / "queries.jsonl"
ST_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Offline stand-ins: the same canned rewrite and passage for every question. They exercise
# the transform stage; they say nothing about how a real LLM would rewrite.
FAKE_LLM_REPLIES = {
    "Rewrite the question": (
        "problema con la cuenta de Nimbus\nconsulta de facturación\nmensaje de error"
    ),
    "Write a short passage": (
        "Según la documentación de Nimbus, la opción se encuentra en Configuración."
    ),
}


def load_docs() -> list[Document]:
    rows = map(json.loads, CORPUS.read_text(encoding="utf-8").splitlines())
    return [Document(r["text"], r["id"], r["metadata"]) for r in rows]


def load_queries() -> list[dict]:
    return [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines()]


def make_embedder(kind: str | None = None):
    kind = kind or os.environ.get("RAG_EMBEDDER", "hash")
    if kind == "st":
        from astromesh_rag.embedders import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(ST_MODEL), ST_MODEL
    if kind != "hash":
        raise SystemExit(f"unknown embedder {kind!r}; use hash or st")
    return HashEmbedder(), "hash"


def make_llm():
    """(llm, model label, endpoint host). Host only: a URL may carry credentials."""
    base_url = os.environ.get("RAG_BENCH_BASE_URL")
    if not base_url:
        return FakeLLM(FAKE_LLM_REPLIES), "fake", None
    from astromesh_rag.llms import OpenAICompatLLM

    model = os.environ.get("RAG_BENCH_MODEL", "")
    llm = OpenAICompatLLM(base_url, os.environ.get("RAG_BENCH_API_KEY", ""), model)
    return llm, model, urlparse(base_url).hostname


def warn_if_offline(embedder_label: str, llm_label: str) -> None:
    if embedder_label == "hash" or llm_label == "fake":
        print(
            f"WARNING: embedder={embedder_label} llm={llm_label}: scores are meaningless with "
            "the test embedder / fake LLM. Set RAG_EMBEDDER=st and RAG_BENCH_* for real runs."
        )


async def ingest(embedder):
    """One ingestion into a vector index and a BM25 index. Every doc fits in one chunk."""
    index, bm25 = MemoryIndex(), BM25Index()
    pipeline = IngestPipeline(RecursiveSplitter(800, 80), embedder, index, bm25)
    await pipeline.ingest(load_docs())
    return index, bm25


def print_result(result) -> None:
    print(f"\n> {result.query}")
    for rank, h in enumerate(result.hits, 1):
        details = " ".join(f"{k}={v:.3f}" for k, v in h.details.items())
        print(f"  {rank}. {h.score:7.3f}  {h.source:<12} {h.chunk.doc_id:<32} {details}")
    print("  trace:")
    for t in result.trace:
        err = f"  error={t['error']}" if "error" in t else ""
        print(f"    {t['stage']:<9} {t['name']:<22} {t['ms']:8.1f} ms  {t['in']}->{t['out']}{err}")


EXAMPLE_QUERIES = [
    "¿Qué significa ERR-4012?",
    "me cobraron dos veces este mes",
    "¿Cuántos días tengo para pedir que me devuelvan la plata?",
]


async def run_example(build) -> None:
    """build(embedder, index, bm25, llm) -> QueryPipeline; runs EXAMPLE_QUERIES through it."""
    embedder, emb_label = make_embedder()
    llm, llm_label, _ = make_llm()
    warn_if_offline(emb_label, llm_label)
    index, bm25 = await ingest(embedder)
    pipeline = build(embedder, index, bm25, llm)
    for q in EXAMPLE_QUERIES:
        print_result(await pipeline.query(q, k=5))
