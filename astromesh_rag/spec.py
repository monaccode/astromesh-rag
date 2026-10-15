"""`kind: RAGPipeline` YAML -> Pipeline, compatible with the Astromesh core's files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import SpecError
from .fusion import RRF, WeightedFusion
from .patterns import PATTERNS
from .pipeline import IngestPipeline, Pipeline
from .registry import BuildContext
from .registry import registry as default_registry
from .retrievers import BM25Index

_SECTIONS = ("chunking", "embeddings", "vector_store", "reranking", "retrieval")
_FALLBACK_DIMENSION = 384  # the core's default when the spec gives none


@dataclass
class RAGSpec:
    name: str
    chunking: dict = field(default_factory=dict)
    embeddings: dict = field(default_factory=dict)
    vector_store: dict = field(default_factory=dict)
    reranking: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)


def load_spec(raw: dict) -> RAGSpec:
    if not isinstance(raw, dict):
        raise SpecError("RAGPipeline body must be a mapping")
    if raw.get("kind") != "RAGPipeline":
        raise SpecError(f"Expected kind: RAGPipeline, got: {raw.get('kind')}")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SpecError("RAGPipeline metadata must be a mapping")
    if not metadata.get("name"):
        raise SpecError("RAGPipeline missing metadata.name")
    spec = raw.get("spec", {})
    if not isinstance(spec, dict):
        raise SpecError("RAGPipeline spec must be a mapping")
    for s in _SECTIONS:
        if s in spec and not isinstance(spec[s], dict):
            raise SpecError(f"RAGPipeline spec.{s} must be a mapping")
    return RAGSpec(metadata["name"], **{s: spec.get(s, {}) for s in _SECTIONS})


def load_spec_file(path) -> RAGSpec:
    return load_spec(yaml.safe_load(Path(path).read_text()))


def build(spec: RAGSpec, *, llm=None, registry=default_registry) -> Pipeline:
    from . import builtins  # noqa: F401  (idempotent; registers the defaults)

    emb, vs, rr, rt = spec.embeddings, spec.vector_store, spec.reranking, spec.retrieval
    ctx = BuildContext(dimension=emb.get("dimension"), llm=llm)
    embedder = registry.build("embedder", emb.get("provider", "ollama"), emb, ctx)
    ctx.embedder = embedder
    ctx.dimension = emb.get("dimension") or embedder.dimension or _FALLBACK_DIMENSION

    pattern = rt.get("pattern", "naive")
    if pattern not in PATTERNS:
        raise SpecError(f"unknown retrieval.pattern '{pattern}'; known: {', '.join(PATTERNS)}")
    if pattern in ("multi_query", "hyde") and llm is None:
        raise SpecError(f"pattern '{pattern}' needs an llm: build(spec, llm=...)")

    splitter = registry.build(
        "splitter", spec.chunking.get("strategy", "recursive"), spec.chunking, ctx
    )
    index = registry.build("index", vs.get("backend", "faiss"), vs, ctx)
    bm25 = None
    if pattern == "hybrid":
        path = rt.get("bm25_path")
        bm25 = BM25Index.load(path) if path and Path(path).exists() else BM25Index(path=path)
    reranker = None
    if rr.get("enabled"):
        reranker = registry.build("reranker", rr.get("provider", "cross_encoder"), rr, ctx)

    top_k = rt.get("top_k", 5)
    h = rt.get("hybrid", {})
    fusion_name = rt.get("fusion", h.get("fusion", "rrf"))
    if fusion_name not in ("rrf", "weighted"):
        raise SpecError(f"unknown retrieval.fusion '{fusion_name}'; known: rrf, weighted")
    weights = h.get("weights", [0.5, 0.5])
    if fusion_name == "weighted" and not (isinstance(weights, list) and len(weights) == 2):
        raise SpecError("weighted fusion needs exactly two weights: [vector, bm25]")
    common = {
        "reranker": reranker,
        "rerank_k": rr.get("top_k") or top_k,
        "min_score": rt.get("similarity_threshold"),
        "candidates": rt.get("candidates") or top_k * 2,
    }
    if pattern == "naive":
        query = PATTERNS["naive"](embedder, index, **common)
    elif pattern == "hybrid":
        fusion = WeightedFusion(weights) if fusion_name == "weighted" else RRF()
        query = PATTERNS["hybrid"](embedder, index, bm25, fusion=fusion, **common)
    else:
        variants = rt.get("multi_query", {}).get("variants", rt.get("variants"))
        extra = {"variants": variants} if pattern == "multi_query" and variants else {}
        query = PATTERNS[pattern](embedder, index, llm, **extra, **common)
    return Pipeline(IngestPipeline(splitter, embedder, index, bm25), query, default_k=top_k)
