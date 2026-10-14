"""Built-in registrations. Components are imported lazily; extras load in their constructors."""

from __future__ import annotations

from .errors import SpecError
from .registry import registry


def _window(cfg, default_overlap=50):
    return cfg.get("chunk_size", 500), cfg.get("overlap", default_overlap)


def _guard(keys, make):
    try:
        return make()
    except ValueError as e:
        raise SpecError(f"invalid chunking config ({keys}): {e}") from e


def _fixed(cfg, ctx):
    from .splitters import FixedSplitter

    size, overlap = _window(cfg)
    return _guard("chunk_size, overlap", lambda: FixedSplitter(size, overlap))


def _recursive(cfg, ctx):
    from .splitters import RecursiveSplitter

    size, overlap = _window(cfg)
    return _guard(
        "chunk_size, overlap",
        lambda: RecursiveSplitter(size, overlap, cfg.get("separators")),
    )


def _sentence(cfg, ctx):
    from .splitters import SentenceSplitter

    size, overlap = _window(cfg, 0)
    return SentenceSplitter(size, overlap)


def _semantic(cfg, ctx):
    from .splitters import SemanticSplitter

    return SemanticSplitter(
        cfg.get("chunk_size", 500), cfg.get("similarity_threshold", 0.5), ctx.embedder
    )


def _ollama(cfg, ctx):
    from .embedders import OllamaEmbedder

    return OllamaEmbedder(
        cfg.get("endpoint", "http://ollama:11434"),
        cfg.get("model", "nomic-embed-text"),
        cfg.get("dimension"),
    )


def _hf(cfg, ctx):
    from .embedders import HFEmbedder

    return HFEmbedder(
        cfg.get("model", "BAAI/bge-small-en-v1.5"),
        cfg.get("endpoint"),
        cfg.get("api_key"),
        cfg.get("dimension"),
    )


def _st(cfg, ctx):
    from .embedders import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(cfg.get("model", "all-MiniLM-L6-v2"))


def _hash(cfg, ctx):
    from .testing import HashEmbedder

    return HashEmbedder()


def _memory(cfg, ctx):
    from .indexes import MemoryIndex

    return MemoryIndex()


def _faiss(cfg, ctx):
    from .indexes.faiss import FaissIndex

    return FaissIndex(cfg.get("dimensions", ctx.dimension))


def _chroma(cfg, ctx):
    from .indexes.chroma import ChromaIndex

    return ChromaIndex(
        cfg.get("collection", "astromesh"),
        cfg.get("host"),
        cfg.get("port", 8000),
        cfg.get("persist_directory"),
    )


def _qdrant(cfg, ctx):
    from .indexes.qdrant import QdrantIndex

    return QdrantIndex(
        cfg.get("collection", "astromesh"),
        cfg.get("url", "http://localhost:6333"),
        cfg.get("api_key"),
        ctx.dimension,
    )


def pg_dsn(cfg: dict) -> str:
    if cfg.get("dsn"):
        return cfg["dsn"]
    c = cfg.get("connection", {})
    return (
        f"postgresql://{c.get('user', 'astromesh')}:{c.get('password', 'astromesh')}"
        f"@{c.get('host', 'localhost')}:{c.get('port', 5432)}/{c.get('database', 'astromesh')}"
    )


def _pgvector(cfg, ctx):
    from .indexes.pgvector import PgVectorIndex

    return PgVectorIndex(pg_dsn(cfg), cfg.get("collection", "embeddings"), ctx.dimension)


def _cross_encoder(cfg, ctx):
    from .rerankers import CrossEncoderReranker

    return CrossEncoderReranker(cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))


def _cohere(cfg, ctx):
    from .rerankers import CohereReranker

    return CohereReranker(cfg.get("api_key", ""), cfg.get("model", "rerank-english-v3.0"))


def _openai_compat(cfg, ctx):
    from .llms import OpenAICompatLLM

    return OpenAICompatLLM(
        cfg["base_url"],
        cfg.get("api_key", ""),
        cfg["model"],
        cfg.get("temperature", 0.0),
        cfg.get("timeout", 60.0),
    )


for _k, _n, _f in [
    ("splitter", "fixed", _fixed),
    ("splitter", "recursive", _recursive),
    ("splitter", "sentence", _sentence),
    ("splitter", "semantic", _semantic),
    ("embedder", "ollama", _ollama),
    ("embedder", "hf", _hf),
    ("embedder", "sentence_transformer", _st),
    ("embedder", "sentence_transformers", _st),
    ("embedder", "hash", _hash),
    ("index", "memory", _memory),
    ("index", "faiss", _faiss),
    ("index", "chroma", _chroma),
    ("index", "qdrant", _qdrant),
    ("index", "pgvector", _pgvector),
    ("reranker", "cross_encoder", _cross_encoder),
    ("reranker", "cohere", _cohere),
    ("llm", "openai_compat", _openai_compat),
]:
    registry.register(_k, _n, _f)
