# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Data model (`Document`, `Chunk`, `Hit`, `QueryResult`) and async stage contracts
  (`Splitter`, `Embedder`, `VectorIndex`, `KeywordIndex`, `Retriever`, `QueryTransform`,
  `Fusion`, `Reranker`, `LLM`). Chunk ids are `<doc_id>#<n>`; `doc_id` is the document's id
  or a SHA-256 prefix of its text.
- `IngestPipeline`, `QueryPipeline` and `Pipeline`. Every query returns a `trace` with one
  entry per stage (`stage`, `name`, `ms`, `in`, `out`, `error`); a failing retriever is
  recorded and the others' hits are used.
- `kind: RAGPipeline` YAML loading (`load_spec`, `load_spec_file`, `build`), compatible
  with the Astromesh core's files, plus new `retrieval` keys: `pattern`, `candidates`,
  `hybrid.fusion`, `hybrid.weights`, `bm25_path`, `variants`.
- An open registry (`registry.register(kind, name, factory)`) for splitters, embedders,
  indexes, rerankers and LLMs, usable by name from YAML.
- Ported from the core: `fixed`, `recursive`, `sentence` and `semantic` splitters; Ollama,
  Hugging Face and sentence-transformers embedders; FAISS, Chroma, Qdrant and pgvector
  indexes; cross-encoder and Cohere rerankers.
- New: a pure-Python `memory` index, an in-memory BM25 keyword index with JSON
  persistence, RRF and weighted fusion, and `OpenAICompatLLM` for any OpenAI-compatible
  chat endpoint.
- Retrieval patterns `naive`, `hybrid`, `multi_query` and `hyde`, from Python and YAML.
- Retrieval metrics: recall@k, precision@k, hit rate@k, MRR and nDCG@k.
- `astromesh_rag.testing`: `HashEmbedder` and `FakeLLM`, deterministic and offline.
- Optional extras per integration; a missing one raises `MissingExtraError` with the
  `pip install` to run. The core depends only on PyYAML.
- Nimbus, a simulated 200-document corpus with 60 labeled questions, an example per
  pattern and a benchmark script that ranks the patterns with and without reranking.
- CI on Python 3.10–3.13, integration jobs for FAISS, Chroma, sentence-transformers,
  Qdrant and pgvector, and a release workflow through TestPyPI to PyPI.

### Fixed

Relative to the RAG implementation in the Astromesh core this library was extracted from:

- A second ingestion overwrote the first: every document's chunks got the ids `doc_0`,
  `doc_1`… Ids are now stable per document, and re-ingesting a document replaces its
  chunks instead of duplicating them.
- Qdrant rejected the string point ids; points now use a UUIDv5 of the chunk id.
- pgvector filters were interpolated into the SQL; they are now bound parameters, and the
  table name is validated.
- Chroma collections used L2 distance instead of cosine, so scores did not match the other
  backends.
- `semantic` chunking from YAML silently fell back to sentence splitting because no
  embedder reached it; it now uses the pipeline's embedder.
- `retrieval.similarity_threshold` and `reranking.top_k` were accepted and ignored. The
  threshold now drops vector hits below that cosine (never applied to reranker scores),
  and `reranking.top_k` caps the reranker's output.
- `overlap >= chunk_size` hung the splitter; it is now a `SpecError` at build time.
