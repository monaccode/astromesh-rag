# astromesh-rag v0.1 — design

**Date:** 2026-10-14 · **Status:** approved in conversation, pending written review
**Repo:** `monaccode/astromesh-rag` (public, Apache-2.0) · **PyPI:** `astromesh-rag`

## Goal

One RAG library with two users:

1. **The community.** `pip install astromesh-rag` and build any RAG pattern from composable
   stages, in Python or YAML, without Astromesh.
2. **The Astromesh runtime.** The core delegates all of its RAG to this library and keeps
   only the wiring (agent YAML, REST API, tools, prompt injection) — the same move made with
   `astromesh-okf`.

v0.1 is the foundation the later phases build on: the stage contracts, everything the core
implements today ported, three patterns that prove composition, and the core migrated onto
it with no change to existing `kind: RAGPipeline` files.

**Success criteria**

1. Every `kind: RAGPipeline` file the core accepts today builds and queries through the
   library with the same results shape; the full core test suite passes after migration.
2. `naive`, `hybrid`, `multi_query` and `hyde` run from YAML and from Python.
3. A third party adds a stage (e.g. a new vector DB) with `registry.register(...)` and uses
   it by name from YAML, without editing the library.
4. Ingesting two documents never overwrites the first; re-ingesting a document replaces its
   chunks instead of duplicating them.
5. The core package depends only on PyYAML; each integration is an optional extra whose
   absence produces an error naming the extra to install.
6. A reproducible benchmark ranks the patterns on a simulated corpus, overall and per query
   type, with real models; every number published in the README and docs-site comes from a
   recorded run of it.

## Roadmap (each phase its own spec → plan → release)

| Phase | Scope |
|---|---|
| **1 — v0.1 (this spec)** | Contracts, port of the core, `naive`/`hybrid`/`multi_query`/`hyde`, retrieval metrics, simulated corpus + benchmark, OpenAI-compatible `LLM`, core migration, docs |
| 2 — Patterns | parent-document, contextual retrieval, corrective (CRAG), self-RAG, agentic loop, GraphRAG, RAPTOR |
| 3 — Providers | Managed RAG as `Retriever`/`VectorIndex`: Vertex AI RAG Engine, Bedrock Knowledge Bases, Azure AI Search, OpenAI file search; more vector DBs: Pinecone, Weaviate, Elasticsearch/OpenSearch, Milvus; native hybrid search where the DB has it |
| 4 — Evaluation | Generation metrics (faithfulness, answer relevance, context precision), LLM-as-judge; bridge to the core's `kind: Eval` |
| 5 — Reach | File loaders (PDF, HTML, Markdown, Office), adapters (LangChain, LlamaIndex), MCP server, CLI |

Out of scope for v0.1: everything in phases 2–5.

## 1. Package and architecture

```
astromesh_rag/
  types.py        Document, Chunk, Hit, QueryResult — dataclasses, no dependencies
  stages.py       async contracts: Splitter, Embedder, VectorIndex, QueryTransform,
                  Retriever, Fusion, Reranker, Filter, ContextBuilder, LLM
  pipeline.py     IngestPipeline, QueryPipeline, Pipeline (both, sharing stages)
  registry.py     (kind, name) → factory; built-ins registered on import
  spec.py         load/validate kind: RAGPipeline; spec → Pipeline
  errors.py       RAGError, SpecError, MissingExtraError, StageError
  patterns/       naive.py, hybrid.py, multi_query.py, hyde.py
  splitters/      fixed, recursive, sentence, semantic            (ported)
  embedders/      ollama, hf, sentence_transformers               (ported)
  indexes/        memory (new, no deps), faiss, chroma, qdrant, pgvector (ported)
  retrievers/     vector.py (over a VectorIndex), bm25.py (new, pure Python)
  rerankers/      cross_encoder, cohere                           (ported)
  fusion.py       rrf, weighted score fusion
  llms/           openai_compat.py — any OpenAI-compatible chat endpoint (gateway, Ollama, vLLM, OpenAI)
  metrics.py      recall@k, precision@k, MRR, nDCG@k, hit rate — pure functions over ranked ids + qrels
  testing.py      HashEmbedder (deterministic, no network), FakeLLM — public, for users' tests too
```

**Design rules**

- **Core depends only on PyYAML.** Extras: `ollama` (httpx), `hf` (httpx),
  `sentence-transformers`, `faiss`, `chroma`, `qdrant`, `pgvector` (asyncpg), `cohere`
  (httpx), `openai` (httpx), `all`. Importing a component whose extra is missing raises
  `MissingExtraError("pip install 'astromesh-rag[qdrant]'")` at pipeline build time.
- **Open registry.** `registry.register(kind, name, factory)` where `kind` is one of
  `splitter, embedder, index, reranker, llm` (retrievers, fusion and transforms are composed
  by the patterns in Python, not looked up by name).
  A later registration under the same name replaces the earlier one — this is how the core
  puts its Rust chunker behind `splitter/recursive`.
- **Stable ids.** Chunk id = `f"{doc_id}#{n}"`. `doc_id` comes from the `Document`, else a
  SHA-256 prefix of its text. Ingesting a `doc_id` first deletes that document's previous
  chunks (`VectorIndex.delete_document(doc_id)`), so re-ingestion replaces.
- **Async everywhere**, matching the core. Known core bugs fixed in the port: Qdrant point
  ids become UUIDv5 of the chunk id (Qdrant rejects `doc_0`); pgvector filters are bound
  parameters, not interpolated SQL; Chroma collections use cosine space; `semantic`
  chunking actually uses an embedder (from YAML it silently fell back to `sentence`). Synchronous libraries are called through
  `asyncio.to_thread`.
- **Python ≥ 3.10**, like `astromesh-okf`.

## 2. Data model and contracts

```python
@dataclass
class Document:  id: str | None; text: str; metadata: dict = {}
@dataclass
class Chunk:     id: str; doc_id: str; text: str; metadata: dict = {}
@dataclass
class Hit:       chunk: Chunk; score: float; source: str   # which retriever found it
@dataclass
class QueryResult:
    query: str; hits: list[Hit]; context: str = ""; answer: str | None = None
    trace: list[dict] = []          # one entry per stage: {stage, name, ms, in, out, error?}
    def to_list(self) -> list[dict]  # [{content, metadata, score}] — the core's current shape
```

| Stage | Signature |
|---|---|
| `Splitter` | `async split(doc: Document) -> list[Chunk]` (async so `semantic` can use the pipeline's embedder) |
| `Embedder` | `async embed(texts: list[str]) -> list[list[float]]`, `dimension: int \| None` |
| `VectorIndex` | `async upsert(chunks, vectors)`, `async search(vector, k, filters) -> list[Hit]`, `async delete_document(doc_id)` |
| `Retriever` | `async retrieve(query: str, k: int) -> list[Hit]` |
| `QueryTransform` | `async transform(query: str) -> list[str]` (1..N queries or hypothetical texts) |
| `Fusion` | `fuse(lists: list[list[Hit]], k: int) -> list[Hit]` |
| `Reranker` | `async rerank(query, hits, k) -> list[Hit]` |
| `Filter` | `filter(hits) -> list[Hit]` (e.g. `min_score`) |
| `ContextBuilder` | `build(hits) -> str` (default: texts joined by blank lines, as `AgentRAG` does today) |
| `LLM` | `async complete(prompt: str) -> str` — the only thing patterns need from a model |

`IngestPipeline(splitter, embedder, index, bm25=None).ingest(docs) -> int`;
`QueryPipeline(...).query(q, k) -> QueryResult`; `Pipeline` holds both over the same index.

## 3. YAML and patterns

Existing files keep working unchanged. New optional keys under `spec.retrieval`:

```yaml
retrieval:
  top_k: 5
  similarity_threshold: 0.7     # today inert → drops vector hits whose cosine is below it
  pattern: naive                # naive (default) | hybrid | multi_query | hyde
  hybrid:      { weights: [0.5, 0.5], fusion: rrf }   # rrf (default) | weighted
  multi_query: { variants: 3 }
  candidates:  20               # how many each retriever fetches before rerank (default top_k * 2, as today)
```

| Pattern | Query flow | LLM |
|---|---|---|
| `naive` | vector (threshold) → rerank? → top_k → context | no |
| `hybrid` | vector (threshold) ∥ bm25 → fusion → rerank? → top_k → context | no |
| `multi_query` | LLM → N variants (+ original) → vector per variant → RRF → rerank? → filter | yes |
| `hyde` | LLM writes a hypothetical answer → embed it and the original question → vector ×2 → RRF → rerank? → top_k | yes |

- **BM25** is built during the same ingestion, in memory, persisted to `bm25_path` if set;
  `hybrid` without a BM25 index is a `SpecError` at build time.
- **LLM-backed patterns** need `Pipeline.from_spec(spec, llm=...)`; missing `llm` is a
  `SpecError` at build time, not at the first query. Prompts are module constants,
  overridable by argument.
- **No silently inert fields.** Today two keys are accepted and ignored:
  `retrieval.similarity_threshold` (now drops vector hits below that cosine; it is not applied after reranking because cross-encoder scores are not on a 0–1 scale) and `reranking.top_k` (now
  caps the reranker's output; defaults to `retrieval.top_k`). Unknown keys are kept
  tolerated, as today: provider sections carry provider-specific keys, and specs created
  from Forge or Cortex must keep loading.

## 4. Core migration (astromesh core, next minor)

- `pyproject.toml`: `astromesh-rag>=0.1,<0.2` from PyPI (no path source — the OS build uses
  pip). Heavy RAG extras stay optional in the core; `api.main` must still import with no
  extras installed.
- `astromesh/rag/` becomes a thin facade with today's names and signatures:
  `build_pipeline(spec)`, `RAGPipelineLoader`, `RAGPipelineSpec`, `spec_from_raw`,
  `RAGPipeline`, `RAGResult`, `result_to_list`, `AgentRAG`, `format_knowledge`.
  Callers (`runtime/engine.py`, `api/routes/rag.py`, `api/routes/rag_resources.py`,
  `tools/builtin/rag.py`) are not edited beyond what the facade requires.
- `spec_from_raw` keeps raising `ValueError` (callers map it to 4xx); the facade converts
  `SpecError` → `ValueError`.
- The Rust chunker is registered over `splitter/recursive` when `astromesh._native` imports
  and `ASTROMESH_FORCE_PYTHON` is unset.
- The core's `ModelRouter` is wrapped as the library's `LLM` for `multi_query`/`hyde`.
- `rag_ingest` gains an optional `doc_id` argument.
- Old per-component modules under `astromesh/rag/{chunking,embeddings,stores,reranking}`
  are deleted once nothing imports them; their tests move to the library or are rewritten
  against the facade.
- **Behavior changes (CHANGELOG):** stable chunk ids — one ingestion no longer overwrites
  another (fix); `similarity_threshold` now filters; `reranking.top_k` now caps the
  reranker's output.
- Release order: library on PyPI → core → OS (`runtime.pin`) → Nexus pool. Cortex and Forge
  only need updates if the YAML schema they hand-keep gains the new `retrieval` keys.

## 5. Simulated corpus, examples and benchmark

The point is to show *when each pattern wins and why*, with numbers, not only that it runs.

**Corpus** (`examples/corpus/`): the documentation of a fictional product — support
articles, policies, catalog — generated by `generate.py` with a fixed seed, and committed as
`corpus.jsonl` so every run uses the same text. About 150–250 documents, built with
deliberate traps, each tagged in its metadata:

| Trap | Example | Pattern expected to win |
|---|---|---|
| Exact identifiers | `ERR-4012`, `SKU-77A`, plan names | `hybrid` (BM25 catches the token) |
| Vocabulary mismatch | "me cobraron dos veces" vs "cargo duplicado" | `multi_query`, `hyde` |
| Near-duplicate versions | refund policy 2024 vs 2026 | any pattern + reranker |
| Noise | similar but irrelevant articles | reranker, `similarity_threshold` |

**Queries** (`examples/corpus/queries.jsonl`): ~60 hand-curated questions in Spanish and
English, each with its relevant document ids (qrels) and its type (`exact`, `paraphrase`,
`distractor`, `noise`). Expectations in the table above are hypotheses the benchmark tests;
the README reports what the run measured, including where a hypothesis failed.

**Examples** (`examples/naive.py`, `hybrid.py`, `multi_query.py`, `hyde.py`, plus
`pipeline.rag.yaml`): each loads the corpus, runs a few queries and prints hits with score
and `source`, plus the `trace` of every stage. They run offline with `HashEmbedder` and
`FakeLLM` by default (shape only, scores meaningless — said in the output), and with real
models when the environment provides them.

**Benchmark** (`examples/benchmark.py`): runs `naive`, `hybrid`, `multi_query`, `hyde`,
each with and without the cross-encoder reranker, over the same index. Reports recall@5,
MRR@10, nDCG@10, p50/p95 latency and LLM calls per query — overall and per query type —
as a Markdown table and `results/<date>.json` (config, model names, endpoint host, corpus
hash; never keys).

- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (local, reproducible).
- LLM for `multi_query`/`hyde`: Kimi through the Academia gateway via `llms/openai_compat`,
  configured by `RAG_BENCH_BASE_URL`, `RAG_BENCH_API_KEY`, `RAG_BENCH_MODEL` (env only,
  never committed). The model is recorded in the results.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- The published table and a chart (rendered PNG) go in the README and the docs-site
  Patterns page, with the run's date, models and corpus hash beside them.

## 6. Errors, tests, documentation

**Errors.** Build-time: `SpecError` (bad YAML, unknown component name, missing LLM/BM25),
`MissingExtraError`. Query-time: a failing stage raises `StageError(stage, name, cause)`;
in `hybrid`, a failing retriever is recorded in `trace` and the other's hits are used —
only both failing raises. `AgentRAG` keeps its contract: never fails a run.

**Tests.**
- Unit tests per stage; patterns tested with `HashEmbedder` and `FakeLLM` (no network,
  no models).
- Compatibility test: the core's `config/rag/product-knowledge.rag.yaml` builds (with the
  store swapped to `memory`) and returns the core's result shape.
- Ingestion tests for criteria 4 (two docs both present; re-ingest replaces).
- Metrics tested against hand-computed values; the benchmark has a smoke test with
  `HashEmbedder`/`FakeLLM` in CI (shape only); the real run is manual and recorded.
- Integration jobs in CI per extra with Docker services: Qdrant, Postgres+pgvector, Chroma;
  FAISS and sentence-transformers in their own jobs.
- Core: the full suite stays green after migration.

**Documentation.**
- Repo: `README.md`, `README.es.md`, `docs/patterns.md`, `docs/spec.md` (YAML reference),
  `docs/extending.md`, `CHANGELOG.md`, `CONTRIBUTING.md`.
- **The README is a first-class deliverable** (user requirement): visually polished and
  well explained, built with the `frontend-design` skill and checked by rendering every
  diagram (no syntax errors, no clipped labels) before commit. It must contain, in order:
  logo; one-line pitch; badges (layer first, per the ecosystem badge grammar); a short
  explanation of the RAG lifecycle in plain language; a **Mermaid mindmap** of the library
  (lifecycle stages, patterns, integrations, extension points) themed with the Astromesh
  palette as in the `astromesh-sdd` README; a **flowchart per pattern** (naive, hybrid,
  multi_query, hyde) next to a table "when to use it / what it costs / needs an LLM?";
  a **sequence diagram** of one hybrid query showing each stage and the `trace`; the
  **benchmark ranking** (overall and per query type, with the run's models and date); a
  quickstart in Python and in YAML; the extras table; "make it yours" (registry);
  "how Astromesh uses it"; roadmap; contributing; license. The Spanish README mirrors it
  with translated diagram labels.
- docs-site: a "RAG" section — Introduction (the lifecycle, when to use which pattern),
  Patterns, Pipeline YAML reference, Extending, How Astromesh uses it; `ecosystem.ts`
  entry (sector chosen at docs time; the library sits under the runtime's knowledge layer,
  next to OKF). Published only after the release ships.
- Brand: logo in the Astromesh orbit family, like OKF and SDD.
