# astromesh-rag v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `astromesh-rag` 0.1.0 — composable RAG stages, the core's RAG ported, four patterns, retrieval metrics, a simulated corpus with a recorded benchmark, a designed README and docs — and move the Astromesh core onto it.

**Architecture:** Small async stage contracts (`Splitter`, `Embedder`, `VectorIndex`, `Retriever`, `QueryTransform`, `Fusion`, `Reranker`, `LLM`) composed by `IngestPipeline`/`QueryPipeline`. A registry maps YAML names to factories; patterns are functions that assemble a `QueryPipeline`. The core keeps `astromesh/rag/` as a facade over the library.

**Tech Stack:** Python ≥3.10, PyYAML, uv + hatchling, pytest (+pytest-asyncio auto mode), ruff, httpx (extras), GitHub Actions with Docker services (Qdrant, Postgres+pgvector).

**Spec:** `docs/superpowers/specs/2026-10-14-astromesh-rag-design.md`

## Global Constraints

- Package `astromesh-rag`, import `astromesh_rag`, version `0.1.0`, Apache-2.0, `requires-python = ">=3.10"`.
- Runtime dependency: `pyyaml>=6.0` only. Extras: `ollama = ["httpx>=0.27"]`, `hf = ["httpx>=0.27"]`, `cohere = ["httpx>=0.27"]`, `openai = ["httpx>=0.27"]`, `sentence-transformers = ["sentence-transformers>=3.0"]`, `faiss = ["faiss-cpu>=1.8", "numpy>=1.26"]`, `chroma = ["chromadb>=0.5"]`, `qdrant = ["qdrant-client>=1.11"]`, `pgvector = ["asyncpg>=0.30"]`, `all` = union.
- Missing extra → `MissingExtraError` whose message contains `pip install 'astromesh-rag[<extra>]'`, raised when the pipeline is built, not on `import astromesh_rag`.
- Chunk id = `f"{doc_id}#{n}"`; `doc_id` = `Document.id` or `"sha256:" + sha256(text)[:16]`.
- `QueryResult.to_list()` items: `{"doc_id": <chunk id>, "content": str, "metadata": dict, "score": float, "source": str}` — `doc_id`/`content`/`metadata`/`score` are what the core consumes today.
- Everything async except `Fusion.fuse` and context building.
- `retrieval.similarity_threshold` filters vector hits by cosine; never applied to rerank scores.
- Default `candidates` = `top_k * 2`; `reranking.top_k` caps reranker output, default `retrieval.top_k`.
- ruff: `target-version = "py310"`, `line-length = 100`, same rule set as `astromesh-okf/pyproject.toml`; CI runs `ruff check` and `ruff format --check`.
- Conventional commits, each ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; core commits need a CHANGELOG entry (core CLAUDE.md rule).
- Core release constraints: dependency from PyPI (no uv path source); `astromesh.api.main` imports with no extras.

## Review Focus

1. **Re-ingesting a document with fewer chunks than before** must leave no stale chunks (old `#3` gone when new version has 2) → Task 6 `test_reingest_with_fewer_chunks_removes_stale`.
2. **Empty corpus / empty query / query with no hits** returns an empty `QueryResult` with empty context, never an exception → Task 6 `test_query_empty_index`, `test_query_blank_string`.
3. **Hybrid when one retriever raises** returns the other's hits and records the error in `trace` → Task 6 `test_hybrid_survives_one_failing_retriever`.
4. **A YAML from the core that mixes unknown provider keys** (e.g. `vector_store.connection`, `reranking.model`) loads without error → Task 7 `test_core_yaml_loads_unchanged`.
5. **Metadata filters with quotes or SQL** (`{"title": "x'; DROP TABLE"}`) are passed as parameters to pgvector → Task 3 `test_pgvector_filter_is_parameterized` (integration) and unit check of the generated SQL.

---

## File map

| Path | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `LICENSE`, `.gitignore`, `uv.lock` | package | 1 |
| `astromesh_rag/{__init__,types,errors,stages,registry,_extras}.py` | contracts, data model, registry | 1 |
| `astromesh_rag/testing.py` | `HashEmbedder`, `FakeLLM` | 2 |
| `astromesh_rag/splitters/{fixed,recursive,sentence,semantic}.py` | splitting | 2 |
| `astromesh_rag/indexes/{memory,faiss,chroma,qdrant,pgvector}.py` | vector indexes | 3 |
| `astromesh_rag/embedders/{ollama,hf,sentence_transformers}.py`, `rerankers/{cross_encoder,cohere}.py`, `llms/openai_compat.py` | model-backed stages | 4 |
| `astromesh_rag/retrievers/{vector,bm25}.py`, `fusion.py`, `context.py` | retrieval | 5 |
| `astromesh_rag/pipeline.py` | ingest/query orchestration, trace | 6 |
| `astromesh_rag/spec.py`, `patterns/{naive,hybrid,multi_query,hyde}.py`, `builtins.py` | YAML + patterns + built-in registrations | 7 |
| `astromesh_rag/metrics.py` | retrieval metrics | 8 |
| `examples/corpus/{generate.py,corpus.jsonl,queries.jsonl}`, `examples/*.py`, `examples/benchmark.py` | corpus, examples, benchmark | 9 |
| `.github/workflows/{ci,release}.yml` | CI | 10 |
| `assets/*`, `README.md`, `README.es.md`, `docs/*.md`, `CHANGELOG.md`, `CONTRIBUTING.md` | brand + docs | 11 |
| (GitHub, PyPI) | publish 0.1.0 | 12 |
| `results/<date>.json`, README results | real benchmark run | 13 |
| `../astromesh/astromesh/rag/*`, core tests, `pyproject.toml`, `CHANGELOG.md` | core migration | 14 |
| `../astromesh/docs-site/...` | docs-site RAG section | 15 |

---

### Task 1: Package skeleton, data model, contracts, registry

**Files:** Create `pyproject.toml` (copy `../astromesh-okf/pyproject.toml`, change name/description/keywords/urls/extras per Global Constraints, `packages = ["astromesh_rag"]`, dev group `["pytest>=8", "pytest-asyncio>=0.24", "ruff>=0.16", "respx>=0.21"]`, add `[tool.pytest.ini_options] asyncio_mode = "auto"`, no `[project.scripts]`), `LICENSE` (copy from okf), `astromesh_rag/__init__.py`, `types.py`, `errors.py`, `stages.py`, `registry.py`, `_extras.py`, `tests/test_types.py`, `tests/test_registry.py`.

**Interfaces — Produces:**
```python
# types.py
@dataclass
class Document:
    text: str
    id: str | None = None
    metadata: dict = field(default_factory=dict)
    def resolved_id(self) -> str: ...          # id or "sha256:"+hex[:16]
@dataclass
class Chunk:
    id: str; doc_id: str; text: str; metadata: dict = field(default_factory=dict)
@dataclass
class Hit:
    chunk: Chunk; score: float; source: str
    details: dict = field(default_factory=dict)   # {"vector": .., "bm25": .., "rrf": .., "rerank": ..}
@dataclass
class QueryResult:
    query: str; hits: list[Hit] = field(default_factory=list); context: str = ""
    answer: str | None = None; trace: list[dict] = field(default_factory=list)
    def to_list(self) -> list[dict]: ...
def chunk_id(doc_id: str, n: int) -> str: ...      # f"{doc_id}#{n}"
# errors.py
class RAGError(Exception)
class SpecError(RAGError)
class MissingExtraError(RAGError)                  # __init__(self, extra: str, module: str)
class StageError(RAGError)                         # __init__(self, stage: str, name: str, cause: Exception)
# _extras.py
def require(module: str, extra: str): ...          # importlib.import_module or MissingExtraError
# stages.py  (typing.Protocol, runtime_checkable)
Splitter.split(doc) -> list[Chunk]                          (async)
Embedder.embed(texts: list[str]) -> list[list[float]]       (async); dimension: int | None
VectorIndex.upsert(chunks, vectors); search(vector, k, filters=None) -> list[Hit]; delete_document(doc_id)  (async)
KeywordIndex.add(chunks); delete_document(doc_id)           (async)
Retriever.retrieve(query: str, k: int) -> list[Hit]         (async); name: str
QueryTransform.transform(query: str) -> list[str]           (async)
Fusion.fuse(lists: list[list[Hit]], k: int) -> list[Hit]
Reranker.rerank(query, hits, k) -> list[Hit]                (async)
LLM.complete(prompt: str) -> str                            (async)
# registry.py
KINDS = ("splitter","embedder","index","reranker","llm")
@dataclass
class BuildContext: dimension: int | None = None; embedder: Embedder | None = None; llm: LLM | None = None
class Registry:
    def register(self, kind: str, name: str, factory: Callable[[dict, BuildContext], object]) -> None
    def build(self, kind: str, name: str, cfg: dict, ctx: BuildContext) -> object  # SpecError if unknown
    def names(self, kind: str) -> list[str]
registry = Registry()   # module-level default
```

- [ ] **Step 1: Failing tests**

```python
# tests/test_types.py
from astromesh_rag.types import Chunk, Document, Hit, QueryResult, chunk_id

def test_document_id_defaults_to_content_hash():
    a, b = Document("hello"), Document("hello")
    assert a.resolved_id() == b.resolved_id()
    assert a.resolved_id().startswith("sha256:") and len(a.resolved_id()) == 7 + 16

def test_explicit_document_id_wins():
    assert Document("x", id="faq-1").resolved_id() == "faq-1"

def test_chunk_id_format():
    assert chunk_id("faq-1", 3) == "faq-1#3"

def test_to_list_matches_core_shape():
    c = Chunk(id="d#0", doc_id="d", text="t", metadata={"a": 1})
    r = QueryResult(query="q", hits=[Hit(c, 0.5, "vector")])
    assert r.to_list() == [
        {"doc_id": "d#0", "content": "t", "metadata": {"a": 1}, "score": 0.5, "source": "vector"}
    ]

# tests/test_registry.py
import pytest
from astromesh_rag.errors import MissingExtraError, SpecError
from astromesh_rag.registry import BuildContext, Registry
from astromesh_rag._extras import require

def test_register_and_build():
    r = Registry()
    r.register("splitter", "x", lambda cfg, ctx: ("built", cfg["n"]))
    assert r.build("splitter", "x", {"n": 2}, BuildContext()) == ("built", 2)

def test_later_registration_replaces():
    r = Registry()
    r.register("splitter", "x", lambda cfg, ctx: 1)
    r.register("splitter", "x", lambda cfg, ctx: 2)
    assert r.build("splitter", "x", {}, BuildContext()) == 2

def test_unknown_name_lists_known():
    r = Registry()
    r.register("index", "memory", lambda cfg, ctx: None)
    with pytest.raises(SpecError, match="unknown index 'nope'.*memory"):
        r.build("index", "nope", {}, BuildContext())

def test_unknown_kind_rejected():
    with pytest.raises(SpecError):
        Registry().register("banana", "x", lambda cfg, ctx: None)

def test_missing_extra_message():
    with pytest.raises(MissingExtraError, match=r"pip install 'astromesh-rag\[qdrant\]'"):
        require("definitely_not_a_module_xyz", "qdrant")
```

- [ ] **Step 2: Run** `uv sync && uv run pytest -q tests/test_types.py tests/test_registry.py` → collection errors (modules missing).

- [ ] **Step 3: Implement.** `Document.resolved_id`: `self.id or "sha256:" + hashlib.sha256(self.text.encode()).hexdigest()[:16]`. `to_list` builds the dicts exactly as the test. `Registry.build` raises `SpecError(f"unknown {kind} '{name}'; known: {', '.join(sorted(...))}")`. `require` wraps `importlib.import_module` and on `ImportError` raises `MissingExtraError(extra, module)` with message `f"{module} is not installed: pip install 'astromesh-rag[{extra}]'"`. `stages.py` uses `typing.Protocol` + `@runtime_checkable`. `__init__.py` exports the public names and `__version__ = "0.1.0"`.

- [ ] **Step 4: Run** tests → pass; `uv run ruff check . && uv run ruff format --check .` clean.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: data model, stage contracts and registry"`

---

### Task 2: Splitters and test doubles

**Files:** Create `astromesh_rag/testing.py`, `astromesh_rag/splitters/{__init__,fixed,recursive,sentence,semantic}.py`, `tests/test_splitters.py`, `tests/test_testing.py`.

**Port from** `../astromesh/astromesh/rag/chunking/{fixed,recursive,sentence,semantic}.py`: keep the splitting algorithms (`_split_recursive`, `_split_sentences`, grouping) byte-for-byte; change only the interface: `async def split(self, doc: Document) -> list[Chunk]`, ids via `chunk_id(doc.resolved_id(), i)`, metadata `{**doc.metadata, "chunk_index": i, "strategy": "<name>"}`. Remove `astromesh._native` imports (the core re-registers its native version in Task 14). `SemanticSplitter(chunk_size, similarity_threshold, embedder: Embedder | None)` embeds all sentences with **one** `await embedder.embed(sentences)` call; with `embedder=None` it falls back to `SentenceSplitter` and sets `metadata["strategy"] = "sentence"` (visible, not silent).

**Interfaces — Produces:**
```python
class HashEmbedder:   # testing.py — deterministic bag-of-words hashing, dimension=64
    dimension = 64
    async def embed(self, texts: list[str]) -> list[list[float]]   # L2-normalized
class FakeLLM:
    def __init__(self, replies: dict[str, str] | None = None, default: str = ""): ...
    calls: list[str]
    async def complete(self, prompt: str) -> str  # first key contained in prompt wins, else default
FixedSplitter(chunk_size=500, overlap=50); RecursiveSplitter(chunk_size=500, overlap=50, separators=None)
SentenceSplitter(chunk_size=500, overlap=0); SemanticSplitter(chunk_size=500, similarity_threshold=0.5, embedder=None)
```
`HashEmbedder`: tokens = `re.findall(r"\w+", text.lower())`; vector[i] += 1 for `i = int(sha1(tok).hexdigest(), 16) % 64`; normalize (zero vector stays zero). Texts sharing words get high cosine — enough for shape tests.

- [ ] **Step 1: Failing tests**
```python
# tests/test_splitters.py
from astromesh_rag.splitters import FixedSplitter, RecursiveSplitter, SemanticSplitter, SentenceSplitter
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Document

async def test_recursive_ids_and_metadata():
    doc = Document("a" * 1200, id="d", metadata={"src": "x"})
    chunks = await RecursiveSplitter(chunk_size=500, overlap=0).split(doc)
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "d" and c.metadata["src"] == "x" for c in chunks)
    assert chunks[0].metadata["strategy"] == "recursive"

async def test_empty_document_gives_no_chunks():
    for s in (FixedSplitter(), RecursiveSplitter(), SentenceSplitter(), SemanticSplitter()):
        assert await s.split(Document("", id="d")) == []

async def test_semantic_without_embedder_says_so():
    chunks = await SemanticSplitter().split(Document("One. Two. Three.", id="d"))
    assert chunks and all(c.metadata["strategy"] == "sentence" for c in chunks)

async def test_semantic_groups_similar_sentences():
    text = "Cats purr softly. Cats purr loudly. Rockets need fuel."
    chunks = await SemanticSplitter(similarity_threshold=0.3, embedder=HashEmbedder()).split(
        Document(text, id="d"))
    assert len(chunks) == 2 and "Rockets" in chunks[1].text
```
```python
# tests/test_testing.py
from astromesh_rag.testing import FakeLLM, HashEmbedder

async def test_hash_embedder_is_deterministic_and_normalized():
    a, b = await HashEmbedder().embed(["red apple", "red apple"])
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-9

async def test_fake_llm_matches_by_substring():
    llm = FakeLLM({"rewrite": "v1\nv2"}, default="none")
    assert await llm.complete("please rewrite this") == "v1\nv2"
    assert await llm.complete("other") == "none" and len(llm.calls) == 2
```
- [ ] **Step 2: Run** → ImportError. **Step 3: Implement** as described. **Step 4: Run** → pass, ruff clean.
- [ ] **Step 5: Commit** — `feat: splitters (fixed, recursive, sentence, semantic) and test doubles`

---

### Task 3: Vector indexes

**Files:** Create `astromesh_rag/indexes/{__init__,memory,faiss,chroma,qdrant,pgvector}.py`, `tests/test_index_memory.py`, `tests/integrations/{test_faiss,test_chroma,test_qdrant,test_pgvector}.py`, `tests/conftest.py` (shared `index_contract` helper).

**Contract every index meets** (tested once, run against each):
`upsert(chunks, vectors)` stores text, `doc_id`, metadata; `search(vector, k, filters)` returns `Hit(source="vector", score=cosine, details={"vector": cosine})` sorted desc; `filters` = exact-match dict on metadata; `delete_document(doc_id)` removes every chunk of that document.

- `MemoryIndex()` — pure Python, cosine, no deps. Also used by tests and the YAML compat test.
- `FaissIndex(dimension)` — port `faiss_store.py`; keep normalize + `IndexFlatIP`; add `delete_document` (tombstone all chunk positions of `doc_id`; keep `_doc_chunks: dict[str, list[int]]`); `require("faiss", "faiss")`, `require("numpy","faiss")`.
- `ChromaIndex(collection, host=None, port=8000, persist_directory=None)` — port `chroma.py`; create collection with `metadata={"hnsw:space": "cosine"}`; store `doc_id` in metadata; `delete_document` = `collection.delete(where={"doc_id": doc_id})`; calls via `asyncio.to_thread`.
- `QdrantIndex(collection, url, api_key=None, dimension)` — port `qdrant.py`; point id = `str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))`; payload `{"chunk_id", "doc_id", "content", **metadata}`; search via `client.query_points(..., query=vector, limit=k, query_filter=...)`; `delete_document` with a `Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])`; sync client through `asyncio.to_thread`.
- `PgVectorIndex(dsn, table, dimension)` — port `pgvector.py`; table columns `chunk_id TEXT PRIMARY KEY, doc_id TEXT, embedding vector(n), content TEXT, metadata JSONB`, index on `doc_id`; filters as `metadata->>$k = $v` with **bound parameters** (`$3, $4, …`); validate `table` against `^[A-Za-z_][A-Za-z0-9_]*$` (SpecError) since it is interpolated; `delete_document` = `DELETE … WHERE doc_id = $1`.

- [ ] **Step 1: Failing tests** — `tests/conftest.py`:
```python
from astromesh_rag.types import Chunk

def chunks(doc, texts, **meta):
    return [Chunk(id=f"{doc}#{i}", doc_id=doc, text=t, metadata=dict(meta)) for i, t in enumerate(texts)]

async def index_contract(index):
    await index.upsert(chunks("a", ["x", "y"], lang="es"), [[1, 0, 0], [0, 1, 0]])
    await index.upsert(chunks("b", ["z"], lang="en"), [[0, 0, 1]])
    hits = await index.search([1, 0.1, 0], 2)
    assert hits[0].chunk.id == "a#0" and hits[0].source == "vector" and hits[0].score > hits[1].score
    assert [h.chunk.id for h in await index.search([0, 0, 1], 5, {"lang": "es"})] in (["a#1", "a#0"], ["a#0", "a#1"])
    await index.delete_document("a")
    assert {h.chunk.doc_id for h in await index.search([1, 1, 1], 5)} == {"b"}
```
`tests/test_index_memory.py`: `async def test_memory_contract(): await index_contract(MemoryIndex())` plus `test_search_empty_index_returns_empty`.
Integration files: each `pytest.importorskip(<module>)`, reads `QDRANT_URL` / `PG_DSN` from env and `pytest.skip` when unset, then runs `index_contract` with `dimension=3` and a unique collection/table per run. `test_pgvector.py` adds `test_pgvector_filter_is_parameterized`: filter `{"lang": "x'; DROP TABLE t;--"}` returns `[]` and the table still exists afterwards.
- [ ] **Step 2: Run** `uv run pytest -q tests/test_index_memory.py` → ImportError.
- [ ] **Step 3: Implement** the five indexes as above.
- [ ] **Step 4: Run** memory tests → pass. Integration locally if services exist: `docker run -d -p 6333:6333 qdrant/qdrant:v1.12.4` and `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16`, then `QDRANT_URL=http://localhost:6333 PG_DSN=postgresql://postgres:pg@localhost:5432/postgres uv run --extra qdrant --extra pgvector pytest -q tests/integrations` → pass (otherwise they skip; CI runs them in Task 10).
- [ ] **Step 5: Commit** — `feat: vector indexes (memory, faiss, chroma, qdrant, pgvector) with delete_document`

---

### Task 4: Embedders, rerankers, OpenAI-compatible LLM

**Files:** Create `astromesh_rag/embedders/{__init__,ollama,hf,sentence_transformers}.py`, `astromesh_rag/rerankers/{__init__,cross_encoder,cohere}.py`, `astromesh_rag/llms/{__init__,openai_compat}.py`, `tests/test_http_stages.py`, `tests/integrations/test_sentence_transformers.py`.

- `OllamaEmbedder(endpoint="http://localhost:11434", model="nomic-embed-text", dimension=None)` — port; `embed(texts)` posts `/api/embed` with `{"model", "input": texts}` (batch endpoint) and returns `["embeddings"]`.
- `HFEmbedder(model, endpoint=None, api_key=None, dimension=None)` — port `hf.py`, batch call, nested-list normalization kept.
- `SentenceTransformersEmbedder(model="all-MiniLM-L6-v2")` — port; `encode` in `asyncio.to_thread`; `dimension` from `model.get_sentence_embedding_dimension()` after first load.
- `CrossEncoderReranker(model)` — port; returns `Hit`s with `score = rerank score`, `details["rerank"] = score`, original `details` kept; predict in `to_thread`.
- `CohereReranker(api_key, model="rerank-english-v3.0")` — port; same `Hit` treatment.
- `OpenAICompatLLM(base_url, api_key, model, temperature: float | None = 0.0, timeout=60.0)` — `POST {base_url}/chat/completions` with `{"model", "messages":[{"role":"user","content":prompt}]}` (+`temperature` only if not None — `kimi-k2.5` rejects non-1 values); returns `choices[0].message.content`; usage tokens appended to `self.usage: list[dict]`.
- All httpx users: `httpx = require("httpx", "<extra>")` inside `__init__`.

- [ ] **Step 1: Failing tests** with `respx`:
```python
import httpx, pytest, respx
from astromesh_rag.embedders import OllamaEmbedder
from astromesh_rag.llms import OpenAICompatLLM
from astromesh_rag.rerankers import CohereReranker
from astromesh_rag.types import Chunk, Hit

@respx.mock
async def test_ollama_batches_in_one_call():
    route = respx.post("http://o/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0], [2.0]]}))
    assert await OllamaEmbedder(endpoint="http://o").embed(["a", "b"]) == [[1.0], [2.0]]
    assert route.call_count == 1

@respx.mock
async def test_openai_compat_returns_message_and_usage():
    respx.post("http://gw/v1/chat/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"content": "hola"}}], "usage": {"total_tokens": 7}}))
    llm = OpenAICompatLLM("http://gw/v1", "k", "kimi")
    assert await llm.complete("hi") == "hola" and llm.usage == [{"total_tokens": 7}]

@respx.mock
async def test_openai_compat_omits_temperature_when_none():
    route = respx.post("http://gw/v1/chat/completions").mock(return_value=httpx.Response(
        200, json={"choices": [{"message": {"content": ""}}]}))
    await OpenAICompatLLM("http://gw/v1", "k", "m", temperature=None).complete("x")
    assert "temperature" not in route.calls[0].request.content.decode()

@respx.mock
async def test_cohere_rerank_keeps_details():
    respx.post("https://api.cohere.ai/v1/rerank").mock(return_value=httpx.Response(
        200, json={"results": [{"index": 1, "relevance_score": 0.9}]}))
    hits = [Hit(Chunk(f"d#{i}", "d", t), 0.1, "vector", {"vector": 0.1}) for i, t in enumerate("ab")]
    out = await CohereReranker(api_key="k").rerank("q", hits, 1)
    assert out[0].chunk.text == "b" and out[0].score == 0.9 and out[0].details == {"vector": 0.1, "rerank": 0.9}
```
Integration `test_sentence_transformers.py` (importorskip): embeds two texts, dimension 384; cross-encoder ranks "Paris is the capital of France" above "Bananas are yellow" for "capital of France".
- [ ] **Step 2–4:** run → fail; implement; run → pass; ruff clean.
- [ ] **Step 5: Commit** — `feat: embedders, rerankers and an OpenAI-compatible LLM`

---

### Task 5: Retrievers, BM25, fusion, context

**Files:** Create `astromesh_rag/retrievers/{__init__,vector,bm25}.py`, `astromesh_rag/fusion.py`, `astromesh_rag/context.py`, `tests/test_retrieval.py`.

**Interfaces — Produces:**
```python
class VectorRetriever:  # name = "vector"
    def __init__(self, embedder, index, min_score: float | None = None, filters: dict | None = None)
    async def retrieve(self, query: str, k: int) -> list[Hit]     # drops hits with score < min_score
class BM25Index:        # also a Retriever; name = "bm25"; KeywordIndex
    def __init__(self, k1: float = 1.5, b: float = 0.75, path: str | None = None)
    async def add(self, chunks: list[Chunk]) -> None
    async def delete_document(self, doc_id: str) -> None
    async def retrieve(self, query: str, k: int) -> list[Hit]     # score = BM25, details={"bm25": s}
    def save(self) -> None; @classmethod def load(cls, path) -> "BM25Index"   # JSON
def tokenize(text: str) -> list[str]   # lowercase, NFKD-strip accents, \w+ tokens; keeps "err-4012" as "err","4012" AND "err-4012"
class RRF:            # fuse(lists, k) ; k_rrf = 60 ; details["rrf"]
class WeightedFusion: # __init__(weights) ; min-max normalize each list then weighted sum ; details["fused"]
def build_context(hits: list[Hit]) -> str   # texts stripped, joined by "\n\n" (same as core format_knowledge)
```
Fused hit: same `Chunk`, `source` = `"+".join(sorted(sources))`, `details` merged from every list.

- [ ] **Step 1: Failing tests**
```python
from astromesh_rag.fusion import RRF, WeightedFusion
from astromesh_rag.retrievers import BM25Index, VectorRetriever, tokenize
from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Chunk, Hit

def C(i, t): return Chunk(f"d{i}#0", f"d{i}", t)

def test_tokenize_keeps_codes_and_strips_accents():
    toks = tokenize("Error ERR-4012 en facturación")
    assert "err-4012" in toks and "facturacion" in toks

async def test_bm25_finds_exact_code():
    bm = BM25Index()
    await bm.add([C(1, "billing failed with ERR-4012"), C(2, "billing works fine"), C(3, "ERR-5000 timeout")])
    hits = await bm.retrieve("what is ERR-4012", 2)
    assert hits[0].chunk.doc_id == "d1" and hits[0].source == "bm25"

async def test_bm25_delete_and_persist(tmp_path):
    bm = BM25Index(path=str(tmp_path / "bm25.json"))
    await bm.add([C(1, "alpha"), C(2, "beta")]); await bm.delete_document("d1"); bm.save()
    again = BM25Index.load(str(tmp_path / "bm25.json"))
    assert [h.chunk.doc_id for h in await again.retrieve("alpha beta", 5)] == ["d2"]

def test_rrf_rewards_agreement():
    a = [Hit(C(1, "x"), 0.9, "vector"), Hit(C(2, "y"), 0.8, "vector")]
    b = [Hit(C(2, "y"), 7.0, "bm25"), Hit(C(3, "z"), 6.0, "bm25")]
    out = RRF().fuse([a, b], 3)
    assert out[0].chunk.id == "d2#0" and out[0].source == "bm25+vector" and "rrf" in out[0].details

def test_weighted_fusion_respects_weights():
    a = [Hit(C(1, "x"), 1.0, "vector"), Hit(C(2, "y"), 0.0, "vector")]
    b = [Hit(C(2, "y"), 1.0, "bm25"), Hit(C(1, "x"), 0.0, "bm25")]
    assert WeightedFusion([0.9, 0.1]).fuse([a, b], 1)[0].chunk.id == "d1#0"

async def test_vector_retriever_threshold():
    idx, emb = MemoryIndex(), HashEmbedder()
    chunks = [C(1, "red apple pie"), C(2, "quantum physics")]
    await idx.upsert(chunks, await emb.embed([c.text for c in chunks]))
    hits = await VectorRetriever(emb, idx, min_score=0.5).retrieve("red apple", 5)
    assert [h.chunk.doc_id for h in hits] == ["d1"]
```
- [ ] **Step 2–4:** run → fail; implement (BM25 standard formula with idf = `log(1 + (N - df + 0.5)/(df + 0.5))`); run → pass.
- [ ] **Step 5: Commit** — `feat: vector and BM25 retrievers, RRF and weighted fusion`

---

### Task 6: Pipelines and trace

**Files:** Create `astromesh_rag/pipeline.py`, `tests/test_pipeline.py`.

**Interfaces — Produces:**
```python
class IngestPipeline:
    def __init__(self, splitter, embedder, index, keyword_index: BM25Index | None = None, batch_size: int = 64)
    async def ingest(self, docs: list[Document]) -> int        # chunks written
class QueryPipeline:
    def __init__(self, retrievers: list[Retriever], *, transform: QueryTransform | None = None,
                 include_original: bool = True, fusion: Fusion | None = None, reranker=None,
                 rerank_k: int | None = None, candidates: int | None = None)
    async def query(self, query: str, k: int = 5) -> QueryResult
class Pipeline:
    def __init__(self, ingest: IngestPipeline, query: QueryPipeline)
    async def ingest(self, docs) -> int ; async def query(self, q, k=5) -> QueryResult
```
**Ingest:** for each doc: `doc_id = doc.resolved_id()`; `await index.delete_document(doc_id)` and same on keyword index; split; embed in batches; upsert; `keyword_index.add`. Returns total chunks.
**Query:** blank query → empty result. `queries = [query]` or `await transform(query)` (+ original first if `include_original`). For each query × retriever: `retrieve(q, candidates or k*2)` timed; exceptions caught → trace entry with `error` and skipped; if **all** retrievals failed → `StageError("retrieve", names, last_exc)`. One list → it; several → `fusion.fuse(lists, candidates)` (default `RRF()`). Reranker → `await reranker.rerank(query, hits, rerank_k or k)`. Truncate to `k`, `context = build_context(hits)`. Each stage appends `{"stage", "name", "ms", "in", "out"}` (+`"error"`) to `trace`. Transform errors raise `StageError("transform", …)`.

- [ ] **Step 1: Failing tests**
```python
import pytest
from astromesh_rag.errors import StageError
from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.pipeline import IngestPipeline, Pipeline, QueryPipeline
from astromesh_rag.retrievers import BM25Index, VectorRetriever
from astromesh_rag.splitters import RecursiveSplitter
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Document

def make(hybrid=False):
    emb, idx, bm = HashEmbedder(), MemoryIndex(), BM25Index()
    ing = IngestPipeline(RecursiveSplitter(chunk_size=40, overlap=0), emb, idx, bm if hybrid else None)
    rets = [VectorRetriever(emb, idx)] + ([bm] if hybrid else [])
    return Pipeline(ing, QueryPipeline(rets)), idx

async def test_two_documents_both_survive():
    p, _ = make()
    await p.ingest([Document("apples are red", id="a"), Document("bananas are yellow", id="b")])
    assert {h.chunk.doc_id for h in (await p.query("are", 10)).hits} == {"a", "b"}

async def test_reingest_with_fewer_chunks_removes_stale():
    p, idx = make()
    await p.ingest([Document("one two three four five six seven eight nine ten eleven twelve", id="a")])
    await p.ingest([Document("short", id="a")])
    assert [h.chunk.id for h in (await p.query("short one two", 10)).hits] == ["a#0"]

async def test_query_empty_index():
    p, _ = make()
    r = await p.query("anything")
    assert r.hits == [] and r.context == ""

async def test_query_blank_string():
    p, _ = make()
    assert (await p.query("   ")).hits == []

async def test_trace_records_each_stage():
    p, _ = make(hybrid=True)
    await p.ingest([Document("ERR-4012 billing failure", id="a")])
    stages = [t["stage"] for t in (await p.query("ERR-4012")).trace]
    assert stages.count("retrieve") == 2 and "fuse" in stages

class Boom:
    name = "boom"
    async def retrieve(self, q, k): raise RuntimeError("down")

async def test_hybrid_survives_one_failing_retriever():
    p, idx = make()
    await p.ingest([Document("apples", id="a")])
    q = QueryPipeline([p.query_pipeline.retrievers[0], Boom()])
    r = await q.query("apples")
    assert r.hits and any(t.get("error") for t in r.trace)

async def test_all_retrievers_failing_raises():
    with pytest.raises(StageError):
        await QueryPipeline([Boom()]).query("x")
```
(`Pipeline` exposes `query_pipeline`; `QueryPipeline` exposes `retrievers`.)
- [ ] **Step 2–4:** run → fail; implement; run → pass.
- [ ] **Step 5: Commit** — `feat: ingest and query pipelines with stable ids, partial-failure fusion and trace`

---

### Task 7: Patterns, YAML spec and built-in registrations

**Files:** Create `astromesh_rag/patterns/{__init__,naive,hybrid,multi_query,hyde}.py`, `astromesh_rag/spec.py`, `astromesh_rag/builtins.py`, `tests/test_patterns.py`, `tests/test_spec.py`, `tests/fixtures/product-knowledge.rag.yaml` (copy of `../astromesh/config/rag/product-knowledge.rag.yaml`).

**Interfaces — Produces:**
```python
# patterns
MULTI_QUERY_PROMPT: str   # "Rewrite the question in {n} different ways ... one per line ...\nQuestion: {query}"
HYDE_PROMPT: str          # "Write a short passage that answers the question ...\nQuestion: {query}"
class LLMQueryVariants:   # QueryTransform; parses non-empty lines, strips numbering, max n
class HypotheticalDocument: # QueryTransform; returns [llm answer]
def naive(embedder, index, *, reranker=None, rerank_k=None, min_score=None, candidates=None) -> QueryPipeline
def hybrid(embedder, index, bm25, *, fusion=None, reranker=None, ...) -> QueryPipeline
def multi_query(embedder, index, llm, *, variants=3, reranker=None, ...) -> QueryPipeline
def hyde(embedder, index, llm, *, reranker=None, ...) -> QueryPipeline
PATTERNS = {"naive": naive, "hybrid": hybrid, "multi_query": multi_query, "hyde": hyde}
# spec.py
@dataclass
class RAGSpec: name; chunking: dict; embeddings: dict; vector_store: dict; reranking: dict; retrieval: dict
def load_spec(raw: dict) -> RAGSpec          # same validation as core spec_from_raw, raising SpecError
def load_spec_file(path) -> RAGSpec
def build(spec: RAGSpec, *, llm=None, registry=registry) -> Pipeline
```
`builtins.py` registers, on import of `astromesh_rag`: splitters `fixed|recursive|sentence|semantic` (semantic gets `ctx.embedder`); embedders `ollama|hf|sentence_transformer|sentence_transformers|hash`; indexes `memory|faiss|chroma|qdrant|pgvector` (pgvector builds DSN from `connection` like core `_pg_dsn`); rerankers `cross_encoder|cohere`; llm `openai_compat`.
`build`: embedder first (dimension = `embeddings.dimension` or embedder's); index; `bm25 = BM25Index(path=retrieval.bm25_path)` when pattern is `hybrid`; reranker if `reranking.enabled`; pattern from `retrieval.pattern` (default `naive`) — unknown → SpecError listing patterns; `multi_query`/`hyde` with `llm=None` → `SpecError("pattern 'hyde' needs an llm: build(spec, llm=...)")`; `min_score = retrieval.similarity_threshold`, `rerank_k = reranking.top_k or retrieval.top_k`, `candidates = retrieval.candidates or top_k*2`. `IngestPipeline` keyword index = the same `bm25` (None otherwise).

- [ ] **Step 1: Failing tests**
```python
# tests/test_patterns.py
from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.patterns import hyde, multi_query
from astromesh_rag.testing import FakeLLM, HashEmbedder

async def test_multi_query_issues_variants_plus_original():
    llm = FakeLLM({"Rewrite": "1. cargo duplicado\n2. doble cobro\n"})
    q = multi_query(HashEmbedder(), MemoryIndex(), llm, variants=2)
    r = await q.query("me cobraron dos veces")
    assert sum(t["stage"] == "retrieve" for t in r.trace) == 3 and len(llm.calls) == 1

async def test_hyde_searches_with_hypothetical_text():
    llm = FakeLLM({"passage": "A duplicate charge is refunded in 5 days."})
    r = await hyde(HashEmbedder(), MemoryIndex(), llm).query("double charge?")
    assert any(t["stage"] == "transform" for t in r.trace)
# tests/test_spec.py
import pytest, yaml
from pathlib import Path
from astromesh_rag.errors import SpecError
from astromesh_rag.spec import build, load_spec
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
    assert spec.name == "product-knowledge" and spec.vector_store["backend"] == "pgvector"

async def test_core_yaml_builds_and_answers_in_core_shape():
    p = build(load_spec(raw(similarity_threshold=0.0)))
    await p.ingest([Document("pgvector stores product docs", id="p")])
    items = (await p.query("product docs", 3)).to_list()
    assert items and set(items[0]) >= {"doc_id", "content", "metadata", "score"}

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
    d = raw(); d["spec"]["vector_store"] = "faiss"
    with pytest.raises(SpecError):
        load_spec(d)
```
- [ ] **Step 2–4:** run → fail; implement; run → pass. `uv run pytest -q` all green.
- [ ] **Step 5: Commit** — `feat: naive, hybrid, multi_query and hyde patterns; YAML spec compatible with the core`

---

### Task 8: Retrieval metrics

**Files:** Create `astromesh_rag/metrics.py`, `tests/test_metrics.py`.

**Interfaces — Produces:** pure functions over `ranked: list[str]` (doc ids in rank order) and `relevant: set[str]`:
`recall_at_k(ranked, relevant, k)`, `precision_at_k(...)`, `hit_rate_at_k(...)` (1.0/0.0), `mrr(ranked, relevant, k=None)`, `ndcg_at_k(ranked, relevant, k)` (binary gains, log2), `dedupe(ranked)` (first occurrence; chunks of one doc count once). Empty `relevant` → `ValueError`.

- [ ] **Step 1: Failing tests (hand-computed)**
```python
import math, pytest
from astromesh_rag.metrics import dedupe, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

R = ["d3", "d1", "d7", "d2"]; REL = {"d1", "d2"}

def test_recall_precision():
    assert recall_at_k(R, REL, 2) == 0.5 and precision_at_k(R, REL, 2) == 0.5
    assert recall_at_k(R, REL, 4) == 1.0

def test_mrr_first_relevant_at_rank_2():
    assert mrr(R, REL) == 0.5 and mrr(R, {"zz"}) == 0.0

def test_ndcg():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(R, REL, 4) == pytest.approx(dcg / idcg)

def test_hit_rate_and_dedupe():
    assert hit_rate_at_k(R, REL, 1) == 0.0 and hit_rate_at_k(R, REL, 2) == 1.0
    assert dedupe(["a", "a", "b"]) == ["a", "b"]

def test_empty_relevant_rejected():
    with pytest.raises(ValueError):
        recall_at_k(R, set(), 3)
```
- [ ] **Step 2–4:** run → fail; implement; pass.
- [ ] **Step 5: Commit** — `feat: retrieval metrics (recall, precision, hit rate, MRR, nDCG)`

---

### Task 9: Simulated corpus, examples, benchmark

**Files:** Create `examples/corpus/generate.py`, `examples/corpus/corpus.jsonl`, `examples/corpus/queries.jsonl`, `examples/corpus/README.md`, `examples/{_common,naive,hybrid,multi_query,hyde}.py`, `examples/pipeline.rag.yaml`, `examples/benchmark.py`, `tests/test_corpus.py`, `tests/test_benchmark_smoke.py`.

**Corpus** — product "Nimbus", a fictional cloud-storage SaaS (Spanish primary, ~20% English). `generate.py` (stdlib only, `random.Random(20261014)`) writes `corpus.jsonl` lines `{"id","text","metadata":{"type","trap","lang","version"?}}`:

| Family | Count | Content | `trap` |
|---|---|---|---|
| Error articles | 40 | `ERR-<4 digits>`: cause, fix; ~15 pairs share wording and differ only in the code | `exact` |
| Plans / SKUs | 20 | `SKU-<2 digits><letter>` with limits and price | `exact` |
| Billing / account FAQs | 50 | written in formal vocabulary ("cargo duplicado", "reembolso prorrateado", "baja de la suscripción") | `paraphrase` |
| Policies with versions | 2 × 15 | same title, `version: 2024` vs `2026`, different numbers; the 2024 one says "Reemplazada por la versión 2026" | `distractor` |
| Noise | 60 | blog posts and release notes reusing product words without answering anything | `noise` |

Templates are hand-written strings in `generate.py` combined from small vocab lists so text is varied but reproducible. `queries.jsonl` (hand-curated, committed, not generated): 60 lines `{"id","query","relevant":[doc ids],"type","lang"}` — 15 `exact` ("¿qué significa ERR-4012?"), 20 `paraphrase` in colloquial wording ("me cobraron dos veces este mes"), 15 `distractor` (answer only in the 2026 version), 10 `noise`-adjacent. Every `relevant` id must exist (test).

**Examples:** `_common.py` loads the corpus, builds models from env (`RAG_EMBEDDER=hash|st`, default `hash`; `RAG_BENCH_BASE_URL/API_KEY/MODEL` → `OpenAICompatLLM`, else `FakeLLM` with canned variants) and prints a warning line when running on `hash`/`FakeLLM` ("scores are meaningless with the test embedder"). Each pattern file ingests, runs 3 fixed queries and prints rank, score, `source`, `details` and the trace table. `pipeline.rag.yaml` shows the YAML equivalent of `hybrid`.

**Benchmark** `examples/benchmark.py [--embedder hash|st] [--rerank both|on|off] [--k 5] [--out results/]`: one ingestion into `MemoryIndex` + `BM25Index`, then for each pattern × rerank setting × query: ranked doc ids (deduped), metrics from Task 8, latency, LLM calls (count transform calls). Prints a Markdown table overall and per `type`, writes `results/<YYYY-MM-DD>-<embedder>.json` with config, model names, `base_url` host only, corpus SHA-256, per-query rows. Never writes the API key.

- [ ] **Step 1: Failing tests**
```python
# tests/test_corpus.py
import json
from pathlib import Path
C = Path(__file__).parents[1] / "examples" / "corpus"

def test_corpus_regenerates_identically(tmp_path):
    import runpy, sys
    sys.argv = ["generate.py", str(tmp_path / "c.jsonl")]
    runpy.run_path(str(C / "generate.py"), run_name="__main__")
    assert (tmp_path / "c.jsonl").read_bytes() == (C / "corpus.jsonl").read_bytes()

def test_queries_reference_real_docs_and_cover_types():
    ids = {json.loads(l)["id"] for l in (C / "corpus.jsonl").read_text().splitlines()}
    qs = [json.loads(l) for l in (C / "queries.jsonl").read_text().splitlines()]
    assert len(qs) >= 60 and {q["type"] for q in qs} == {"exact", "paraphrase", "distractor", "noise"}
    assert all(set(q["relevant"]) <= ids and q["relevant"] for q in qs)
# tests/test_benchmark_smoke.py
import json, runpy, sys
from pathlib import Path

def test_benchmark_runs_offline(tmp_path, monkeypatch):
    for v in ("RAG_BENCH_BASE_URL", "RAG_BENCH_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    sys.argv = ["benchmark.py", "--embedder", "hash", "--rerank", "off", "--out", str(tmp_path)]
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "benchmark.py"), run_name="__main__")
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert {r["pattern"] for r in data["summary"]} == {"naive", "hybrid", "multi_query", "hyde"}
    assert "api_key" not in json.dumps(data).lower()
```
- [ ] **Step 2: Run** → fail. **Step 3:** write `generate.py`, run it to produce `corpus.jsonl`; write `queries.jsonl` by hand reading the corpus; write examples and benchmark. **Step 4:** `uv run pytest -q tests/test_corpus.py tests/test_benchmark_smoke.py` pass; `uv run python examples/hybrid.py` prints hits and a trace.
- [ ] **Step 5: Commit** — `feat: simulated Nimbus corpus, per-pattern examples and a benchmark`

---

### Task 10: CI and release workflows

**Files:** Create `.github/workflows/ci.yml`, `.github/workflows/release.yml` (copy okf's `release.yml`: tag-version check, build, TestPyPI then PyPI with trusted publishing, smoke install), `uv.lock` (`uv lock`).

`ci.yml` jobs: `lint` (ruff check + format --check); `core` matrix py 3.10–3.13 (`uv sync --locked`, `pytest -q` — integrations skip); `extras` matrix: `faiss` (test_faiss), `chroma` (test_chroma), `sentence-transformers` (test_sentence_transformers); `qdrant` with `services: qdrant: image: qdrant/qdrant:v1.12.4, ports: [6333:6333]` and `QDRANT_URL=http://localhost:6333`; `pgvector` with `services: postgres: image: pgvector/pgvector:pg16, env: POSTGRES_PASSWORD: pg, ports: [5432:5432]`, health check `pg_isready`, `PG_DSN=postgresql://postgres:pg@localhost:5432/postgres`. Integration tests must **not** skip in their job: set `RAG_REQUIRE_INTEGRATION=1`, and conftest turns the env-missing skip into a failure when it is set.

- [ ] **Step 1:** write workflows; `uv lock`; `uv run --with pyyaml python -c "import yaml,sys;[yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml`.
- [ ] **Step 2:** `uv sync --locked && uv run pytest -q && uv run ruff check . && uv run ruff format --check .` green.
- [ ] **Step 3: Commit** — `ci: lint, tests on 3.10–3.13, integration jobs with Qdrant and pgvector, release workflow`

---

### Task 11: Logo, README (designed), docs

**Files:** Create `assets/logo-mark.svg`, `assets/logo.png`, `assets/icon.png`, `README.md`, `README.es.md`, `docs/patterns.md`, `docs/spec.md`, `docs/extending.md`, `CHANGELOG.md`, `CONTRIBUTING.md`.

- [ ] **Step 1: Logo.** Same family as `../astromesh-okf/assets/logo-mark.svg` and `../astromesh-sdd/assets/logo-mark.svg` (navy, two orbits, glow, wordmark `ASTRO` white / `MESH #1fb4f2` / `RAG` in its own accent). Central object: retrieval made visible — a stack of document cards being pulled by a beam/lens into one highlighted card (choose and render 2 options, pick with the user). Render with headless Chrome as done for SDD; read the PNGs before committing.
- [ ] **Step 2: README** — invoke `frontend-design:frontend-design` first, then write in this order (spec §6): logo; pitch "Every RAG pattern, one set of parts."; badges (layer first: `astromesh-runtime` or the sector chosen with the user, release, CI, PyPI, license); 4-line plain-language lifecycle; **mindmap** (Astromesh `cScale` palette as in `../astromesh-sdd/README.md`) with branches Ingest, Retrieve, Patterns, Integrations, Make it yours; **one flowchart per pattern** + table (when to use / cost / LLM?); **sequenceDiagram** of a hybrid query (embed ∥ BM25 → RRF → rerank → context, with trace); **benchmark section** (placeholder heading only, filled in Task 13 — the section states "results from the run on <date>" and is not published before that run); quickstart Python (5 lines) and YAML; extras table; "Make it yours" (registry); "How Astromesh uses it"; roadmap (phases 2–5); contributing; license. Render every Mermaid block with mermaid@11.4.1 headless (as done for SDD) and look at the screenshot: no syntax errors, no clipped labels, palette applied. `README.es.md` mirrors it with Spanish diagram labels.
- [ ] **Step 3: docs/** — `patterns.md` (each pattern: idea, stages, YAML, Python, failure modes, cost), `spec.md` (every YAML key, default, effect — including `similarity_threshold` semantics and `reranking.top_k`), `extending.md` (write a `VectorIndex`, register it, use it from YAML; a custom `QueryTransform`). `CHANGELOG.md` `[Unreleased]` → Added/Fixed (fixed = the core bugs the port corrects). `CONTRIBUTING.md` like okf's.
- [ ] **Step 4: Commit** — `docs: logo, designed README with pattern diagrams, docs and changelog`

---

### Task 12: Publish 0.1.0 (requires explicit user confirmation at each external step)

- [ ] Ask the user to confirm creating the public repo, then `gh repo create monaccode/astromesh-rag --public --source . --push` + topics `rag, retrieval-augmented-generation, llm, vector-search, bm25, hybrid-search, hyde, astromesh`.
- [ ] Watch CI until every job is green (fix and push otherwise).
- [ ] PyPI/TestPyPI trusted publishers and the `pypi`/`testpypi` environments are manual steps on pypi.org/GitHub: give the user the exact values (owner `monaccode`, repo `astromesh-rag`, workflow `release.yml`, environments) and wait for confirmation.
- [ ] Move `[Unreleased]` → `[0.1.0] - <date>`, commit `chore(release): astromesh-rag 0.1.0`, tag `v0.1.0` annotated, push the tag alone, watch `release.yml`; `pip install astromesh-rag==0.1.0` in a clean venv and run `examples/naive.py`.

---

### Task 13: Real benchmark run and published results

- [ ] Get from the user the Academia gateway URL and a key; export `RAG_BENCH_BASE_URL`, `RAG_BENCH_API_KEY`, `RAG_BENCH_MODEL` in the shell only.
- [ ] `uv run --extra sentence-transformers --extra openai python examples/benchmark.py --embedder st --rerank both --out results/` → read the table; check it is plausible (e.g. `hybrid` should not lose to `naive` on `exact`; if it does, investigate before publishing — a bug is likelier than a finding).
- [ ] Render a grouped bar chart (nDCG@10 per pattern × query type) to `assets/benchmark.png` (load the `dataviz` skill first).
- [ ] Fill the README/README.es benchmark section from the JSON: overall table, per-type table, chart, run date, models, corpus hash, and a short honest reading — including any hypothesis from the spec that the data did not support.
- [ ] Commit `results/<date>-st.json`, chart and README: `docs: benchmark results from <date>`; push.

---

### Task 14: Core migration (repo `../astromesh`, branch `feat/rag-library`)

**Files:** Modify `pyproject.toml` (+ `astromesh-rag>=0.1,<0.2`; map core extras `chromadb`, `qdrant`, `faiss`, `postgres`, `embeddings` to also pull `astromesh-rag[<extra>]`), `astromesh/rag/{pipeline,factory,loader,agent_rag}.py` (facade), `astromesh/rag/__init__.py` (native registration), `astromesh/tools/builtin/rag.py` (`doc_id`), `astromesh/runtime/engine.py` (pass `llm` adapter to `build_pipeline`), `CHANGELOG.md`; Delete `astromesh/rag/{chunking,embeddings,stores,reranking}/`; tests updated.

- [ ] **Step 1: Failing tests** in `tests/test_rag_library.py`:
```python
from astromesh.rag.factory import build_pipeline
from astromesh.rag.loader import spec_from_raw

RAW = {"kind": "RAGPipeline", "metadata": {"name": "t"}, "spec": {
    "embeddings": {"provider": "hash"}, "vector_store": {"backend": "memory"}}}

async def test_second_ingest_does_not_overwrite_first():
    p = build_pipeline(spec_from_raw(RAW))
    await p.ingest("apples are red", {}, doc_id="a")
    await p.ingest("bananas are yellow", {}, doc_id="b")
    contents = {c["content"] for c in (await p.query("are", top_k=10)).chunks}
    assert contents == {"apples are red", "bananas are yellow"}

def test_spec_errors_stay_value_errors():
    import pytest
    with pytest.raises(ValueError):
        spec_from_raw({"kind": "Agent"})

def test_native_chunker_registered_when_available():
    from astromesh_rag import registry
    import astromesh.rag  # noqa: F401  registers
    assert "recursive" in registry.names("splitter")
```
- [ ] **Step 2:** `uv run pytest tests/test_rag_library.py` → fail.
- [ ] **Step 3: Facade.** `pipeline.py`: `RAGResult(chunks, query, metadata)` kept; `class RAGPipeline` wraps an `astromesh_rag.Pipeline`: `async ingest(document, metadata, doc_id_prefix=None, doc_id=None) -> int` (doc id = `doc_id or doc_id_prefix` else content hash); `async query(query, top_k=5) -> RAGResult(chunks=result.to_list(), query=query, metadata={"trace": result.trace})`; `result_to_list` unchanged. `loader.py`: `RAGPipelineSpec`/`spec_from_raw` delegate to `astromesh_rag.spec.load_spec`, converting `SpecError` → `ValueError`. `factory.build_pipeline(spec, llm=None)` → `RAGPipeline(astromesh_rag.spec.build(...))`. `agent_rag.py` unchanged. `__init__.py`: if `astromesh._native.rust_recursive_chunk` imports and `ASTROMESH_FORCE_PYTHON` is unset, register a `splitter/recursive` factory whose `split` calls it and converts dicts to `Chunk`s with `chunk_id`. `engine._resolve_rag`: build an `LLM` adapter around the agent's model router (`complete(prompt)` → one router call, text out) only when `retrieval.pattern in {"multi_query","hyde"}`. `rag_ingest` tool: optional `doc_id` parameter.
- [ ] **Step 4:** rewrite tests that import deleted modules (`test_rag.py`, `test_rag_factory.py`) to target the facade or remove assertions now covered by the library's own tests; run the full suite `uv run pytest -q` → green; `uv run ruff check astromesh/ tests/ && uv run ruff format --check astromesh/ tests/`; `ASTROMESH_FORCE_PYTHON=1 uv run pytest -q tests/test_rag*.py` green; `uv run python -c "import astromesh.api.main"` with base deps only.
- [ ] **Step 5:** `uv lock` in root, `astromesh-node/`, `astromesh-cli/` (they reference the core). CHANGELOG via `/changelog-automation`: Changed (RAG delegated to astromesh-rag), Fixed (ingest overwrite; Qdrant ids; pgvector filter injection; semantic chunking inert; similarity_threshold and reranking.top_k inert). Commit `refactor(rag): delegate RAG to astromesh-rag` on `feat/rag-library`; do not merge or release without the user.

---

### Task 15: docs-site RAG section (after 0.1.0 is on PyPI and the core release ships)

**Files (repo `../astromesh`, worktree from `origin/develop`):** `docs-site/src/content/docs/rag/{introduction,patterns,yaml-reference,extending,in-astromesh}.mdx`, `docs-site/astro.config.mjs` (sidebar "RAG"), `docs-site/src/data/ecosystem.ts` (entry `rag`, sector agreed with the user), `docs-site/src/components/RagShowcase.astro` + `index.mdx`, `getting-started/ecosystem.md` and `architecture/ecosystem-dependencies.md` rows.

- [ ] Write pages from the repo docs (patterns with diagrams, YAML reference, extending, benchmark table + chart from Task 13, how the core uses it).
- [ ] `npm test` and `npm run build` in `docs-site` (symlink `node_modules` from the main checkout), screenshot the RAG intro and the home showcase, read them.
- [ ] Ask the user before pushing to `develop` and `main`; watch `docs.yml`; curl the live pages.
