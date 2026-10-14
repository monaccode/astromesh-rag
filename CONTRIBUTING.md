# Contributing

Thanks for helping. Bug reports with a few documents, a query and the `trace` that shows
the problem are the most useful contribution there is.

## Setup

You need [uv](https://docs.astral.sh/uv/). `uv.lock` is committed and CI installs with
`--locked`, so run `uv lock` and commit the lock whenever you change `pyproject.toml`.

```bash
git clone https://github.com/monaccode/astromesh-rag
cd astromesh-rag
uv sync
```

## Tests

```bash
uv run pytest -q                       # core: no extras, no network, no models
uv run --python 3.10 pytest -q         # the oldest supported Python
```

The core suite runs on `HashEmbedder` and `FakeLLM` from `astromesh_rag.testing`, so it
needs nothing but PyYAML. Each backend has its own test under `tests/integrations/`, which
skips when its extra is not installed. Install one extra at a time, as CI does:

```bash
uv run --extra faiss pytest -q tests/integrations/test_faiss.py
uv run --extra chroma pytest -q tests/integrations/test_chroma.py
uv run --extra sentence-transformers pytest -q tests/integrations/test_sentence_transformers.py
```

Qdrant and pgvector need a server. Without one, the Qdrant test runs against an in-process
instance and the pgvector test skips:

```bash
docker run -d -p 6333:6333 qdrant/qdrant:v1.12.4
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16
export QDRANT_URL=http://localhost:6333 PG_DSN=postgresql://postgres:pg@localhost:5432/postgres
RAG_REQUIRE_INTEGRATION=1 uv run --extra qdrant pytest -q tests/integrations/test_qdrant.py
RAG_REQUIRE_INTEGRATION=1 uv run --extra pgvector pytest -q tests/integrations/test_pgvector.py
```

`RAG_REQUIRE_INTEGRATION=1` turns a missing server URL into a failure instead of a skip;
CI sets it.

The examples and the benchmark run offline by default (`HashEmbedder`, `FakeLLM`) and say
so; see the [README](README.md#benchmark) for a real run.

## Lint

```bash
uv run ruff check .
uv run ruff format --check .   # CI gate; also formats Python blocks inside Markdown
uv run ruff format .           # writes the fix
```

## Commits

[Conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`test:`, `chore:`, `ci:`. Add user-visible changes to `CHANGELOG.md` under `[Unreleased]`.

## Adding a backend

A vector database, an embedder or a reranker. [docs/extending.md](docs/extending.md)
explains the contract; to ship one in the library:

1. **Module.** Add it under `astromesh_rag/indexes/`, `embedders/` or `rerankers/`. Import
   its client in the constructor with `require("<module>", "<extra>")` from
   `astromesh_rag._extras`, never at module level: importing `astromesh_rag` must not need
   any extra. Run blocking client calls through `asyncio.to_thread`. An index scores by
   cosine similarity, sets `source="vector"` and `details={"vector": score}`, and deletes
   every chunk of a document in `delete_document`.
2. **Register it.** Add a factory and a row to the table at the bottom of
   `astromesh_rag/builtins.py`, reading its keys from the YAML section with defaults.
3. **Extra.** Add `<extra> = ["<package>>=<version you tested>"]` to
   `[project.optional-dependencies]` (a floor, no ceiling), add it to `all`, and run
   `uv lock`.
4. **Test.** Add `tests/integrations/test_<name>.py` with `pytest.importorskip`. An index
   must pass `index_contract` from `tests/conftest.py`; a server-backed one reads its URL
   with `require_env`. Add a job (or a row in the `extras` matrix) to
   `.github/workflows/ci.yml`.
5. **Docs.** A row in the extras table of `README.md` and `README.es.md`, and its keys in
   `docs/spec.md`.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and `__version__` in `astromesh_rag/__init__.py`,
   run `uv lock`, and move `[Unreleased]` in `CHANGELOG.md` to a dated version section.
2. Tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` checks that the tag
   matches the version, runs the tests and lint, builds with `uv build`, publishes to
   TestPyPI, installs that build from TestPyPI and runs `examples/naive.py` offline, and
   only then publishes to PyPI. Both uploads use trusted publishing and skip files that
   already exist, so re-running a release is safe.

Trusted publishing needs a one-time setup by a maintainer, on both indexes: a publisher for
the `astromesh-rag` project with owner `monaccode`, repository `astromesh-rag` and workflow
`release.yml` (environment `testpypi` on test.pypi.org, `pypi` on pypi.org), and both
environments created in the GitHub repository settings.
