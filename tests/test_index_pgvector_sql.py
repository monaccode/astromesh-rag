import pytest

from astromesh_rag.errors import SpecError
from astromesh_rag.indexes.pgvector import PgVectorIndex, _where


def test_filter_uses_placeholders_not_values():
    sql, params = _where({"lang": "x'; DROP TABLE t;--", "k": "v"}, start=3)
    assert sql == "WHERE metadata->>$3 = $4 AND metadata->>$5 = $6"
    assert "DROP" not in sql
    assert params == ["lang", "x'; DROP TABLE t;--", "k", "v"]


def test_no_filter_is_empty():
    assert _where(None, start=3) == ("", [])


def test_table_name_validated():
    with pytest.raises(SpecError):
        PgVectorIndex(dsn="x", table="t; DROP TABLE y", dimension=3)


def test_filter_values_json_encoded_for_non_strings():
    _, params = _where({"a": True, "n": 3, "s": "x"}, start=1)
    assert params == ["a", "true", "n", "3", "s", "x"]


def _fake_asyncpg(monkeypatch):
    import asyncio

    from astromesh_rag.indexes import pgvector

    created = []

    class Conn:
        async def execute(self, *a):
            await asyncio.sleep(0)

        async def fetch(self, *a):
            return []

    class Acq:
        async def __aenter__(self):
            return Conn()

        async def __aexit__(self, *a):
            return False

    class Pool:
        def acquire(self):
            return Acq()

    class Mod:
        @staticmethod
        async def create_pool(dsn):
            created.append(dsn)
            await asyncio.sleep(0.01)
            return Pool()

    monkeypatch.setattr(pgvector, "require", lambda m, e: Mod)
    return created


async def test_concurrent_first_calls_create_one_pool(monkeypatch):
    import asyncio

    created = _fake_asyncpg(monkeypatch)
    idx = PgVectorIndex(dsn="d", table="t", dimension=3)
    await asyncio.gather(*(idx.search([1, 0, 0], 1) for _ in range(4)))
    assert created == ["d"]


def test_missing_asyncpg_fails_at_construction(monkeypatch):
    from astromesh_rag.errors import MissingExtraError
    from astromesh_rag.indexes import pgvector

    def boom(module, extra):
        raise MissingExtraError(extra, module)

    monkeypatch.setattr(pgvector, "require", boom)
    with pytest.raises(MissingExtraError, match="astromesh-rag\\[pgvector\\]"):
        PgVectorIndex(dsn="d", table="t", dimension=3)
