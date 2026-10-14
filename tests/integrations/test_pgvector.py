import os
import uuid

import pytest
from conftest import chunks, index_contract

pytest.importorskip("asyncpg")

from astromesh_rag.indexes.pgvector import PgVectorIndex


def _index():
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        pytest.skip("PG_DSN not set")
    return PgVectorIndex(dsn=dsn, table=f"t_{uuid.uuid4().hex}", dimension=3)


async def test_pgvector_contract():
    await index_contract(_index())


async def test_pgvector_filter_is_parameterized():
    index = _index()
    await index.upsert(chunks("a", ["x"], lang="es"), [[1, 0, 0]])
    assert await index.search([1, 0, 0], 5, {"lang": "x'; DROP TABLE t;--"}) == []
    assert len(await index.search([1, 0, 0], 5)) == 1  # table still exists
