"""PostgreSQL + pgvector index."""

from __future__ import annotations

import asyncio
import json
import re

from .._extras import require
from ..errors import SpecError
from ..types import Chunk, Hit

_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _where(filters: dict | None, start: int) -> tuple[str, list]:
    """WHERE clause with bound parameters ($start, $start+1, ...); values never enter the SQL."""
    if not filters:
        return "", []
    parts, params = [], []
    for key, value in filters.items():
        n = start + len(params)
        parts.append(f"metadata->>${n} = ${n + 1}")
        params += [key, value if isinstance(value, str) else json.dumps(value)]
    return "WHERE " + " AND ".join(parts), params


def _vec(v) -> str:
    return "[" + ",".join(str(float(x)) for x in v) + "]"


class PgVectorIndex:
    def __init__(self, dsn, table, dimension):
        if not _TABLE.match(table):
            raise SpecError(f"invalid pgvector table name: {table!r}")
        self._asyncpg = require("asyncpg", "pgvector")
        self.dsn, self.table, self.dimension = dsn, table, dimension
        self._pool = None
        self._lock = asyncio.Lock()

    async def _get_pool(self):
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    pool = await self._asyncpg.create_pool(self.dsn)
                    async with pool.acquire() as conn:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        await conn.execute(
                            f"CREATE TABLE IF NOT EXISTS {self.table} ("
                            f"chunk_id TEXT PRIMARY KEY, doc_id TEXT, "
                            f"embedding vector({int(self.dimension)}), "
                            "content TEXT, metadata JSONB)"
                        )
                        await conn.execute(
                            f"CREATE INDEX IF NOT EXISTS {self.table}_doc_id "
                            f"ON {self.table} (doc_id)"
                        )
                    self._pool = pool
        return self._pool

    async def upsert(self, chunks, vectors):
        if not chunks:
            return
        pool = await self._get_pool()
        rows = [
            (c.id, c.doc_id, _vec(v), c.text, json.dumps(c.metadata))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                f"INSERT INTO {self.table} (chunk_id, doc_id, embedding, content, metadata) "
                "VALUES ($1, $2, $3::vector, $4, $5::jsonb) "
                "ON CONFLICT (chunk_id) DO UPDATE SET doc_id = EXCLUDED.doc_id, "
                "embedding = EXCLUDED.embedding, content = EXCLUDED.content, "
                "metadata = EXCLUDED.metadata",
                rows,
            )

    async def search(self, vector, k, filters=None):
        pool = await self._get_pool()
        where, params = _where(filters, start=3)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT chunk_id, doc_id, content, metadata, 1 - (embedding <=> $1::vector) AS score "
                f"FROM {self.table} {where} ORDER BY embedding <=> $1::vector LIMIT $2",
                _vec(vector),
                k,
                *params,
            )
        return [
            Hit(
                chunk=Chunk(
                    id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    text=r["content"],
                    metadata=json.loads(r["metadata"]),
                ),
                score=float(r["score"]),
                source="vector",
                details={"vector": float(r["score"])},
            )
            for r in rows
        ]

    async def delete_document(self, doc_id):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.table} WHERE doc_id = $1", doc_id)

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
