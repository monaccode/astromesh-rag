"""Qdrant index. url=":memory:" runs an in-process instance."""

from __future__ import annotations

import asyncio
import uuid

from .._extras import require
from ..types import Chunk, Hit


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantIndex:
    def __init__(self, collection, url="http://localhost:6333", api_key=None, dimension=384):
        require("qdrant_client", "qdrant")
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection = collection
        self._client = (
            QdrantClient(location=":memory:")
            if url == ":memory:"
            else QdrantClient(url=url, api_key=api_key)
        )
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    @staticmethod
    def _filter(conds: dict):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in conds.items()]
        )

    async def upsert(self, chunks, vectors):
        if not chunks:
            return
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_point_id(c.id),
                vector=[float(x) for x in v],
                payload={**c.metadata, "chunk_id": c.id, "doc_id": c.doc_id, "content": c.text},
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        await asyncio.to_thread(self._client.upsert, collection_name=self.collection, points=points)

    async def search(self, vector, k, filters=None):
        r = await asyncio.to_thread(
            self._client.query_points,
            collection_name=self.collection,
            query=[float(x) for x in vector],
            limit=k,
            query_filter=self._filter(filters) if filters else None,
            with_payload=True,
        )
        hits = []
        for p in r.points:
            pl = dict(p.payload)
            cid, doc_id, text = pl.pop("chunk_id"), pl.pop("doc_id"), pl.pop("content")
            hits.append(
                Hit(
                    chunk=Chunk(id=cid, doc_id=doc_id, text=text, metadata=pl),
                    score=float(p.score),
                    source="vector",
                    details={"vector": float(p.score)},
                )
            )
        return hits

    async def delete_document(self, doc_id):
        await asyncio.to_thread(
            self._client.delete,
            collection_name=self.collection,
            points_selector=self._filter({"doc_id": doc_id}),
        )
