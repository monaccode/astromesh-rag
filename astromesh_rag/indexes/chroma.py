"""ChromaDB index (cosine space)."""

from __future__ import annotations

import asyncio

from .._extras import require
from ..types import Chunk, Hit


class ChromaIndex:
    def __init__(self, collection, host=None, port=8000, persist_directory=None):
        chromadb = require("chromadb", "chroma")
        if host:
            client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client()
        self._col = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    async def upsert(self, chunks, vectors):
        if not chunks:
            return
        # Chroma metadata values must be scalars
        metas = [
            {
                **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
                "doc_id": c.doc_id,
            }
            for c in chunks
        ]
        await asyncio.to_thread(
            self._col.upsert,
            ids=[c.id for c in chunks],
            embeddings=[list(map(float, v)) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=metas,
        )

    async def search(self, vector, k, filters=None):
        kw = {"query_embeddings": [list(map(float, vector))], "n_results": k}
        if filters:
            items = [{fk: fv} for fk, fv in filters.items()]
            kw["where"] = items[0] if len(items) == 1 else {"$and": items}
        r = await asyncio.to_thread(self._col.query, **kw)
        hits = []
        for i, cid in enumerate(r["ids"][0]):
            meta = dict(r["metadatas"][0][i])
            doc_id = meta.pop("doc_id")
            s = 1 - r["distances"][0][i]
            hits.append(
                Hit(
                    chunk=Chunk(id=cid, doc_id=doc_id, text=r["documents"][0][i], metadata=meta),
                    score=s,
                    source="vector",
                    details={"vector": s},
                )
            )
        return hits

    async def delete_document(self, doc_id):
        await asyncio.to_thread(self._col.delete, where={"doc_id": doc_id})
