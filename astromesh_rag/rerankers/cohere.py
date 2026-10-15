from dataclasses import replace

from astromesh_rag._extras import raise_for_status, require
from astromesh_rag.types import Hit

_URL = "https://api.cohere.ai/v1/rerank"


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self._httpx = require("httpx", "cohere")
        self.api_key = api_key
        self.model = model

    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        async with self._httpx.AsyncClient() as client:
            r = await client.post(
                _URL,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [h.chunk.text for h in hits],
                    "top_n": k,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            raise_for_status(r)
        out = [
            replace(
                hits[i["index"]],
                score=i["relevance_score"],
                details={**hits[i["index"]].details, "rerank": i["relevance_score"]},
            )
            for i in r.json().get("results", [])
        ]
        return sorted(out, key=lambda h: h.score, reverse=True)[:k]
