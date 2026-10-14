import asyncio
from dataclasses import replace

from astromesh_rag._extras import require
from astromesh_rag.types import Hit


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._st = require("sentence_transformers", "sentence-transformers")
        self.model_name = model
        self._model = None

    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []

        def run():
            if self._model is None:
                self._model = self._st.CrossEncoder(self.model_name)
            return self._model.predict([[query, h.chunk.text] for h in hits])

        scores = await asyncio.to_thread(run)
        out = [
            replace(h, score=float(s), details={**h.details, "rerank": float(s)})
            for h, s in zip(hits, scores, strict=True)
        ]
        return sorted(out, key=lambda h: h.score, reverse=True)[:k]
