import asyncio
import threading

from astromesh_rag._extras import require


class SentenceTransformersEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self._st = require("sentence_transformers", "sentence-transformers")
        self.model_name = model
        self._model = None
        self._lock = threading.Lock()
        self.dimension: int | None = None

    def _load(self):
        with self._lock:
            if self._model is None:
                m = self._st.SentenceTransformer(self.model_name)
                dim = getattr(m, "get_embedding_dimension", None)
                self.dimension = (dim or m.get_sentence_embedding_dimension)()
                self._model = m
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def run():
            return self._load().encode(texts).tolist()

        return await asyncio.to_thread(run)
