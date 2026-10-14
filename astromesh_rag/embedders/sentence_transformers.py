import asyncio

from astromesh_rag._extras import require


class SentenceTransformersEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self._st = require("sentence_transformers", "sentence-transformers")
        self.model_name = model
        self._model = None
        self.dimension: int | None = None

    def _load(self):
        if self._model is None:
            self._model = self._st.SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def run():
            return self._load().encode(texts).tolist()

        return await asyncio.to_thread(run)
