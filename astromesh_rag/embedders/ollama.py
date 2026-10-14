from astromesh_rag._extras import require


class OllamaEmbedder:
    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimension: int | None = None,
    ):
        self._httpx = require("httpx", "ollama")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.endpoint}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=60.0,
            )
            r.raise_for_status()
            return r.json()["embeddings"]
