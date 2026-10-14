from astromesh_rag._extras import require

_DEFAULT_API = "https://api-inference.huggingface.co/pipeline/feature-extraction"


class HFEmbedder:
    """Hugging Face Inference API / TEI embedder."""

    def __init__(
        self,
        model: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
    ):
        self._httpx = require("httpx", "hf")
        self.model = model
        self.endpoint = endpoint or f"{_DEFAULT_API}/{model}"
        self.api_key = api_key
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with self._httpx.AsyncClient() as client:
            r = await client.post(
                self.endpoint, json={"inputs": texts}, headers=headers, timeout=60.0
            )
            r.raise_for_status()
            data = r.json()
        # HF may nest one level deeper per input: [[[...]]] -> [[...]]
        return [v[0] if v and isinstance(v[0], list) else v for v in data]
