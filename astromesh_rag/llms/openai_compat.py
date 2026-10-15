from astromesh_rag._extras import raise_for_status, require


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None = 0.0,
        timeout: float = 60.0,
    ):
        self._httpx = require("httpx", "openai")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.usage: list[dict] = []

    async def complete(self, prompt: str) -> str:
        body: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        async with self._httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            raise_for_status(r)
        data = r.json()
        if data.get("usage"):
            self.usage.append(data["usage"])
        return data["choices"][0]["message"]["content"]
