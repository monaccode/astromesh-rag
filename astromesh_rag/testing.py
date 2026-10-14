"""Deterministic test doubles."""

from __future__ import annotations

import hashlib
import re


class HashEmbedder:
    """Bag-of-words hashing embedder, L2-normalized. Shared words give high cosine."""

    dimension = 64

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dimension
            for tok in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self.dimension] += 1
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / norm for x in vec] if norm else vec)
        return out


class FakeLLM:
    def __init__(self, replies: dict[str, str] | None = None, default: str = ""):
        self.replies = replies or {}
        self.default = default
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, reply in self.replies.items():
            if key in prompt:
                return reply
        return self.default
