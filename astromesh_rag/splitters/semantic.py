from __future__ import annotations

from ..stages import Embedder
from ..types import Chunk, Document, chunk_id
from .sentence import SentenceSplitter


class SemanticSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        similarity_threshold: float = 0.5,
        embedder: Embedder | None = None,
    ):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder

    async def split(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []

        if self.embedder is None:
            # Fall back to sentence splitting (strategy="sentence" makes it visible)
            return await SentenceSplitter(chunk_size=self.chunk_size).split(doc)

        sentences = SentenceSplitter._split_sentences(doc.text)
        if not sentences:
            return []

        embeddings = await self.embedder.embed(sentences)

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim >= self.similarity_threshold and (
                sum(len(s) for s in groups[-1]) + len(sentences[i]) <= self.chunk_size
            ):
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        doc_id = doc.resolved_id()
        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            content = "".join(group).strip()
            if content:
                chunks.append(
                    Chunk(
                        chunk_id(doc_id, idx),
                        doc_id,
                        content,
                        {**doc.metadata, "chunk_index": idx, "strategy": "semantic"},
                    )
                )

        return chunks

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
