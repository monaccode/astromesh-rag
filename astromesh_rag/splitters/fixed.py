from __future__ import annotations

from ..types import Chunk, Document, chunk_id


def check_window(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"chunk_size must be > 0 and 0 <= overlap < chunk_size, got chunk_size={chunk_size}, "
            f"overlap={overlap}"
        )


class FixedSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        check_window(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def split(self, doc: Document) -> list[Chunk]:
        document = doc.text
        if not document:
            return []
        doc_id = doc.resolved_id()
        chunks: list[Chunk] = []
        start = 0
        i = 0

        while start < len(document):
            end = start + self.chunk_size
            content = document[start:end]
            chunks.append(
                Chunk(
                    chunk_id(doc_id, i),
                    doc_id,
                    content,
                    {**doc.metadata, "chunk_index": i, "strategy": "fixed"},
                )
            )
            i += 1
            start += self.chunk_size - self.overlap
            if start >= len(document):
                break

        return chunks
