from __future__ import annotations

from ..types import Chunk, Document, chunk_id


class FixedSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
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
