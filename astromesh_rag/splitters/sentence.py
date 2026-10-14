from __future__ import annotations

import re

from ..types import Chunk, Document, chunk_id


class SentenceSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def split(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []
        doc_id = doc.resolved_id()
        sentences = self._split_sentences(doc.text)
        chunks: list[Chunk] = []
        current = ""
        i = 0

        def emit(content: str) -> None:
            chunks.append(
                Chunk(
                    chunk_id(doc_id, i),
                    doc_id,
                    content.strip(),
                    {**doc.metadata, "chunk_index": i, "strategy": "sentence"},
                )
            )

        for sentence in sentences:
            if current and len(current) + len(sentence) > self.chunk_size:
                emit(current)
                i += 1
                current = sentence
            else:
                current += sentence

        if current.strip():
            emit(current)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text on sentence boundaries, keeping delimiters attached."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        # Re-add trailing space for all but the last
        sentences: list[str] = []
        for idx, part in enumerate(parts):
            if idx < len(parts) - 1:
                sentences.append(part + " ")
            else:
                sentences.append(part)
        return sentences
