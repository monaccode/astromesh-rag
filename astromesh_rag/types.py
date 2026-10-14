"""Data model shared by every stage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    id: str | None = None
    metadata: dict = field(default_factory=dict)

    def resolved_id(self) -> str:
        return self.id or "sha256:" + hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str
    details: dict = field(
        default_factory=dict
    )  # {"vector": .., "bm25": .., "rrf": .., "rerank": ..}


@dataclass
class QueryResult:
    query: str
    hits: list[Hit] = field(default_factory=list)
    context: str = ""
    answer: str | None = None
    trace: list[dict] = field(default_factory=list)

    def to_list(self) -> list[dict]:
        return [
            {
                "doc_id": h.chunk.id,
                "content": h.chunk.text,
                "metadata": h.chunk.metadata,
                "score": h.score,
                "source": h.source,
            }
            for h in self.hits
        ]


def chunk_id(doc_id: str, n: int) -> str:
    return f"{doc_id}#{n}"
