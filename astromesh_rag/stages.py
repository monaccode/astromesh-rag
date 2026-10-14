"""Stage contracts. Everything is async except Fusion.fuse."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Chunk, Document, Hit


@runtime_checkable
class Splitter(Protocol):
    async def split(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    dimension: int | None

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorIndex(Protocol):
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def search(
        self, vector: list[float], k: int, filters: dict | None = None
    ) -> list[Hit]: ...

    async def delete_document(self, doc_id: str) -> None: ...


@runtime_checkable
class KeywordIndex(Protocol):
    async def add(self, chunks: list[Chunk]) -> None: ...

    async def delete_document(self, doc_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    async def retrieve(self, query: str, k: int) -> list[Hit]: ...


@runtime_checkable
class QueryTransform(Protocol):
    async def transform(self, query: str) -> list[str]: ...


@runtime_checkable
class Fusion(Protocol):
    def fuse(self, lists: list[list[Hit]], k: int) -> list[Hit]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


@runtime_checkable
class LLM(Protocol):
    async def complete(self, prompt: str) -> str: ...
