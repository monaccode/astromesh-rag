"""In-memory BM25 keyword index with JSON persistence."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from ..types import Chunk, Hit


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents; "err-4012" yields "err", "4012" and "err-4012"."""
    t = "".join(
        c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)
    )
    return re.findall(r"\w+", t) + re.findall(r"\w+(?:-\w+)+", t)


class BM25Index:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, path: str | None = None):
        self.k1, self.b, self.path = k1, b, path
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter] = {}

    async def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c
            self._tf[c.id] = Counter(tokenize(c.text))

    async def delete_document(self, doc_id: str) -> None:
        for cid in [i for i, c in self._chunks.items() if c.doc_id == doc_id]:
            del self._chunks[cid], self._tf[cid]

    async def retrieve(self, query: str, k: int) -> list[Hit]:
        n = len(self._chunks)
        if not n:
            return []
        lens = {i: sum(tf.values()) for i, tf in self._tf.items()}
        avg = sum(lens.values()) / n or 1.0
        terms = set(tokenize(query))
        idf = {}
        for t in terms:
            df = sum(1 for tf in self._tf.values() if t in tf)
            idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        scored = []
        for cid, tf in self._tf.items():
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += (
                        idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * lens[cid] / avg))
                    )
            if s > 0:
                scored.append((s, cid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Hit(self._chunks[cid], s, "bm25", {"bm25": s}) for s, cid in scored[:k]]

    def save(self) -> None:
        if self.path is None:
            raise ValueError("BM25Index has no path; pass path= to save")
        data = {
            "k1": self.k1,
            "b": self.b,
            "chunks": [
                {"id": c.id, "doc_id": c.doc_id, "text": c.text, "metadata": c.metadata}
                for c in self._chunks.values()
            ],
        }
        tmp = Path(f"{self.path}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str) -> BM25Index:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(data["k1"], data["b"], path)
        for d in data["chunks"]:
            c = Chunk(d["id"], d["doc_id"], d["text"], d["metadata"])
            idx._chunks[c.id] = c
            idx._tf[c.id] = Counter(tokenize(c.text))
        return idx
