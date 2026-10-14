"""hybrid: vector and BM25 in parallel, fused with RRF. BM25 catches exact codes (ERR-4012)."""

import asyncio

from _common import run_example

from astromesh_rag.patterns import hybrid

if __name__ == "__main__":
    asyncio.run(run_example(lambda emb, index, bm25, llm: hybrid(emb, index, bm25)))
