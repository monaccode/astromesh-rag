"""naive: embed the question, take the nearest chunks. No keywords, no LLM."""

import asyncio

from _common import run_example

from astromesh_rag.patterns import naive

if __name__ == "__main__":
    asyncio.run(run_example(lambda emb, index, bm25, llm: naive(emb, index)))
