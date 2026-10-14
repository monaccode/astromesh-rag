"""multi_query: the LLM rewrites the question 3 ways; each variant is searched; RRF fuses."""

import asyncio

from _common import run_example

from astromesh_rag.patterns import multi_query

if __name__ == "__main__":
    asyncio.run(run_example(lambda emb, index, bm25, llm: multi_query(emb, index, llm)))
