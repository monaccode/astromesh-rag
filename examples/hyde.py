"""hyde: the LLM writes a hypothetical answer; that passage is embedded and searched."""

import asyncio

from _common import run_example

from astromesh_rag.patterns import hyde

if __name__ == "__main__":
    asyncio.run(run_example(lambda emb, index, bm25, llm: hyde(emb, index, llm)))
