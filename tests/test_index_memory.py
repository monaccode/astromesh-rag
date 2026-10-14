from conftest import index_contract

from astromesh_rag.indexes.memory import MemoryIndex


async def test_memory_contract():
    await index_contract(MemoryIndex())


async def test_search_empty_index_returns_empty():
    assert await MemoryIndex().search([1, 0, 0], 3) == []
