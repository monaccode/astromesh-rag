from conftest import collision_contract, index_contract

from astromesh_rag.indexes.memory import MemoryIndex


async def test_memory_contract():
    await index_contract(MemoryIndex())


async def test_search_empty_index_returns_empty():
    assert await MemoryIndex().search([1, 0, 0], 3) == []


async def test_memory_metadata_cannot_clobber_fields():
    await collision_contract(MemoryIndex())
