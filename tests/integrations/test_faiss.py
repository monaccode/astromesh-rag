import pytest
from conftest import index_contract

pytest.importorskip("faiss")

from astromesh_rag.indexes.faiss import FaissIndex


async def test_faiss_contract():
    await index_contract(FaissIndex(dimension=3))
