import uuid

import pytest
from conftest import index_contract

pytest.importorskip("chromadb")

from astromesh_rag.indexes.chroma import ChromaIndex


async def test_chroma_contract():
    await index_contract(ChromaIndex(collection=f"t{uuid.uuid4().hex}"))
