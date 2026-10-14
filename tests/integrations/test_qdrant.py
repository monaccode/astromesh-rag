import os
import uuid

import pytest
from conftest import index_contract

pytest.importorskip("qdrant_client")

from astromesh_rag.indexes.qdrant import QdrantIndex


async def test_qdrant_contract():
    url = os.environ.get("QDRANT_URL", ":memory:")
    await index_contract(QdrantIndex(collection=f"t{uuid.uuid4().hex}", url=url, dimension=3))
