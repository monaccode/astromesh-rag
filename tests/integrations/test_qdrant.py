import os
import uuid

import pytest
from conftest import collision_contract, index_contract, require_env

pytest.importorskip("qdrant_client")

from astromesh_rag.indexes.qdrant import QdrantIndex


async def test_qdrant_contract():
    url = (
        require_env("QDRANT_URL")
        if os.environ.get("RAG_REQUIRE_INTEGRATION") == "1"
        else os.environ.get("QDRANT_URL", ":memory:")
    )
    await index_contract(QdrantIndex(collection=f"t{uuid.uuid4().hex}", url=url, dimension=3))


async def test_qdrant_metadata_cannot_clobber_fields():
    await collision_contract(
        QdrantIndex(collection=f"c{uuid.uuid4().hex}", url=":memory:", dimension=3)
    )
