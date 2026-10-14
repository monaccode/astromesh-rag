from astromesh_rag.types import Chunk


def chunks(doc, texts, **meta):
    return [
        Chunk(id=f"{doc}#{i}", doc_id=doc, text=t, metadata=dict(meta)) for i, t in enumerate(texts)
    ]


async def index_contract(index):
    await index.upsert(chunks("a", ["x", "y"], lang="es"), [[1, 0, 0], [0, 1, 0]])
    await index.upsert(chunks("b", ["z"], lang="en"), [[0, 0, 1]])
    hits = await index.search([1, 0.1, 0], 2)
    assert hits[0].chunk.id == "a#0"
    assert hits[0].source == "vector"
    assert hits[0].score > hits[1].score
    assert "vector" in hits[0].details
    assert [h.chunk.id for h in await index.search([0, 0, 1], 5, {"lang": "es"})] in (
        ["a#1", "a#0"],
        ["a#0", "a#1"],
    )
    await index.delete_document("a")
    assert {h.chunk.doc_id for h in await index.search([1, 1, 1], 5)} == {"b"}
