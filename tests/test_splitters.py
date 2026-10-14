from astromesh_rag.splitters import (
    FixedSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    SentenceSplitter,
)
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Document


async def test_recursive_ids_and_metadata():
    doc = Document("a" * 1200, id="d", metadata={"src": "x"})
    chunks = await RecursiveSplitter(chunk_size=500, overlap=0).split(doc)
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "d" and c.metadata["src"] == "x" for c in chunks)
    assert chunks[0].metadata["strategy"] == "recursive"


async def test_empty_document_gives_no_chunks():
    for s in (FixedSplitter(), RecursiveSplitter(), SentenceSplitter(), SemanticSplitter()):
        assert await s.split(Document("", id="d")) == []


async def test_semantic_without_embedder_says_so():
    chunks = await SemanticSplitter().split(Document("One. Two. Three.", id="d"))
    assert chunks
    assert all(c.metadata["strategy"] == "sentence" for c in chunks)


async def test_semantic_groups_similar_sentences():
    text = "Cats purr softly. Cats purr loudly. Rockets need fuel."
    chunks = await SemanticSplitter(similarity_threshold=0.3, embedder=HashEmbedder()).split(
        Document(text, id="d")
    )
    assert len(chunks) == 2
    assert "Rockets" in chunks[1].text
