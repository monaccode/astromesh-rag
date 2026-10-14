from astromesh_rag.types import Chunk, Document, Hit, QueryResult, chunk_id


def test_document_id_defaults_to_content_hash():
    a, b = Document("hello"), Document("hello")
    assert a.resolved_id() == b.resolved_id()
    assert a.resolved_id().startswith("sha256:")
    assert len(a.resolved_id()) == 7 + 16


def test_explicit_document_id_wins():
    assert Document("x", id="faq-1").resolved_id() == "faq-1"


def test_chunk_id_format():
    assert chunk_id("faq-1", 3) == "faq-1#3"


def test_to_list_matches_core_shape():
    c = Chunk(id="d#0", doc_id="d", text="t", metadata={"a": 1})
    r = QueryResult(query="q", hits=[Hit(c, 0.5, "vector")])
    assert r.to_list() == [
        {"doc_id": "d#0", "content": "t", "metadata": {"a": 1}, "score": 0.5, "source": "vector"}
    ]
