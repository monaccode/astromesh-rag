import json
import runpy
import sys
from pathlib import Path

from astromesh_rag.spec import build, load_spec_file

C = Path(__file__).parents[1] / "examples" / "corpus"


def test_corpus_regenerates_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py", str(tmp_path / "c.jsonl")])
    runpy.run_path(str(C / "generate.py"), run_name="__main__")
    assert (tmp_path / "c.jsonl").read_bytes() == (C / "corpus.jsonl").read_bytes()


def test_queries_reference_real_docs_and_cover_types():
    ids = {json.loads(line)["id"] for line in (C / "corpus.jsonl").read_text().splitlines()}
    qs = [json.loads(line) for line in (C / "queries.jsonl").read_text().splitlines()]
    assert len(qs) >= 60
    assert {q["type"] for q in qs} == {"exact", "paraphrase", "distractor", "noise"}
    assert all(set(q["relevant"]) <= ids and q["relevant"] for q in qs)


def test_distractor_queries_point_only_at_current_policies():
    docs = {d["id"]: d for d in map(json.loads, (C / "corpus.jsonl").read_text().splitlines())}
    qs = [json.loads(line) for line in (C / "queries.jsonl").read_text().splitlines()]
    for q in qs:
        if q["type"] == "distractor":
            assert all(docs[r]["metadata"].get("version") == "2026" for r in q["relevant"]), q


def test_example_yaml_builds_with_test_embedder():
    spec = load_spec_file(C.parent / "pipeline.rag.yaml")
    assert spec.retrieval["pattern"] == "hybrid"
    spec.embeddings = {"provider": "hash"}
    assert build(spec).query_pipeline.retrievers[1].name == "bm25"
