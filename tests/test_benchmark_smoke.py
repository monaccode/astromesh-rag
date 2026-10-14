import json
import runpy
import sys
from pathlib import Path


def test_benchmark_runs_offline(tmp_path, monkeypatch):
    for v in ("RAG_BENCH_BASE_URL", "RAG_BENCH_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    argv = ["benchmark.py", "--embedder", "hash", "--rerank", "off", "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "benchmark.py"), run_name="__main__"
    )
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert {r["pattern"] for r in data["summary"]} == {"naive", "hybrid", "multi_query", "hyde"}
    assert "api_key" not in json.dumps(data).lower()
    assert len(data["per_query"]) == 4 * 60
    keys = {"recall@5", "precision@5", "hit_rate@5", "mrr@10", "ndcg@10"}
    keys |= {"p50_ms", "p95_ms", "llm_calls_per_query"}
    assert all(keys <= row.keys() for row in data["summary"])
    assert {r["type"] for r in data["by_type"]} == {"exact", "paraphrase", "distractor", "noise"}
