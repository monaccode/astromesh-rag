"""Rank the four patterns, with and without a cross-encoder, over the Nimbus corpus.

    python examples/benchmark.py [--embedder hash|st] [--rerank both|on|off] [--k 5] [--out results/]

One ingestion feeds every run. Rankings are doc-level: chunk hits map to their doc id and are
deduplicated. recall/precision/hit rate use --k; MRR and nDCG use 10. LLM calls per query count
the `transform` stages in the trace. Latency is the wall time of one query() call, after
one untimed warmup query per pipeline (model loading would otherwise land in p95).
Writes results/<date>-<embedder>.json: config, corpus hash, summary, per-type, per-query rows.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # runnable via runpy too

from _common import (
    CORPUS,
    RERANK_MODEL,
    ingest,
    load_queries,
    make_embedder,
    make_llm,
    warn_if_offline,
)

from astromesh_rag.metrics import (
    dedupe,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from astromesh_rag.patterns import hybrid, hyde, multi_query, naive

DEPTH = 10  # MRR / nDCG cutoff
FETCH = 20  # chunk hits requested per query, so dedupe still leaves >= DEPTH docs


def p(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def summarize(rows: list[dict], k: int) -> dict:
    n = len(rows)
    lat = [r["ms"] for r in rows]
    return {
        f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n,
        "mrr@10": sum(r["mrr@10"] for r in rows) / n,
        "ndcg@10": sum(r["ndcg@10"] for r in rows) / n,
        f"hit_rate@{k}": sum(r[f"hit_rate@{k}"] for r in rows) / n,
        "p50_ms": p(lat, 0.5),
        "p95_ms": p(lat, 0.95),
        "llm_calls_per_query": sum(r["llm_calls"] for r in rows) / n,
    }


def table(rows: list[dict], keys: list[str]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in keys) + " |" for r in rows]
    return "\n".join(lines)


async def main(args) -> Path:
    embedder, emb_label = make_embedder(args.embedder)
    llm, llm_label, host = make_llm()
    warn_if_offline(emb_label, llm_label)
    index, bm25 = await ingest(embedder)
    queries = load_queries()

    reranker = None
    if args.rerank in ("on", "both"):
        from astromesh_rag.rerankers import CrossEncoderReranker

        reranker = CrossEncoderReranker(RERANK_MODEL)
    settings = {"off": [False], "on": [True], "both": [False, True]}[args.rerank]

    def build(name, rr):
        common = {"reranker": reranker if rr else None}
        if name == "naive":
            return naive(embedder, index, **common)
        if name == "hybrid":
            return hybrid(embedder, index, bm25, **common)
        if name == "multi_query":
            return multi_query(embedder, index, llm, **common)
        return hyde(embedder, index, llm, **common)

    k = args.k
    per_query = []
    for name in ("naive", "hybrid", "multi_query", "hyde"):
        for rr in settings:
            pipeline = build(name, rr)
            await pipeline.query(queries[0]["query"], k=FETCH)  # untimed warmup (model load)
            for q in queries:
                t0 = time.perf_counter()
                result = await pipeline.query(q["query"], k=FETCH)
                ms = (time.perf_counter() - t0) * 1000
                ranked = dedupe(h.chunk.doc_id for h in result.hits)
                rel = set(q["relevant"])
                per_query.append(
                    {
                        "pattern": name,
                        "rerank": rr,
                        "query_id": q["id"],
                        "type": q["type"],
                        "ranked": ranked[:DEPTH],
                        f"recall@{k}": recall_at_k(ranked, rel, k),
                        f"precision@{k}": precision_at_k(ranked, rel, k),
                        f"hit_rate@{k}": hit_rate_at_k(ranked, rel, k),
                        "mrr@10": mrr(ranked, rel, DEPTH),
                        "ndcg@10": ndcg_at_k(ranked, rel, DEPTH),
                        "ms": ms,
                        "llm_calls": sum(t["stage"] == "transform" for t in result.trace),
                    }
                )

    def groups(by):
        out: dict[tuple, list[dict]] = {}
        for r in per_query:
            out.setdefault(tuple(r[b] for b in by), []).append(r)
        return out

    summary = [
        {"pattern": pat, "rerank": rr, **summarize(rows, k)}
        for (pat, rr), rows in groups(["pattern", "rerank"]).items()
    ]
    by_type = [
        {"pattern": pat, "rerank": rr, "type": t, **summarize(rows, k)}
        for (pat, rr, t), rows in groups(["pattern", "rerank", "type"]).items()
    ]

    cols = ["pattern", "rerank", f"recall@{k}", "mrr@10", "ndcg@10", f"hit_rate@{k}"]
    print("\n## Overall\n")
    print(table(summary, [*cols, "p50_ms", "p95_ms", "llm_calls_per_query"]))
    print("\n## By query type\n")
    print(table(sorted(by_type, key=lambda r: r["type"]), ["type", *cols]))

    now = datetime.now(timezone.utc)
    data = {
        "config": {
            "embedder": emb_label,
            "reranker": RERANK_MODEL if reranker else None,
            "llm": llm_label,
            "base_url_host": host,
            "k": k,
        },
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "date": now.isoformat(timespec="seconds"),
        "summary": summary,
        "by_type": by_type,
        "per_query": per_query,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{now.date().isoformat()}-{args.embedder}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return path


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embedder", choices=["hash", "st"], default="hash")
    ap.add_argument("--rerank", choices=["both", "on", "off"], default="both")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results")
    return ap.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse()))
