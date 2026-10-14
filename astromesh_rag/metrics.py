"""Retrieval metrics over ranked doc ids. Pure functions, stdlib only.

Callers dedupe chunk ids to doc ids first (see ``dedupe``); these functions do not.
"""

import math
from collections.abc import Iterable


def dedupe(ranked: Iterable[str]) -> list[str]:
    """Keep the first occurrence of each id, preserving rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ranked:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _check(relevant: set[str], k: int | None) -> None:
    if not relevant:
        raise ValueError("relevant must not be empty")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")


def _hits_at(ranked: list[str], relevant: set[str], k: int) -> int:
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    return _hits_at(ranked, relevant, k) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    return _hits_at(ranked, relevant, k) / k


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    return 1.0 if _hits_at(ranked, relevant, k) else 0.0


def mrr(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    _check(relevant, k)
    window = ranked if k is None else ranked[:k]
    for rank, doc_id in enumerate(window, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """nDCG with binary gains: DCG = sum(1/log2(rank+1)) over relevant hits in top k."""
    _check(relevant, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
