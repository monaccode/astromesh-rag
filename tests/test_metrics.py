import math

import pytest

from astromesh_rag.metrics import dedupe, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

R = ["d3", "d1", "d7", "d2"]
REL = {"d1", "d2"}


def test_recall_precision():
    assert recall_at_k(R, REL, 2) == 0.5
    assert precision_at_k(R, REL, 2) == 0.5
    assert recall_at_k(R, REL, 4) == 1.0


def test_mrr_first_relevant_at_rank_2():
    assert mrr(R, REL) == 0.5
    assert mrr(R, {"zz"}) == 0.0


def test_ndcg():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(R, REL, 4) == pytest.approx(dcg / idcg)


def test_hit_rate_and_dedupe():
    assert hit_rate_at_k(R, REL, 1) == 0.0
    assert hit_rate_at_k(R, REL, 2) == 1.0
    assert dedupe(["a", "a", "b"]) == ["a", "b"]


def test_empty_relevant_rejected():
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(R, set(), 3)
