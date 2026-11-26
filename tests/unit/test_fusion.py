"""Reciprocal rank fusion."""

import pytest

from retrieval.fusion import reciprocal_rank_fusion


def test_items_in_both_lists_win():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "a"  # ranks 1 and 2
    assert ids[1] == "c"  # ranks 3 and 1
    assert set(ids) == {"a", "b", "c", "d"}


def test_scores_follow_formula():
    fused = dict(reciprocal_rank_fusion([["a"], ["b", "a"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11 + 1 / 12)
    assert fused["b"] == pytest.approx(1 / 11)


def test_duplicates_in_one_list_count_once_and_ties_keep_order():
    fused = reciprocal_rank_fusion([["x", "x"], ["y"]], k=60)
    assert dict(fused)["x"] == pytest.approx(1 / 61)
    assert [i for i, _ in fused] == ["x", "y"]


def test_empty_and_invalid_k():
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
