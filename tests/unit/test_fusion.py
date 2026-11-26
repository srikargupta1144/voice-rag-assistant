"""
File: tests/unit/test_fusion.py

Description:
    Unit tests for reciprocal rank fusion.

Libraries used:
    pytest - third party; approximate float comparison and error checks
    retrieval.fusion - project; the reciprocal rank fusion function under test

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import pytest

from retrieval.fusion import reciprocal_rank_fusion


def test_items_in_both_lists_win():
    """
    Function name: test_items_in_both_lists_win
    Purpose: Check that items ranked in both lists come first and that every item appears in
        the fused result.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "a"  # ranks 1 and 2
    assert ids[1] == "c"  # ranks 3 and 1
    assert set(ids) == {"a", "b", "c", "d"}


def test_scores_follow_formula():
    """
    Function name: test_scores_follow_formula
    Purpose: Check that each fused score is the sum of 1 / (k + rank) over the lists.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    fused = dict(reciprocal_rank_fusion([["a"], ["b", "a"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11 + 1 / 12)
    assert fused["b"] == pytest.approx(1 / 11)


def test_duplicates_in_one_list_count_once_and_ties_keep_order():
    """
    Function name: test_duplicates_in_one_list_count_once_and_ties_keep_order
    Purpose: Check that a duplicate item within one list is counted once and that tied
        scores keep their first-seen order.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    fused = reciprocal_rank_fusion([["x", "x"], ["y"]], k=60)
    assert dict(fused)["x"] == pytest.approx(1 / 61)
    assert [i for i, _ in fused] == ["x", "y"]


def test_empty_and_invalid_k():
    """
    Function name: test_empty_and_invalid_k
    Purpose: Check that empty lists give an empty result and that k = 0 raises ValueError.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
