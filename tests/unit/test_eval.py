"""
File: tests/unit/test_eval.py

Description:
    Unit tests for the evaluation helpers: metrics, relevance matching, the gate and golden
    set sanity.

Libraries used:
    numpy - third party; vectors for the cosine check
    pytest - third party; approximate float comparison
    common.types - project; the Chunk type used in relevance checks
    eval.dataset - project; Question, relevance matching and the golden set loader under test
    eval.gate - project; the gate check under test
    eval.metrics - project; metric functions under test

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import numpy as np
import pytest

from common.types import Chunk
from eval.dataset import Question, is_relevant, load_questions
from eval.gate import check
from eval.metrics import cosine, parse_numbered_verdicts, recall_at_k, reciprocal_rank


def test_recall_and_rr():
    """
    Function name: test_recall_and_rr
    Purpose: Check Recall@k and reciprocal rank, including the k cut-off and an empty list.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    rel = [False, False, True, False]
    assert recall_at_k(rel, 2) == 0.0 and recall_at_k(rel, 5) == 1.0
    assert reciprocal_rank(rel) == pytest.approx(1 / 3)
    assert reciprocal_rank(rel, k=2) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_relevance_needs_source_page_and_phrase():
    """
    Function name: test_relevance_needs_source_page_and_phrase
    Purpose: Check that relevance needs a matching source, a matching page when the label
        names one, and the key phrase after case and whitespace normalisation.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    q = Question("x", "q", True, relevant=[{"source": "a.pdf", "page": 2, "contains": "late  FEE"}])
    assert is_relevant(Chunk("1", "The late fee is 30.", "a.pdf", 2), q)
    assert not is_relevant(Chunk("2", "The late fee is 30.", "a.pdf", 3), q)
    assert not is_relevant(Chunk("3", "Wires.", "a.pdf", 2), q)
    q2 = Question("y", "q", True, relevant=[{"source": "b.md", "page": None, "contains": "wire"}])
    assert is_relevant(Chunk("4", "Wire costs", "b.md", None), q2)


def test_numbered_verdicts_and_cosine():
    """
    Function name: test_numbered_verdicts_and_cosine
    Purpose: Check that verdict parsing keeps the first verdict per number, ignores numbers
        out of range and is case-insensitive, and that identical vectors have cosine 1.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert parse_numbered_verdicts("1: YES\n2: no\n2: YES\n5: YES", 3) == {1: True, 2: False}
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_gate():
    """
    Function name: test_gate
    Purpose: Check that the gate passes at the baseline, fails below it, passes within the
        tolerance and fails when the metric is missing from the results.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert check({"recall_at_5": 0.9}, {"recall_at_5": 0.9}, "recall_at_5", 0.0)[0]
    assert not check({"recall_at_5": 0.8}, {"recall_at_5": 0.9}, "recall_at_5", 0.0)[0]
    assert check({"recall_at_5": 0.85}, {"recall_at_5": 0.9}, "recall_at_5", 0.06)[0]
    assert not check({}, {"recall_at_5": 0.9}, "recall_at_5", 0.0)[0]


def test_golden_set_shape():
    """
    Function name: test_golden_set_shape
    Purpose: Check the golden set: at least 30 questions, exactly 5 unanswerable, unique ids,
        and relevance labels on every answerable question.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    qs = load_questions()
    assert len(qs) >= 30
    assert sum(not q.answerable for q in qs) == 5
    assert len({q.id for q in qs}) == len(qs)
    assert all(q.relevant for q in qs if q.answerable)
