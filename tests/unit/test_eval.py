"""Evaluation helpers: metrics, relevance matching, gate, golden set sanity."""

import numpy as np
import pytest

from common.types import Chunk
from eval.dataset import Question, is_relevant, load_questions
from eval.gate import check
from eval.metrics import cosine, parse_numbered_verdicts, recall_at_k, reciprocal_rank


def test_recall_and_rr():
    rel = [False, False, True, False]
    assert recall_at_k(rel, 2) == 0.0 and recall_at_k(rel, 5) == 1.0
    assert reciprocal_rank(rel) == pytest.approx(1 / 3)
    assert reciprocal_rank(rel, k=2) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_relevance_needs_source_page_and_phrase():
    q = Question("x", "q", True, relevant=[{"source": "a.pdf", "page": 2, "contains": "late  FEE"}])
    assert is_relevant(Chunk("1", "The late fee is 30.", "a.pdf", 2), q)
    assert not is_relevant(Chunk("2", "The late fee is 30.", "a.pdf", 3), q)
    assert not is_relevant(Chunk("3", "Wires.", "a.pdf", 2), q)
    q2 = Question("y", "q", True, relevant=[{"source": "b.md", "page": None, "contains": "wire"}])
    assert is_relevant(Chunk("4", "Wire costs", "b.md", None), q2)


def test_numbered_verdicts_and_cosine():
    assert parse_numbered_verdicts("1: YES\n2: no\n2: YES\n5: YES", 3) == {1: True, 2: False}
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_gate():
    assert check({"recall_at_5": 0.9}, {"recall_at_5": 0.9}, "recall_at_5", 0.0)[0]
    assert not check({"recall_at_5": 0.8}, {"recall_at_5": 0.9}, "recall_at_5", 0.0)[0]
    assert check({"recall_at_5": 0.85}, {"recall_at_5": 0.9}, "recall_at_5", 0.06)[0]
    assert not check({}, {"recall_at_5": 0.9}, "recall_at_5", 0.0)[0]


def test_golden_set_shape():
    qs = load_questions()
    assert len(qs) >= 30
    assert sum(not q.answerable for q in qs) == 5
    assert len({q.id for q in qs}) == len(qs)
    assert all(q.relevant for q in qs if q.answerable)
