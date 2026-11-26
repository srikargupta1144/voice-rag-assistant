"""
File: eval/metrics.py

Description:
    Pure metric functions (unit-tested; no models).

Flow:
    eval.retrieval_eval -> recall_at_k(), reciprocal_rank()
    eval.answer_eval    -> parse_numbered_verdicts(), cosine()

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    re - standard library; matches numbered YES / NO verdict lines from the judge
    collections.abc - standard library; the Sequence type for metric inputs
    numpy - third party; mean, vector norm and dot product calculations

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np


def recall_at_k(relevance: Sequence[bool], k: int) -> float:
    """
    Function name: recall_at_k
    Purpose: Hit rate for one question: 1.0 when any of the first k results is relevant.
    Parameters:
        relevance (Sequence[bool]): Relevance flag of each result, in rank order.
        k (int): Number of top results to consider.
    Returns:
        float: 1.0 when a relevant result is within the first k, otherwise 0.0.
    """
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool], k: int | None = None) -> float:
    """
    Function name: reciprocal_rank
    Purpose: Return 1 / rank of the first relevant result, or 0 when there is none within k.
    Parameters:
        relevance (Sequence[bool]): Relevance flag of each result, in rank order.
        k (int | None): Number of top results to consider; None or 0 considers all results.
    Returns:
        float: The reciprocal rank of the first relevant result, or 0.0 when none is found.
    """
    for i, rel in enumerate(relevance[:k] if k else relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def mean(values: Sequence[float]) -> float:
    """
    Function name: mean
    Purpose: Arithmetic mean rounded to four decimal places, with 0.0 for an empty input.
    Parameters:
        values (Sequence[float]): Values to average.
    Returns:
        float: The rounded mean, or 0.0 when values is empty.
    """
    return round(float(np.mean(values)), 4) if len(values) else 0.0


_NUMBERED = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)


def parse_numbered_verdicts(text: str, n: int) -> dict[int, bool]:
    """
    Function name: parse_numbered_verdicts
    Purpose: Parse judge lines such as "1: YES" and "2: NO" into {1: True, 2: False}. Numbers
        outside 1..n are ignored and only the first verdict for each number is kept.
    Parameters:
        text (str): Judge output text.
        n (int): Number of statements that were judged.
    Returns:
        dict[int, bool]: Statement number mapped to True for YES and False for NO.
    """
    out: dict[int, bool] = {}
    for m in _NUMBERED.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= n and i not in out:
            out[i] = m.group(2).upper() == "YES"
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Function name: cosine
    Purpose: Cosine similarity of two vectors; a zero norm product is treated as 1.0 to avoid
        division by zero.
    Parameters:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.
    Returns:
        float: The cosine similarity of a and b.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)
