"""Pure metric functions (unit-tested; no models).

Flow:
    eval.retrieval_eval -> recall_at_k(), reciprocal_rank()
    eval.answer_eval    -> parse_numbered_verdicts(), cosine()
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np


def recall_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 when any of the first k results is relevant (hit rate per question)."""
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool], k: int | None = None) -> float:
    """1 / rank of the first relevant result (0 when none within k)."""
    for i, rel in enumerate(relevance[:k] if k else relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def mean(values: Sequence[float]) -> float:
    return round(float(np.mean(values)), 4) if len(values) else 0.0


_NUMBERED = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)


def parse_numbered_verdicts(text: str, n: int) -> dict[int, bool]:
    """Lines like "1: YES" / "2: NO" -> {1: True, 2: False}; numbers outside 1..n ignored."""
    out: dict[int, bool] = {}
    for m in _NUMBERED.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= n and i not in out:
            out[i] = m.group(2).upper() == "YES"
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)
