"""
File: retrieval/fusion.py

Description:
    Reciprocal rank fusion (RRF) of several ranked lists.

    RRF uses only ranks, so the dense cosine scores and BM25 scores never need calibrating
    against each other: score(d) = sum over lists of 1 / (k + rank(d)), rank starting at 1.

Flow:
    retrieval.retriever.HybridRetriever.retrieve()
        -> reciprocal_rank_fusion([dense ids, bm25 ids], k=settings.retrieval.rrf_k)

Libraries used:
    collections.abc - standard library; Sequence type hint

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Function name: reciprocal_rank_fusion
    Purpose: Fuse ranked id lists with RRF. A duplicate id inside one list counts only once,
             and ties keep the order of first appearance.
    Parameters:
        rankings (Sequence[Sequence[str]]): ranked id lists, best first
        k (int): RRF constant; must be positive
    Returns:
        list[tuple[str, float]]: (id, fused score) pairs, best first
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # a duplicate inside one list must not count twice
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, so equal scores keep insertion (first-seen) order.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
