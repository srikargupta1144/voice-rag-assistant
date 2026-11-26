"""Reciprocal rank fusion (RRF) of several ranked lists.

Flow:
    retrieval.retriever.HybridRetriever.retrieve()
        -> reciprocal_rank_fusion([dense ids, bm25 ids], k=settings.retrieval.rrf_k)

RRF uses only ranks, so the dense cosine scores and BM25 scores never need calibrating
against each other: score(d) = sum over lists of 1 / (k + rank(d)), rank starting at 1.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists; ties keep the order of first appearance."""
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
