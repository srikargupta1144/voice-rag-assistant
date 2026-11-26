"""Keyword (sparse) retrieval with Okapi BM25 over the chunk texts.

Flow:
    retrieval.retriever.load_retriever() -> BM25Index(chunk texts)   # rebuilt at load, < 1 s
    retrieval.retriever.HybridRetriever  -> BM25Index.search(query, k)

BM25 catches exact words the dense model blurs: fee names, form numbers (HE-12), amounts.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
# Small stop list: function words only, so amounts and form ids always survive.
STOPWORDS = frozenset(
    (
        "a an and are as at be by can do does for from how i if in is it its my of on or "
        "the this to was what when where which who will with you your me there"
    ).split()
)


def bm25_tokens(text: str) -> list[str]:
    """Lower-cased word tokens without stop words; keeps hyphenated ids like he-12."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """In-memory BM25 over a fixed list of documents (positions match the chunk list)."""

    def __init__(self, texts: list[str]):
        self._bm25 = BM25Okapi([bm25_tokens(t) or ["_"] for t in texts]) if texts else None
        self.size = len(texts)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (position, score) pairs with a positive score, best first."""
        if self._bm25 is None:
            return []
        tokens = bm25_tokens(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]
