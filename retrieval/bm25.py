"""
File: retrieval/bm25.py

Description:
    Keyword (sparse) retrieval with Okapi BM25 over the chunk texts.

    BM25 catches exact words the dense model blurs: fee names, form numbers (HE-12), amounts.

Flow:
    retrieval.retriever.load_retriever() -> BM25Index(chunk texts)   # rebuilt at load, < 1 s
    retrieval.retriever.HybridRetriever  -> BM25Index.search(query, k)

Libraries used:
    re - standard library; word token pattern
    rank_bm25 - third party; BM25Okapi scoring

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
    """
    Function name: bm25_tokens
    Purpose: Produce lower-cased word tokens without stop words; ids joined by hyphens or
             dots, such as he-12, are kept as one token.
    Parameters:
        text (str): text to tokenize
    Returns:
        list[str]: the tokens in order
    """
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """
    Class name: BM25Index
    Purpose: In-memory BM25 over a fixed list of documents (positions match the chunk list).
    """

    def __init__(self, texts: list[str]):
        """
        Function name: BM25Index.__init__
        Purpose: Build the BM25 model over the given texts. A text with no tokens is given a
                 placeholder token so that positions still match; an empty list builds no
                 model.
        Parameters:
            texts (list[str]): document texts, in chunk order
        Returns:
            None
        """
        self._bm25 = BM25Okapi([bm25_tokens(t) or ["_"] for t in texts]) if texts else None
        self.size = len(texts)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """
        Function name: BM25Index.search
        Purpose: Score every document against the query and return the top-k entries that
                 have a positive score, best first.
        Parameters:
            query (str): search text
            k (int): largest number of results to return
        Returns:
            list[tuple[int, float]]: (position, score) pairs; empty when the index is empty
                                     or the query has no tokens
        """
        if self._bm25 is None:
            return []
        tokens = bm25_tokens(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]
