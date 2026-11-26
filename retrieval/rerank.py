"""
File: retrieval/rerank.py

Description:
    Cross-encoder re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2 by default).

    Scores are raw logits (no sigmoid): above 0 usually means relevant, strongly negative means
    unrelated. rag.pipeline compares the top logit with retrieval.not_found_threshold.

Flow:
    retrieval.retriever.HybridRetriever.retrieve(mode="full")
        -> get_reranker(settings).score(query, passages)   # one logit per passage

Libraries used:
    functools - standard library; lru_cache to keep one re-ranker per model in the process
    typing - standard library; Protocol and the TYPE_CHECKING guard
    torch - third party; Identity activation so the model returns raw logits
    sentence_transformers - third party; CrossEncoder model
    configs - project; Settings type (type checking only)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from configs import Settings


class Reranker(Protocol):
    """
    Class name: Reranker
    Purpose: What the retriever needs from a re-ranker; tests pass a fake.
    """

    def score(self, query: str, passages: list[str]) -> list[float]:
        """
        Function name: Reranker.score
        Purpose: Score how relevant each passage is to the query.
        Parameters:
            query (str): the search query
            passages (list[str]): passages to score
        Returns:
            list[float]: one relevance score per passage, in passage order
        """
        ...


class CrossEncoderReranker:
    """
    Class name: CrossEncoderReranker
    Purpose: sentence-transformers CrossEncoder on CPU.
    """

    def __init__(self, model_name: str, cache_dir: str):
        """
        Function name: CrossEncoderReranker.__init__
        Purpose: Load the cross-encoder on CPU with an identity activation, so that scores
                 stay raw logits; the not-found threshold is calibrated on them.
        Parameters:
            model_name (str): model name or path
            cache_dir (str): folder where the model files are downloaded and cached
        Returns:
            None
        """
        import torch
        from sentence_transformers import CrossEncoder

        # Identity activation keeps raw logits; the not-found threshold is calibrated on them.
        self._model = CrossEncoder(
            model_name, device="cpu", cache_folder=cache_dir, activation_fn=torch.nn.Identity()
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        """
        Function name: CrossEncoderReranker.score
        Purpose: Compute the relevance logit for each (query, passage) pair.
        Parameters:
            query (str): the search query
            passages (list[str]): passages to score
        Returns:
            list[float]: one logit per passage, in passage order; empty when there are none
        """
        if not passages:
            return []
        out = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(x) for x in out]


@lru_cache(maxsize=2)
def _cached(model: str, cache_dir: str) -> CrossEncoderReranker:
    """
    Function name: _cached
    Purpose: Create a re-ranker once per argument set and reuse it (up to two cached models).
    Parameters:
        model (str): model name or path
        cache_dir (str): model download and cache folder
    Returns:
        CrossEncoderReranker: the cached re-ranker
    """
    return CrossEncoderReranker(model, cache_dir)


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    """
    Function name: get_reranker
    Purpose: Return the process-wide re-ranker for the configured model, cached under the
             models folder in "hf".
    Parameters:
        settings (Settings): application settings (retrieval and paths sections)
    Returns:
        CrossEncoderReranker: the shared re-ranker
    """
    cache = str(settings.paths.resolve(settings.paths.models_dir) / "hf")
    return _cached(settings.retrieval.rerank_model, cache)
