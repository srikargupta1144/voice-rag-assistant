"""Cross-encoder re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2 by default).

Flow:
    retrieval.retriever.HybridRetriever.retrieve(mode="full")
        -> get_reranker(settings).score(query, passages)   # one logit per passage

Scores are raw logits (no sigmoid): above 0 usually means relevant, strongly negative means
unrelated. rag.pipeline compares the top logit with retrieval.not_found_threshold.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from configs import Settings


class Reranker(Protocol):
    """What the retriever needs from a re-ranker; tests pass a fake."""

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder on CPU."""

    def __init__(self, model_name: str, cache_dir: str):
        import torch
        from sentence_transformers import CrossEncoder

        # Identity activation keeps raw logits; the not-found threshold is calibrated on them.
        self._model = CrossEncoder(
            model_name, device="cpu", cache_folder=cache_dir, activation_fn=torch.nn.Identity()
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logit for each (query, passage) pair."""
        if not passages:
            return []
        out = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(x) for x in out]


@lru_cache(maxsize=2)
def _cached(model: str, cache_dir: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model, cache_dir)


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    """Process-wide re-ranker for the configured model."""
    cache = str(settings.paths.resolve(settings.paths.models_dir) / "hf")
    return _cached(settings.retrieval.rerank_model, cache)
