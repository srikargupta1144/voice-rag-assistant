"""Dense text encoder (sentence-transformers, BAAI/bge-small-en-v1.5 by default).

Flow:
    ingest.build.build_index()      -> get_encoder(settings).embed_documents(chunk texts)
    retrieval.retriever (dense leg) -> get_encoder(settings).embed_query(question)
    eval answer relevancy           -> embed_documents() for cosine similarity

Vectors are L2-normalised so inner product equals cosine similarity (FAISS IndexFlatIP and
Pinecone's cosine metric then agree). The model downloads once into models/hf.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from configs import Settings


class Encoder(Protocol):
    """What the retriever needs from an encoder; tests pass a fake."""

    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """bge-style encoder: instruction prefix on queries only, normalised float32 output."""

    def __init__(self, model_name: str, cache_dir: str, query_instruction: str, batch: int = 32):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu", cache_folder=cache_dir)
        self._instruction = query_instruction
        self._batch = batch
        self.dim = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode passages (no instruction)."""
        vecs = self._model.encode(
            texts, batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode one search query with the retrieval instruction."""
        return self.embed_documents([self._instruction + text])[0]


@lru_cache(maxsize=2)
def _cached(model: str, cache_dir: str, instruction: str, batch: int) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(model, cache_dir, instruction, batch)


def get_encoder(settings: Settings) -> SentenceTransformerEncoder:
    """Process-wide encoder for the configured model."""
    e = settings.embedding
    cache = str(settings.paths.resolve(settings.paths.models_dir) / "hf")
    return _cached(e.model, cache, e.query_instruction, e.batch_size)
