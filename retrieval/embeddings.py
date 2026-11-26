"""
File: retrieval/embeddings.py

Description:
    Dense text encoder (sentence-transformers, BAAI/bge-small-en-v1.5 by default).

    Vectors are L2-normalised so inner product equals cosine similarity (FAISS IndexFlatIP and
    Pinecone's cosine metric then agree). The model downloads once into models/hf.

Flow:
    ingest.build.build_index()      -> get_encoder(settings).embed_documents(chunk texts)
    retrieval.retriever (dense leg) -> get_encoder(settings).embed_query(question)
    eval answer relevancy           -> embed_documents() for cosine similarity

Libraries used:
    functools - standard library; lru_cache to keep one encoder per model in the process
    typing - standard library; Protocol and the TYPE_CHECKING guard
    numpy - third party; embedding arrays
    sentence_transformers - third party; SentenceTransformer model
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

import numpy as np

if TYPE_CHECKING:
    from configs import Settings


class Encoder(Protocol):
    """
    Class name: Encoder
    Purpose: What the retriever needs from an encoder; tests pass a fake.
    """

    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Function name: Encoder.embed_documents
        Purpose: Encode passages into normalised vectors.
        Parameters:
            texts (list[str]): passages to encode
        Returns:
            np.ndarray: one row per passage
        """
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """
        Function name: Encoder.embed_query
        Purpose: Encode one search query into a normalised vector.
        Parameters:
            text (str): the query
        Returns:
            np.ndarray: the query vector
        """
        ...


class SentenceTransformerEncoder:
    """
    Class name: SentenceTransformerEncoder
    Purpose: bge-style encoder: instruction prefix on queries only, normalised float32 output.
    """

    def __init__(self, model_name: str, cache_dir: str, query_instruction: str, batch: int = 32):
        """
        Function name: SentenceTransformerEncoder.__init__
        Purpose: Load the sentence-transformers model on CPU and record its vector dimension.
        Parameters:
            model_name (str): model name or path
            cache_dir (str): folder where the model files are downloaded and cached
            query_instruction (str): prefix added to queries before encoding
            batch (int): encoding batch size
        Returns:
            None
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu", cache_folder=cache_dir)
        self._instruction = query_instruction
        self._batch = batch
        self.dim = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Function name: SentenceTransformerEncoder.embed_documents
        Purpose: Encode passages without the query instruction.
        Parameters:
            texts (list[str]): passages to encode
        Returns:
            np.ndarray: float32 array of L2-normalised vectors, one row per passage
        """
        vecs = self._model.encode(
            texts, batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Function name: SentenceTransformerEncoder.embed_query
        Purpose: Encode one search query with the retrieval instruction as a prefix.
        Parameters:
            text (str): the query
        Returns:
            np.ndarray: float32 L2-normalised query vector
        """
        return self.embed_documents([self._instruction + text])[0]


@lru_cache(maxsize=2)
def _cached(model: str, cache_dir: str, instruction: str, batch: int) -> SentenceTransformerEncoder:
    """
    Function name: _cached
    Purpose: Create an encoder once per argument set and reuse it (up to two cached models).
    Parameters:
        model (str): model name or path
        cache_dir (str): model download and cache folder
        instruction (str): query instruction prefix
        batch (int): encoding batch size
    Returns:
        SentenceTransformerEncoder: the cached encoder
    """
    return SentenceTransformerEncoder(model, cache_dir, instruction, batch)


def get_encoder(settings: Settings) -> SentenceTransformerEncoder:
    """
    Function name: get_encoder
    Purpose: Return the process-wide encoder for the configured model, cached under the
             models folder in "hf".
    Parameters:
        settings (Settings): application settings (embedding and paths sections)
    Returns:
        SentenceTransformerEncoder: the shared encoder
    """
    e = settings.embedding
    cache = str(settings.paths.resolve(settings.paths.models_dir) / "hf")
    return _cached(e.model, cache, e.query_instruction, e.batch_size)
