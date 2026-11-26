"""Shared fakes so unit tests never load a model, call an LLM or touch audio hardware.

Flow:
    tests/unit/* -> fixtures here
        settings      -> Settings(_env_file=None) pointed at a tmp index dir
        chunks        -> a small fixed corpus of common.types.Chunk
        encoder       -> FakeEncoder: hashed bag-of-words vectors (deterministic, normalised)
        reranker      -> FakeReranker: word-overlap score
        retriever     -> HybridRetriever over FaissStore with the fakes
        FakeLLM       -> scripted streaming replies
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pytest

from common.types import Chunk
from configs import Settings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_WORD = re.compile(r"[a-z0-9]+")


class FakeEncoder:
    """Hashed bag-of-words embedding; similar wording gives similar vectors."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeReranker:
    """Score = shared words minus a constant, so unrelated passages score negative."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_WORD.findall(query.lower()))
        return [float(len(q & set(_WORD.findall(p.lower()))) - 2) for p in passages]


class FakeLLM:
    """Streams a scripted reply in small pieces; records the messages it received."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "", rewrite: str = "rewritten question?"):
        self.reply = reply
        self.rewrite = rewrite
        self.calls: list[list[dict]] = []

    def stream(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append(messages)
        text = self.rewrite if "Rewrite" in messages[0]["content"] else self.reply
        for i in range(0, len(text), 7):
            yield text[i : i + 7]

    def complete(self, messages, *, temperature=None, max_tokens=None):
        return "".join(self.stream(messages, temperature=temperature, max_tokens=max_tokens))

    def health(self) -> bool:
        return True


CORPUS = [
    Chunk(
        "faq.md:0:0",
        "Monthly fee\nEveryday Checking has a monthly fee of 8 dollars.",
        "faq.md",
        None,
        "Monthly fee",
    ),
    Chunk(
        "faq.md:0:1",
        "Wires\nOutgoing international wires cost 45 dollars.",
        "faq.md",
        None,
        "Wires",
    ),
    Chunk(
        "policy.pdf:4:0",
        "Exclusions\nFlood and earthquake damage are excluded.",
        "policy.pdf",
        4,
        "Exclusions",
    ),
    Chunk(
        "policy.pdf:5:0",
        "Deductibles\nThe standard deductible is 1,000 dollars.",
        "policy.pdf",
        5,
        "Deductibles",
    ),
]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        paths={
            "index_dir": str(tmp_path / "index"),
            "docs_dir": str(tmp_path / "docs"),
            "models_dir": str(tmp_path / "models"),
        },
        vector_store={"backend": "faiss"},
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return list(CORPUS)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def retriever(chunks, encoder, reranker):
    from retrieval.retriever import HybridRetriever
    from retrieval.vectorstore import FaissStore

    store = FaissStore(encoder.dim)
    store.add(chunks, encoder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(chunks, store, encoder, reranker, candidates=10, rerank_candidates=10)
