"""
File: tests/conftest.py

Description:
    Shared fakes so unit tests never load a model, call an LLM or touch audio hardware.

Flow:
    tests/unit/* -> fixtures here
        settings      -> Settings(_env_file=None) pointed at a tmp index dir
        chunks        -> a small fixed corpus of common.types.Chunk
        encoder       -> FakeEncoder: hashed bag-of-words vectors (deterministic, normalised)
        reranker      -> FakeReranker: word-overlap score
        retriever     -> HybridRetriever over FaissStore with the fakes
        FakeLLM       -> scripted streaming replies

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    hashlib - standard library; MD5 hashing of words into vector positions
    os - standard library; sets KMP_DUPLICATE_LIB_OK
    re - standard library; splits text into lower-case words
    numpy - third party; fake embedding vectors
    pytest - third party; fixture definitions
    common.types - project; the Chunk type of the test corpus
    configs - project; the Settings class for the settings fixture
    retrieval.retriever - project; HybridRetriever built by the retriever fixture
    retrieval.vectorstore - project; FaissStore used by the retriever fixture

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Class name: FakeEncoder
    Purpose: Hashed bag-of-words embedding; similar wording gives similar vectors. Replaces
        the real sentence encoder in tests.
    """

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        """
        Function name: FakeEncoder._vec
        Purpose: Build a normalised bag-of-words vector by hashing each lower-case word to
            one of dim positions and counting occurrences.
        Parameters:
            text (str): Text to embed.
        Returns:
            np.ndarray: A float32 vector of length dim with unit norm (all zeros when the
                text has no words).
        """
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Function name: FakeEncoder.embed_documents
        Purpose: Embed a list of passages.
        Parameters:
            texts (list[str]): Passages to embed.
        Returns:
            np.ndarray: One row per text, shape (len(texts), dim); shape (0, dim) when empty.
        """
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        """
        Function name: FakeEncoder.embed_query
        Purpose: Embed a single query the same way as a passage.
        Parameters:
            text (str): Query text.
        Returns:
            np.ndarray: The query vector of length dim.
        """
        return self._vec(text)


class FakeReranker:
    """
    Class name: FakeReranker
    Purpose: Score = shared words minus a constant, so unrelated passages score negative.
        Replaces the real cross-encoder in tests.
    """

    def score(self, query: str, passages: list[str]) -> list[float]:
        """
        Function name: FakeReranker.score
        Purpose: Score each passage by the number of distinct words it shares with the query,
            minus 2.
        Parameters:
            query (str): Query text.
            passages (list[str]): Passages to score.
        Returns:
            list[float]: One score per passage, in input order.
        """
        q = set(_WORD.findall(query.lower()))
        return [float(len(q & set(_WORD.findall(p.lower()))) - 2) for p in passages]


class FakeLLM:
    """
    Class name: FakeLLM
    Purpose: Streams a scripted reply in small pieces; records the messages it received.
        Returns the rewrite text when the system prompt asks for a rewrite.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "", rewrite: str = "rewritten question?"):
        """
        Function name: FakeLLM.__init__
        Purpose: Store the scripted answer and rewrite replies and start an empty call log.
        Parameters:
            reply (str): Text returned for answer prompts.
            rewrite (str): Text returned for query rewrite prompts.
        Returns:
            None: Initialises the instance.
        """
        self.reply = reply
        self.rewrite = rewrite
        self.calls: list[list[dict]] = []

    def stream(self, messages, *, temperature=None, max_tokens=None):
        """
        Function name: FakeLLM.stream
        Purpose: Record the messages and yield the scripted reply in pieces of seven
            characters; the rewrite text is used when the first message contains "Rewrite".
        Parameters:
            messages (list[dict]): Chat messages with "role" and "content" keys.
            temperature (float | None): Ignored; accepted to match the client interface.
            max_tokens (int | None): Ignored; accepted to match the client interface.
        Returns:
            Iterator[str]: yields pieces of the scripted reply.
        """
        self.calls.append(messages)
        text = self.rewrite if "Rewrite" in messages[0]["content"] else self.reply
        for i in range(0, len(text), 7):
            yield text[i : i + 7]

    def complete(self, messages, *, temperature=None, max_tokens=None):
        """
        Function name: FakeLLM.complete
        Purpose: Return the whole scripted reply by joining the pieces from stream().
        Parameters:
            messages (list[dict]): Chat messages with "role" and "content" keys.
            temperature (float | None): Passed through to stream().
            max_tokens (int | None): Passed through to stream().
        Returns:
            str: The full scripted reply.
        """
        return "".join(self.stream(messages, temperature=temperature, max_tokens=max_tokens))

    def health(self) -> bool:
        """
        Function name: FakeLLM.health
        Purpose: Report the fake LLM as always reachable.
        Parameters:
            None
        Returns:
            bool: Always True.
        """
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
    """
    Function name: settings
    Purpose: Fixture that builds settings without reading a .env file, with the index, docs
        and models folders under the test's temporary directory and the FAISS vector store.
    Parameters:
        tmp_path (Path): pytest temporary directory for this test.
    Returns:
        Settings: The test settings.
    """
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
    """
    Function name: chunks
    Purpose: Fixture that provides a fresh copy of the four-chunk test corpus.
    Parameters:
        None
    Returns:
        list[Chunk]: The corpus chunks (two FAQ chunks and two policy PDF chunks).
    """
    return list(CORPUS)


@pytest.fixture
def encoder() -> FakeEncoder:
    """
    Function name: encoder
    Purpose: Fixture that provides the fake hashed bag-of-words encoder.
    Parameters:
        None
    Returns:
        FakeEncoder: A new fake encoder.
    """
    return FakeEncoder()


@pytest.fixture
def reranker() -> FakeReranker:
    """
    Function name: reranker
    Purpose: Fixture that provides the fake word-overlap re-ranker.
    Parameters:
        None
    Returns:
        FakeReranker: A new fake re-ranker.
    """
    return FakeReranker()


@pytest.fixture
def retriever(chunks, encoder, reranker):
    """
    Function name: retriever
    Purpose: Fixture that builds a HybridRetriever over a FAISS store filled with the test
        corpus, using the fake encoder and re-ranker.
    Parameters:
        chunks (list[Chunk]): Test corpus fixture.
        encoder (FakeEncoder): Fake encoder fixture.
        reranker (FakeReranker): Fake re-ranker fixture.
    Returns:
        HybridRetriever: The retriever, with 10 candidates for fusion and for re-ranking.
    """
    from retrieval.retriever import HybridRetriever
    from retrieval.vectorstore import FaissStore

    store = FaissStore(encoder.dim)
    store.add(chunks, encoder.embed_documents([c.text for c in chunks]))
    return HybridRetriever(chunks, store, encoder, reranker, candidates=10, rerank_candidates=10)
