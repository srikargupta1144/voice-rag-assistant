"""
File: tests/unit/test_retrieval.py

Description:
    Unit tests for the retrieval interface: modes, BM25, the FAISS store, persistence and
    backend selection.

Libraries used:
    unittest.mock - standard library; MagicMock stands in for the Pinecone client
    numpy - third party; test vectors
    pytest - third party; parametrisation and error checks
    retrieval.bm25 - project; the BM25 index and tokenizer under test
    retrieval.retriever - project; chunk file reading and writing under test
    retrieval.vectorstore - project; FaissStore, PineconeStore and backend selection under
        test

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from retrieval.bm25 import BM25Index, bm25_tokens
from retrieval.retriever import read_chunks, write_chunks
from retrieval.vectorstore import FaissStore, PineconeStore, build_vector_store


def _auto(settings):
    """
    Function name: _auto
    Purpose: Return a copy of the settings with the vector store backend set to "auto".
    Parameters:
        settings (Settings): Test settings.
    Returns:
        Settings: The updated copy.
    """
    vs = settings.vector_store.model_copy(update={"backend": "auto"})
    return settings.model_copy(update={"vector_store": vs})


def test_bm25_tokens_keep_ids_and_drop_stopwords():
    """
    Function name: test_bm25_tokens_keep_ids_and_drop_stopwords
    Purpose: Check that BM25 tokens are lower-cased, keep hyphenated ids and drop stop words.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert bm25_tokens("What is the HE-12 endorsement?") == ["he-12", "endorsement"]


def test_bm25_ranks_exact_terms():
    """
    Function name: test_bm25_ranks_exact_terms
    Purpose: Check that BM25 ranks the document with the query term first and returns
        nothing for a query of stop words only.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    idx = BM25Index(["flood is excluded", "wire costs 45 dollars", "monthly fee"])
    assert idx.search("international wire", 2)[0][0] == 1
    assert idx.search("the of", 2) == []  # only stop words


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "full"])
def test_every_mode_finds_the_wire_chunk(retriever, mode):
    """
    Function name: test_every_mode_finds_the_wire_chunk
    Purpose: Check that every retrieval mode ranks the wire chunk first for a wire question.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
        mode (str): Retrieval mode (parametrised).
    Returns:
        None: The test passes or fails through assertions.
    """
    hits = retriever.retrieve("How much do international wires cost?", mode=mode, top_k=2)
    assert hits[0].chunk.chunk_id == "faq.md:0:1"


def test_full_mode_records_all_scores(retriever):
    """
    Function name: test_full_mode_records_all_scores
    Purpose: Check that full mode records both the fusion and re-rank scores and uses the
        re-rank score as the hit score.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    hit = retriever.retrieve("flood earthquake excluded", mode="full", top_k=1)[0]
    assert {"rrf", "rerank"} <= set(hit.scores)
    assert hit.score == hit.scores["rerank"]


def test_unknown_mode_rejected(retriever):
    """
    Function name: test_unknown_mode_rejected
    Purpose: Check that an unknown retrieval mode raises ValueError.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")


def test_faiss_store_roundtrip(tmp_path, chunks, encoder):
    """
    Function name: test_faiss_store_roundtrip
    Purpose: Check that a FAISS store saved to disk and loaded again returns the right chunk
        for a query and keeps the vector count.
    Parameters:
        tmp_path (Path): pytest temporary directory for the saved index.
        chunks (list[Chunk]): Test corpus fixture.
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    store = FaissStore(encoder.dim)
    store.add(chunks, encoder.embed_documents([c.text for c in chunks]))
    store.save(tmp_path)
    loaded = FaissStore(encoder.dim)
    loaded.load(tmp_path)
    loaded.set_chunks(chunks)
    hit = loaded.search(encoder.embed_query("standard deductible"), 1)[0]
    assert hit.chunk.chunk_id == "policy.pdf:5:0"
    assert loaded.count() == len(chunks)


def test_faiss_rejects_length_mismatch(chunks):
    """
    Function name: test_faiss_rejects_length_mismatch
    Purpose: Check that adding a different number of chunks and vectors raises ValueError.
    Parameters:
        chunks (list[Chunk]): Test corpus fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    with pytest.raises(ValueError):
        FaissStore(4).add(chunks, np.zeros((1, 4), dtype=np.float32))


def test_chunk_file_roundtrip(tmp_path, chunks):
    """
    Function name: test_chunk_file_roundtrip
    Purpose: Check that chunks written to disk are read back unchanged and that reading a
        missing folder raises FileNotFoundError.
    Parameters:
        tmp_path (Path): pytest temporary directory for the chunk file.
        chunks (list[Chunk]): Test corpus fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    write_chunks(tmp_path, chunks, {"chunks": len(chunks)})
    assert read_chunks(tmp_path) == chunks
    with pytest.raises(FileNotFoundError):
        read_chunks(tmp_path / "missing")


def test_backend_defaults_to_faiss_without_key(settings, monkeypatch):
    """
    Function name: test_backend_defaults_to_faiss_without_key
    Purpose: Check that the auto backend selects FAISS when no Pinecone API key is set.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (pytest.MonkeyPatch): Removes PINECONE_API_KEY from the environment.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    assert isinstance(build_vector_store(_auto(settings), 8), FaissStore)


def test_backend_switches_to_pinecone_when_key_set(settings, monkeypatch):
    """
    Function name: test_backend_switches_to_pinecone_when_key_set
    Purpose: Check that the auto backend selects Pinecone when a Pinecone API key is set.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (pytest.MonkeyPatch): Sets PINECONE_API_KEY in the environment.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    client = MagicMock()
    client.has_index.return_value = True
    assert isinstance(build_vector_store(_auto(settings), 8, client=client), PineconeStore)
