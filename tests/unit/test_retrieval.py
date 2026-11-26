"""Retrieval interface: modes, BM25, FAISS store, persistence, backend selection."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from retrieval.bm25 import BM25Index, bm25_tokens
from retrieval.retriever import read_chunks, write_chunks
from retrieval.vectorstore import FaissStore, PineconeStore, build_vector_store


def _auto(settings):
    vs = settings.vector_store.model_copy(update={"backend": "auto"})
    return settings.model_copy(update={"vector_store": vs})


def test_bm25_tokens_keep_ids_and_drop_stopwords():
    assert bm25_tokens("What is the HE-12 endorsement?") == ["he-12", "endorsement"]


def test_bm25_ranks_exact_terms():
    idx = BM25Index(["flood is excluded", "wire costs 45 dollars", "monthly fee"])
    assert idx.search("international wire", 2)[0][0] == 1
    assert idx.search("the of", 2) == []  # only stop words


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "full"])
def test_every_mode_finds_the_wire_chunk(retriever, mode):
    hits = retriever.retrieve("How much do international wires cost?", mode=mode, top_k=2)
    assert hits[0].chunk.chunk_id == "faq.md:0:1"


def test_full_mode_records_all_scores(retriever):
    hit = retriever.retrieve("flood earthquake excluded", mode="full", top_k=1)[0]
    assert {"rrf", "rerank"} <= set(hit.scores)
    assert hit.score == hit.scores["rerank"]


def test_unknown_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")


def test_faiss_store_roundtrip(tmp_path, chunks, encoder):
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
    with pytest.raises(ValueError):
        FaissStore(4).add(chunks, np.zeros((1, 4), dtype=np.float32))


def test_chunk_file_roundtrip(tmp_path, chunks):
    write_chunks(tmp_path, chunks, {"chunks": len(chunks)})
    assert read_chunks(tmp_path) == chunks
    with pytest.raises(FileNotFoundError):
        read_chunks(tmp_path / "missing")


def test_backend_defaults_to_faiss_without_key(settings, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    assert isinstance(build_vector_store(_auto(settings), 8), FaissStore)


def test_backend_switches_to_pinecone_when_key_set(settings, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    client = MagicMock()
    client.has_index.return_value = True
    assert isinstance(build_vector_store(_auto(settings), 8, client=client), PineconeStore)
