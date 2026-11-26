"""Build the search indexes from the documents folder.

Flow:
    ingest.cli ingest  /  api POST /ingest
        -> build_index(settings)
            -> ingest.loader.load_documents(docs_dir)
            -> ingest.chunking.chunk_documents(docs, chunk_tokens, overlap_tokens)
            -> retrieval.embeddings.get_encoder().embed_documents(chunk texts)
            -> retrieval.vectorstore.build_vector_store() -> add() -> save(index_dir)
            -> retrieval.retriever.write_chunks(index_dir, chunks, manifest)

BM25 is not written to disk: it is rebuilt from chunks.jsonl when the retriever loads,
which takes well under a second for this corpus and keeps one source of truth.
Re-running replaces the local index (FAISS is rebuilt from scratch; Pinecone upserts by
chunk id).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ingest.chunking import chunk_documents, count_tokens
from ingest.loader import load_documents
from retrieval.retriever import write_chunks

if TYPE_CHECKING:
    from configs import Settings
    from retrieval.embeddings import Encoder

log = logging.getLogger(__name__)


def build_index(settings: Settings, encoder: Encoder | None = None) -> dict:
    """Load, chunk, embed and index every document; returns the manifest."""
    from retrieval.embeddings import get_encoder
    from retrieval.vectorstore import build_vector_store

    t0 = time.perf_counter()
    docs_dir = settings.paths.resolve(settings.paths.docs_dir)
    index_dir = settings.paths.resolve(settings.paths.index_dir)
    documents = load_documents(docs_dir)
    if not documents:
        raise FileNotFoundError(f"no .pdf/.md/.txt files under {docs_dir}")
    chunks = chunk_documents(
        documents, settings.chunking.chunk_tokens, settings.chunking.overlap_tokens
    )
    encoder = encoder or get_encoder(settings)
    vectors = encoder.embed_documents([c.text for c in chunks])
    store = build_vector_store(settings, encoder.dim)
    store.add(chunks, vectors)
    store.save(index_dir)
    sources = sorted({c.source for c in chunks})
    manifest = {
        "documents": len(documents),
        "sources": sources,
        "chunks": len(chunks),
        "max_chunk_tokens": max(count_tokens(c.text) for c in chunks),
        "embedding_model": settings.embedding.model,
        "dim": encoder.dim,
        "vector_store": store.name,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    write_chunks(index_dir, chunks, manifest)
    log.info("ingest done %s", " ".join(f"{k}={v}" for k, v in manifest.items()))
    return manifest
