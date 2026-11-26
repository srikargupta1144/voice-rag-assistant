"""
File: ingest/build.py

Description:
    Build the search indexes from the documents folder.

    BM25 is not written to disk: it is rebuilt from chunks.jsonl when the retriever loads,
    which takes well under a second for this corpus and keeps one source of truth.
    Re-running replaces the local index (FAISS is rebuilt from scratch; Pinecone upserts by
    chunk id).

Flow:
    ingest.cli ingest  /  api POST /ingest
        -> build_index(settings)
            -> ingest.loader.load_documents(docs_dir)
            -> ingest.chunking.chunk_documents(docs, chunk_tokens, overlap_tokens)
            -> retrieval.embeddings.get_encoder().embed_documents(chunk texts)
            -> retrieval.vectorstore.build_vector_store() -> add() -> save(index_dir)
            -> retrieval.retriever.write_chunks(index_dir, chunks, manifest)

Libraries used:
    logging - standard library; logs the ingestion summary
    time - standard library; measures how long ingestion takes
    typing - standard library; TYPE_CHECKING guard for type-only imports
    configs - project; Settings type (type checking only)
    ingest.chunking - project; chunk_documents and count_tokens
    ingest.loader - project; load_documents
    retrieval.embeddings - project; get_encoder and the Encoder type
    retrieval.retriever - project; write_chunks
    retrieval.vectorstore - project; build_vector_store

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Function name: build_index
    Purpose: Load, chunk, embed and index every document under the documents folder, then
             write the chunk records and the ingestion manifest next to the vector index.
    Parameters:
        settings (Settings): application settings (paths, chunking, embedding, vector store)
        encoder (Encoder | None): encoder to use; when None the configured encoder is loaded
    Returns:
        dict: the ingestion manifest (document and chunk counts, sources, largest chunk size,
              embedding model, vector dimension, vector store name and elapsed seconds)
    """
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
