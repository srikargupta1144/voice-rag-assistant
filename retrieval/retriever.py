"""
File: retrieval/retriever.py

Description:
    Hybrid retriever: dense (FAISS or Pinecone) and BM25, fused with RRF, then cross-encoder
    re-ranking.

    Modes exist so eval.retrieval_eval can measure each step (dense, bm25, hybrid, full).

Flow:
    rag.pipeline.AnswerPipeline, eval.retrieval_eval, api /health
        -> load_retriever(settings)
            -> read_chunks(index_dir)                       # chunks.jsonl from ingestion
            -> build_vector_store(...).load() (+ FaissStore.set_chunks)
            -> BM25Index(chunk texts)
        -> HybridRetriever.retrieve(query, mode, top_k)
            -> dense:  encoder.embed_query() -> store.search()
            -> bm25:   BM25Index.search()
            -> hybrid: retrieval.fusion.reciprocal_rank_fusion([dense, bm25])
            -> full:   hybrid, then reranker.score() over the fused candidates

Libraries used:
    json - standard library; reads and writes chunks.jsonl and manifest.json
    logging - standard library; logs when the retriever is loaded
    pathlib - standard library; Path for the index folder
    typing - standard library; TYPE_CHECKING guard for type-only imports
    common.types - project; Chunk and Hit records
    configs - project; Settings type (type checking only)
    retrieval.bm25 - project; BM25Index
    retrieval.embeddings - project; get_encoder and the Encoder type
    retrieval.fusion - project; reciprocal_rank_fusion
    retrieval.rerank - project; get_reranker and the Reranker type
    retrieval.vectorstore - project; build_vector_store, FaissStore and the VectorStore type

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from common.types import Chunk, Hit
from retrieval.bm25 import BM25Index
from retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from configs import Settings
    from retrieval.embeddings import Encoder
    from retrieval.rerank import Reranker
    from retrieval.vectorstore import VectorStore

log = logging.getLogger(__name__)

MODES = ("dense", "bm25", "hybrid", "full")
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class HybridRetriever:
    """
    Class name: HybridRetriever
    Purpose: Combines the dense store, BM25 and the re-ranker over one chunk list.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        store: VectorStore,
        encoder: Encoder,
        reranker: Reranker | None,
        candidates: int = 20,
        rrf_k: int = 60,
        rerank_candidates: int = 20,
    ):
        """
        Function name: HybridRetriever.__init__
        Purpose: Keep the search components, index the chunks by id and build the BM25 index
                 over the chunk texts.
        Parameters:
            chunks (list[Chunk]): all indexed chunks, in index order
            store (VectorStore): dense vector store over the same chunks
            encoder (Encoder): encoder for queries
            reranker (Reranker | None): cross-encoder re-ranker; None skips re-ranking
            candidates (int): hits taken from each leg (dense and BM25) before fusion
            rrf_k (int): RRF constant
            rerank_candidates (int): fused hits passed to the re-ranker in "full" mode
        Returns:
            None
        """
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.store = store
        self.encoder = encoder
        self.reranker = reranker
        self.bm25 = BM25Index([c.text for c in chunks])
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.rerank_candidates = rerank_candidates

    def dense(self, query: str, k: int) -> list[Hit]:
        """
        Function name: HybridRetriever.dense
        Purpose: Search with the dense leg only (query embedding against the vector store).
        Parameters:
            query (str): search text
            k (int): largest number of hits to return
        Returns:
            list[Hit]: hits by cosine similarity, best first
        """
        return self.store.search(self.encoder.embed_query(query), k)

    def sparse(self, query: str, k: int) -> list[Hit]:
        """
        Function name: HybridRetriever.sparse
        Purpose: Search with the BM25 leg only.
        Parameters:
            query (str): search text
            k (int): largest number of hits to return
        Returns:
            list[Hit]: hits with a positive BM25 score, best first
        """
        return [
            Hit(chunk=self.chunks[i], score=s, scores={"bm25": s})
            for i, s in self.bm25.search(query, k)
        ]

    def hybrid(self, query: str, k: int) -> list[Hit]:
        """
        Function name: HybridRetriever.hybrid
        Purpose: Run RRF over the dense and BM25 candidate lists. Each hit keeps the scores
                 from both legs and gets the fused score as its main score.
        Parameters:
            query (str): search text
            k (int): largest number of hits to return
        Returns:
            list[Hit]: fused hits, best first
        """
        dense = self.dense(query, self.candidates)
        sparse = self.sparse(query, self.candidates)
        merged: dict[str, Hit] = {}
        for h in dense + sparse:
            if h.chunk.chunk_id in merged:
                merged[h.chunk.chunk_id].scores.update(h.scores)
            else:
                merged[h.chunk.chunk_id] = Hit(chunk=h.chunk, scores=dict(h.scores))
        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in dense], [h.chunk.chunk_id for h in sparse]], k=self.rrf_k
        )
        out = []
        for cid, score in fused[:k]:
            hit = merged[cid]
            hit.score = score
            hit.scores["rrf"] = score
            out.append(hit)
        return out

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        """
        Function name: HybridRetriever.rerank
        Purpose: Re-order hits by cross-encoder logit and keep the best k. When there is no
                 re-ranker or no hits, the first k hits are returned unchanged.
        Parameters:
            query (str): search text
            hits (list[Hit]): candidate hits; their scores are updated in place
            k (int): largest number of hits to return
        Returns:
            list[Hit]: the best k hits, best first
        """
        if not hits or self.reranker is None:
            return hits[:k]
        scores = self.reranker.score(query, [h.chunk.text for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.score = s
            h.scores["rerank"] = s
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    def retrieve(self, query: str, mode: str = "full", top_k: int = 5) -> list[Hit]:
        """
        Function name: HybridRetriever.retrieve
        Purpose: Search in the given mode (see the module Flow section). "full" re-ranks the
                 top rerank_candidates hybrid hits.
        Parameters:
            query (str): search text
            mode (str): one of "dense", "bm25", "hybrid" or "full"
            top_k (int): largest number of hits to return
        Returns:
            list[Hit]: hits, best first
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "dense":
            return self.dense(query, top_k)
        if mode == "bm25":
            return self.sparse(query, top_k)
        if mode == "hybrid":
            return self.hybrid(query, top_k)
        return self.rerank(query, self.hybrid(query, self.rerank_candidates), top_k)


def write_chunks(index_dir: Path, chunks: list[Chunk], manifest: dict) -> None:
    """
    Function name: write_chunks
    Purpose: Save the chunk records (chunks.jsonl) and the ingestion manifest (manifest.json)
             next to the vector index, creating the folder if needed.
    Parameters:
        index_dir (Path): index folder
        chunks (list[Chunk]): chunks to write, one JSON line each
        manifest (dict): ingestion summary
    Returns:
        None
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_chunks(index_dir: Path) -> list[Chunk]:
    """
    Function name: read_chunks
    Purpose: Load the chunk records written by write_chunks(). Raises FileNotFoundError when
             there is no index.
    Parameters:
        index_dir (Path): index folder
    Returns:
        list[Chunk]: the chunks in file order
    """
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(f"no index at {index_dir}; run: uv run vra ingest")
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def load_retriever(
    settings: Settings,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
    with_reranker: bool = True,
) -> HybridRetriever:
    """
    Function name: load_retriever
    Purpose: Build a retriever from the on-disk index: read the chunks, load the vector
             store and, when needed, the configured encoder and re-ranker.
    Parameters:
        settings (Settings): application settings (paths and retrieval sections)
        encoder (Encoder | None): encoder to use; when None the configured encoder is loaded
        reranker (Reranker | None): re-ranker to use; when None the configured re-ranker is
                                    loaded if with_reranker is True
        with_reranker (bool): False builds a retriever without a re-ranker
    Returns:
        HybridRetriever: the ready retriever
    """
    from retrieval.embeddings import get_encoder
    from retrieval.rerank import get_reranker
    from retrieval.vectorstore import FaissStore, build_vector_store

    index_dir = settings.paths.resolve(settings.paths.index_dir)
    chunks = read_chunks(index_dir)
    encoder = encoder or get_encoder(settings)
    store = build_vector_store(settings, encoder.dim)
    store.load(index_dir)
    if isinstance(store, FaissStore):
        store.set_chunks(chunks)
    if reranker is None and with_reranker:
        reranker = get_reranker(settings)
    r = settings.retrieval
    log.info("retriever loaded chunks=%d store=%s", len(chunks), store.name)
    return HybridRetriever(
        chunks,
        store,
        encoder,
        reranker,
        candidates=r.candidates,
        rrf_k=r.rrf_k,
        rerank_candidates=r.rerank_candidates,
    )
