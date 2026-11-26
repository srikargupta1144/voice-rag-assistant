"""
File: retrieval/vectorstore.py

Description:
    Vector store interface with two backends: FAISS (default, local) and Pinecone (optional).

    Selection (settings.vector_store.effective_backend()):
        "faiss"    -> FaissStore: exact inner-product index (IndexFlatIP) saved under data/index
        "pinecone" -> PineconeStore: serverless index; only chosen when PINECONE_API_KEY is set
                      (backend "auto") or forced with VRA_VECTOR_STORE__BACKEND=pinecone.

    The Pinecone SDK is imported lazily and the client can be injected, so tests and machines
    without the optional `pinecone` extra never touch it.

Flow:
    ingest.build.build_index()
        -> build_vector_store(settings, dim) -> store.add(chunks, vectors) -> store.save(dir)
    retrieval.retriever.load_retriever()
        -> build_vector_store(settings, dim) -> store.load(dir) -> store.search(vector, k)

Libraries used:
    json - standard library; writes the FAISS row to chunk id list
    logging - standard library; logs Pinecone index creation
    pathlib - standard library; Path for the index folder
    typing - standard library; Any, Protocol and the TYPE_CHECKING guard
    numpy - third party; vector arrays
    faiss - third party; IndexFlatIP and index file read and write
    pinecone - third party (optional extra); Pinecone client and ServerlessSpec
    common.types - project; Chunk and Hit records
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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from common.types import Chunk, Hit

if TYPE_CHECKING:
    from configs import Settings

log = logging.getLogger(__name__)


class VectorStore(Protocol):
    """
    Class name: VectorStore
    Purpose: Dense index over chunks; vectors are L2-normalised so scores are cosine
             similarity.
    """

    name: str

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Function name: VectorStore.add
        Purpose: Add chunks and their vectors to the index.
        Parameters:
            chunks (list[Chunk]): chunks to add
            vectors (np.ndarray): one vector per chunk, in the same order
        Returns:
            None
        """
        ...

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """
        Function name: VectorStore.search
        Purpose: Find the chunks closest to a query vector.
        Parameters:
            vector (np.ndarray): query vector
            k (int): largest number of hits to return
        Returns:
            list[Hit]: hits by cosine similarity, best first
        """
        ...

    def count(self) -> int:
        """
        Function name: VectorStore.count
        Purpose: Report how many vectors the index holds.
        Parameters:
            None
        Returns:
            int: number of stored vectors
        """
        ...

    def save(self, directory: Path) -> None:
        """
        Function name: VectorStore.save
        Purpose: Write the index to a folder.
        Parameters:
            directory (Path): index folder
        Returns:
            None
        """
        ...

    def load(self, directory: Path) -> None:
        """
        Function name: VectorStore.load
        Purpose: Read the index from a folder.
        Parameters:
            directory (Path): index folder
        Returns:
            None
        """
        ...


class FaissStore:
    """
    Class name: FaissStore
    Purpose: Exact inner-product search; at a few thousand chunks a flat index takes < 1 ms.
    """

    name = "faiss"
    INDEX_FILE = "faiss.index"
    IDS_FILE = "faiss_ids.json"

    def __init__(self, dim: int):
        """
        Function name: FaissStore.__init__
        Purpose: Create an empty flat inner-product FAISS index.
        Parameters:
            dim (int): vector dimension
        Returns:
            None
        """
        import faiss

        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Function name: FaissStore.add
        Purpose: Append chunks; row i of vectors belongs to chunks[i]. Raises ValueError when
                 the lengths differ.
        Parameters:
            chunks (list[Chunk]): chunks to add
            vectors (np.ndarray): one normalised vector per chunk
        Returns:
            None
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks):
            self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self._chunks.extend(chunks)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """
        Function name: FaissStore.search
        Purpose: Return the top-k chunks by cosine similarity.
        Parameters:
            vector (np.ndarray): normalised query vector
            k (int): largest number of hits to return
        Returns:
            list[Hit]: hits with the "dense" score, best first; empty when no chunks are set
        """
        if not self._chunks:
            return []
        q = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        return [
            Hit(chunk=self._chunks[i], score=float(s), scores={"dense": float(s)})
            for s, i in zip(scores[0], idx[0], strict=True)
            if i >= 0
        ]

    def count(self) -> int:
        """
        Function name: FaissStore.count
        Purpose: Report how many chunks the store holds.
        Parameters:
            None
        Returns:
            int: number of attached chunk records
        """
        return len(self._chunks)

    def set_chunks(self, chunks: list[Chunk]) -> None:
        """
        Function name: FaissStore.set_chunks
        Purpose: Attach chunk records after load(); positions match index rows.
        Parameters:
            chunks (list[Chunk]): chunk records in index row order
        Returns:
            None
        """
        self._chunks = list(chunks)

    def save(self, directory: Path) -> None:
        """
        Function name: FaissStore.save
        Purpose: Write the index and the row to chunk_id order, creating the folder if
                 needed.
        Parameters:
            directory (Path): index folder
        Returns:
            None
        """
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(directory / self.INDEX_FILE))
        ids = [c.chunk_id for c in self._chunks]
        (directory / self.IDS_FILE).write_text(json.dumps(ids), encoding="utf-8")

    def load(self, directory: Path) -> None:
        """
        Function name: FaissStore.load
        Purpose: Read the index file and its dimension. Chunk records are not read here;
                 retrieval.retriever.load_retriever() attaches them with set_chunks().
        Parameters:
            directory (Path): index folder
        Returns:
            None
        """
        self._index = self._faiss.read_index(str(directory / self.INDEX_FILE))
        self.dim = self._index.d


class PineconeStore:
    """
    Class name: PineconeStore
    Purpose: Pinecone serverless index; chunk text and provenance travel in vector metadata.
    """

    name = "pinecone"
    UPSERT_BATCH = 100

    def __init__(
        self,
        dim: int,
        index_name: str,
        namespace: str,
        api_key: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        client: Any | None = None,
    ):
        """
        Function name: PineconeStore.__init__
        Purpose: Connect to Pinecone (or use the injected client), create the index if it
                 does not exist, and open it.
        Parameters:
            dim (int): vector dimension
            index_name (str): Pinecone index name
            namespace (str): namespace used for every read and write
            api_key (str): Pinecone API key; used only when no client is injected
            cloud (str): cloud provider for a new serverless index
            region (str): cloud region for a new serverless index
            client (Any | None): Pinecone client to use; None creates one from the SDK
        Returns:
            None
        """
        self.dim = dim
        self.namespace = namespace
        if client is None:
            from pinecone import Pinecone  # optional extra: uv sync --extra pinecone

            client = Pinecone(api_key=api_key)
        self._client = client
        self._index_name = index_name
        self._ensure_index(cloud, region)
        self._index = client.Index(index_name)

    def _ensure_index(self, cloud: str, region: str) -> None:
        """
        Function name: PineconeStore._ensure_index
        Purpose: Create the serverless index on first use (cosine metric, matching
                 dimension). Without the SDK installed, a plain dict spec is passed so that
                 injected test clients still work.
        Parameters:
            cloud (str): cloud provider
            region (str): cloud region
        Returns:
            None
        """
        if self._client.has_index(self._index_name):
            return
        try:
            from pinecone import ServerlessSpec

            spec: Any = ServerlessSpec(cloud=cloud, region=region)
        except ImportError:  # injected test clients run without the SDK installed
            spec = {"serverless": {"cloud": cloud, "region": region}}
        log.info("pinecone create_index name=%s dim=%d", self._index_name, self.dim)
        self._client.create_index(
            name=self._index_name, dimension=self.dim, metric="cosine", spec=spec
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """
        Function name: PineconeStore.add
        Purpose: Upsert in batches of UPSERT_BATCH; ids are chunk ids, so re-ingestion
                 overwrites in place. Raises ValueError when the lengths differ.
        Parameters:
            chunks (list[Chunk]): chunks to add
            vectors (np.ndarray): one normalised vector per chunk
        Returns:
            None
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        records = [
            {"id": c.chunk_id, "values": v.astype(float).tolist(), "metadata": _to_meta(c)}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(records), self.UPSERT_BATCH):
            self._index.upsert(vectors=records[i : i + self.UPSERT_BATCH], namespace=self.namespace)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """
        Function name: PineconeStore.search
        Purpose: Query the top-k matches; their metadata is turned back into Chunk records.
        Parameters:
            vector (np.ndarray): normalised query vector
            k (int): largest number of hits to return
        Returns:
            list[Hit]: hits with the "dense" score, in the order Pinecone returns them
        """
        res = self._index.query(
            vector=vector.astype(float).tolist(),
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )
        matches = _get(res, "matches") or []
        hits = []
        for m in matches:
            score = float(_get(m, "score"))
            chunk = _from_meta(str(_get(m, "id")), _get(m, "metadata") or {})
            hits.append(Hit(chunk=chunk, score=score, scores={"dense": score}))
        return hits

    def count(self) -> int:
        """
        Function name: PineconeStore.count
        Purpose: Report how many vectors the namespace holds, from the index statistics.
        Parameters:
            None
        Returns:
            int: vector count of the namespace, or 0 when the namespace is not listed
        """
        stats = self._index.describe_index_stats()
        namespaces = _get(stats, "namespaces") or {}
        ns = namespaces.get(self.namespace) if isinstance(namespaces, dict) else None
        return int(_get(ns, "vector_count") or 0) if ns is not None else 0

    def save(self, directory: Path) -> None:
        """
        Function name: PineconeStore.save
        Purpose: Do nothing: there is nothing to write locally, because the vectors live in
                 Pinecone.
        Parameters:
            directory (Path): index folder (not used)
        Returns:
            None
        """

    def load(self, directory: Path) -> None:
        """
        Function name: PineconeStore.load
        Purpose: Do nothing: there is nothing to read locally, because the vectors live in
                 Pinecone.
        Parameters:
            directory (Path): index folder (not used)
        Returns:
            None
        """


def _get(obj: Any, key: str) -> Any:
    """
    Function name: _get
    Purpose: Read a field from SDK response objects or plain dicts alike.
    Parameters:
        obj (Any): response object, dict or None
        key (str): field name
    Returns:
        Any: the field value, or None when obj is None or the field is missing
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_meta(c: Chunk) -> dict[str, Any]:
    """
    Function name: _to_meta
    Purpose: Convert a chunk into Pinecone metadata. Pinecone metadata cannot hold null
             values, so a missing page is stored as -1.
    Parameters:
        c (Chunk): chunk to convert
    Returns:
        dict[str, Any]: text, source, page and section fields
    """
    # Pinecone metadata cannot hold null values, so a missing page is stored as -1.
    return {"text": c.text, "source": c.source, "page": c.page or -1, "section": c.section}


def _from_meta(chunk_id: str, meta: dict[str, Any]) -> Chunk:
    """
    Function name: _from_meta
    Purpose: Rebuild a chunk from Pinecone metadata; a page that is missing or not positive
             becomes None.
    Parameters:
        chunk_id (str): the vector id, which is the chunk id
        meta (dict[str, Any]): metadata stored by _to_meta()
    Returns:
        Chunk: the rebuilt chunk record
    """
    page = meta.get("page", -1)
    page = int(page) if page is not None and int(page) > 0 else None
    return Chunk(
        chunk_id=chunk_id,
        text=str(meta.get("text", "")),
        source=str(meta.get("source", "")),
        page=page,
        section=str(meta.get("section", "")),
    )


def build_vector_store(settings: Settings, dim: int, client: Any | None = None) -> VectorStore:
    """
    Function name: build_vector_store
    Purpose: Create the configured backend: FAISS unless Pinecone is selected.
    Parameters:
        settings (Settings): application settings (vector_store section)
        dim (int): vector dimension
        client (Any | None): Pinecone client to inject (used by tests); ignored for FAISS
    Returns:
        VectorStore: a FaissStore or a PineconeStore
    """
    vs = settings.vector_store
    backend = vs.effective_backend()
    if backend == "pinecone":
        return PineconeStore(
            dim=dim,
            index_name=vs.pinecone_index,
            namespace=vs.pinecone_namespace,
            api_key=vs.effective_key(),
            cloud=vs.pinecone_cloud,
            region=vs.pinecone_region,
            client=client,
        )
    return FaissStore(dim)
