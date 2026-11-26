"""Vector store interface with two backends: FAISS (default, local) and Pinecone (optional).

Flow:
    ingest.build.build_index()
        -> build_vector_store(settings, dim) -> store.add(chunks, vectors) -> store.save(dir)
    retrieval.retriever.load_retriever()
        -> build_vector_store(settings, dim) -> store.load(dir) -> store.search(vector, k)

Selection (settings.vector_store.effective_backend()):
    "faiss"    -> FaissStore: exact inner-product index (IndexFlatIP) saved under data/index
    "pinecone" -> PineconeStore: serverless index; only chosen when PINECONE_API_KEY is set
                  (backend "auto") or forced with VRA_VECTOR_STORE__BACKEND=pinecone.
The Pinecone SDK is imported lazily and the client can be injected, so tests and machines
without the optional `pinecone` extra never touch it.
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
    """Dense index over chunks; vectors are L2-normalised so scores are cosine similarity."""

    name: str

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[Hit]: ...

    def count(self) -> int: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...


class FaissStore:
    """Exact inner-product search; at a few thousand chunks a flat index takes < 1 ms."""

    name = "faiss"
    INDEX_FILE = "faiss.index"
    IDS_FILE = "faiss_ids.json"

    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Append chunks; row i of vectors belongs to chunks[i]."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks):
            self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self._chunks.extend(chunks)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """Top-k chunks by cosine similarity."""
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
        return len(self._chunks)

    def set_chunks(self, chunks: list[Chunk]) -> None:
        """Attach chunk records after load() (positions match index rows)."""
        self._chunks = list(chunks)

    def save(self, directory: Path) -> None:
        """Write the index and the row -> chunk_id order."""
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(directory / self.INDEX_FILE))
        ids = [c.chunk_id for c in self._chunks]
        (directory / self.IDS_FILE).write_text(json.dumps(ids), encoding="utf-8")

    def load(self, directory: Path) -> None:
        """Read the index; chunks come from retrieval.index_store via set_chunks()."""
        self._index = self._faiss.read_index(str(directory / self.INDEX_FILE))
        self.dim = self._index.d


class PineconeStore:
    """Pinecone serverless index; chunk text and provenance travel in vector metadata."""

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
        """Create the serverless index on first use (cosine metric, matching dim)."""
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
        """Upsert in batches; ids are chunk ids so re-ingestion overwrites in place."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        records = [
            {"id": c.chunk_id, "values": v.astype(float).tolist(), "metadata": _to_meta(c)}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(records), self.UPSERT_BATCH):
            self._index.upsert(vectors=records[i : i + self.UPSERT_BATCH], namespace=self.namespace)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """Top-k matches; metadata is turned back into Chunk records."""
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
        stats = self._index.describe_index_stats()
        namespaces = _get(stats, "namespaces") or {}
        ns = namespaces.get(self.namespace) if isinstance(namespaces, dict) else None
        return int(_get(ns, "vector_count") or 0) if ns is not None else 0

    def save(self, directory: Path) -> None:
        """Nothing to write locally: the vectors live in Pinecone."""

    def load(self, directory: Path) -> None:
        """Nothing to read locally: the vectors live in Pinecone."""


def _get(obj: Any, key: str) -> Any:
    """Read a field from SDK response objects or plain dicts alike."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_meta(c: Chunk) -> dict[str, Any]:
    # Pinecone metadata cannot hold null values, so a missing page is stored as -1.
    return {"text": c.text, "source": c.source, "page": c.page or -1, "section": c.section}


def _from_meta(chunk_id: str, meta: dict[str, Any]) -> Chunk:
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
    """Instantiate the configured backend (FAISS unless Pinecone is selected)."""
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
