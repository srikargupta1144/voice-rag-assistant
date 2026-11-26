"""Plain data records shared by ingestion, retrieval, answering and the API.

Flow:
    ingest.chunking.chunk_documents()     -> list[Chunk]
    retrieval.* (stores, fusion, rerank)  -> list[Hit] (a Chunk plus scores)
    rag.citations / api.schemas           -> Citation built from a Hit
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage with its provenance."""

    chunk_id: str
    text: str
    source: str  # file name, e.g. acme_home_insurance_policy.pdf
    page: int | None = None  # 1-based PDF page; None for Markdown/text
    section: str = ""  # nearest heading above the passage

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(
            **{k: d.get(k) for k in ("chunk_id", "text", "source", "page")},
            section=d.get("section") or "",
        )

    def label(self) -> str:
        """Citation label: "source, p. N" for PDFs, "source, Section" otherwise."""
        if self.page is not None:
            return f"{self.source}, p. {self.page}"
        return f"{self.source}, {self.section}" if self.section else self.source


@dataclass
class Hit:
    """A chunk returned by a retriever, with the scores each stage assigned."""

    chunk: Chunk
    score: float = 0.0  # score of the stage that produced the current ordering
    scores: dict[str, float] = field(default_factory=dict)  # dense, bm25, rrf, rerank
