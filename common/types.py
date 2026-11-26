"""
File: common/types.py

Description:
    Plain data records shared by ingestion, retrieval, answering and the API.

Flow:
    ingest.chunking.chunk_documents()     -> list[Chunk]
    retrieval.* (stores, fusion, rerank)  -> list[Hit] (a Chunk plus scores)
    rag.citations / api.schemas           -> Citation built from a Hit

Libraries used:
    dataclasses - standard library; dataclass, field and asdict for the records

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Chunk:
    """
    Class name: Chunk
    Purpose: One retrievable passage with its provenance.
    """

    chunk_id: str
    text: str
    source: str  # file name, e.g. acme_home_insurance_policy.pdf
    page: int | None = None  # 1-based PDF page; None for Markdown/text
    section: str = ""  # nearest heading above the passage

    def to_dict(self) -> dict:
        """
        Function name: Chunk.to_dict
        Purpose: Convert the chunk to a plain dictionary.
        Parameters:
            None
        Returns:
            dict: The chunk fields as a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        """
        Function name: Chunk.from_dict
        Purpose: Build a chunk from a dictionary; missing fields become None and a missing
                 or empty section becomes "".
        Parameters:
            d (dict): Dictionary with chunk_id, text, source, page and section keys.
        Returns:
            Chunk: The new chunk.
        """
        return cls(
            **{k: d.get(k) for k in ("chunk_id", "text", "source", "page")},
            section=d.get("section") or "",
        )

    def label(self) -> str:
        """
        Function name: Chunk.label
        Purpose: Build the citation label: "source, p. N" for PDFs, "source, Section"
                 otherwise, or just the source when there is no section.
        Parameters:
            None
        Returns:
            str: The citation label.
        """
        if self.page is not None:
            return f"{self.source}, p. {self.page}"
        return f"{self.source}, {self.section}" if self.section else self.source


@dataclass
class Hit:
    """
    Class name: Hit
    Purpose: A chunk returned by a retriever, with the scores each stage assigned.
    """

    chunk: Chunk
    score: float = 0.0  # score of the stage that produced the current ordering
    scores: dict[str, float] = field(default_factory=dict)  # dense, bm25, rrf, rerank
