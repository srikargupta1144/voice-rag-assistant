"""Citation parsing and the not-found decision.

Flow:
    rag.pipeline.AnswerPipeline
        -> evidence_is_weak(hits, threshold)          # before the LLM: skip it when weak
        -> is_not_found(answer_text)                  # after the LLM: model declined
        -> resolve_citations(answer_text, hits)
            -> numeric markers [1], [2, 3]            # the format the prompt asks for
            -> label markers [source.pdf, p. 3]       # tolerated if the model copies labels
            -> returns display text + list[Citation]
    rag.pipeline / voice.tts callers
        -> strip_citations(text)                      # nothing in brackets is spoken
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from common.types import Hit
from rag.prompts import NOT_FOUND

_NUMERIC = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_LABEL = re.compile(
    r"\[\s*([\w.\- ]+?\.(?:pdf|md|markdown|txt))\s*(?:,\s*(?:p\.?|page)\s*(\d+))?[^\]]*\]",
    re.IGNORECASE,
)
_ANY_BRACKET = re.compile(r"\s*\[[^\]]*\]")
_NOT_FOUND_RE = re.compile(
    r"could\s*n[o']?t\s+find\s+(that|this|it|the answer)\s+in\s+the\s+documents", re.IGNORECASE
)


@dataclass(frozen=True)
class Citation:
    """One cited passage, as returned by the API."""

    source: str
    page: int | None
    section: str
    chunk_id: str
    label: str

    def to_dict(self) -> dict:
        return asdict(self)


def evidence_is_weak(hits: Sequence[Hit], threshold: float) -> bool:
    """True when nothing was retrieved or the best re-ranker score is below threshold.

    Hits without a re-ranker score (dense/bm25/hybrid modes) are never judged weak here;
    the prompt's own not-found instruction still applies.
    """
    if not hits:
        return True
    rerank = [h.scores["rerank"] for h in hits if "rerank" in h.scores]
    return bool(rerank) and max(rerank) < threshold


def is_not_found(answer: str) -> bool:
    """True when the model used the not-found reply (exact or close wording)."""
    return bool(_NOT_FOUND_RE.search(answer)) or answer.strip() == NOT_FOUND


def resolve_citations(answer: str, hits: Sequence[Hit]) -> tuple[str, list[Citation]]:
    """Replace markers with "[source, page]" labels; return text and unique citations."""
    cited: dict[str, Citation] = {}

    def cite(hit: Hit) -> str:
        c = hit.chunk
        cited.setdefault(c.chunk_id, Citation(c.source, c.page, c.section, c.chunk_id, c.label()))
        return f"[{c.label()}]"

    def numeric(m: re.Match) -> str:
        labels = []
        for n in re.split(r"[,;]", m.group(1)):
            i = int(n.strip())
            if 1 <= i <= len(hits):
                labels.append(cite(hits[i - 1]))
        return "".join(labels)  # out-of-range numbers are dropped, not invented

    def label(m: re.Match) -> str:
        source, page = m.group(1).strip().lower(), m.group(2)
        for h in hits:
            if h.chunk.source.lower() == source and (
                page is None or h.chunk.page is None or str(h.chunk.page) == page
            ):
                return cite(h)
        return ""

    text = _NUMERIC.sub(numeric, answer)
    text = _LABEL.sub(label, text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), list(cited.values())


def strip_citations(text: str) -> str:
    """Text to speak: every bracketed marker removed, spacing tidied."""
    text = _ANY_BRACKET.sub("", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()
