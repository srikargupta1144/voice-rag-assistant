"""
File: rag/citations.py

Description:
    Citation parsing and the not-found decision.

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

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    re - standard library; regular expressions for citation markers and the not-found reply
    collections.abc - standard library; Sequence type for the list of hits
    dataclasses - standard library; the Citation data class and its dictionary form
    common.types - project; the Hit type for retrieved passages
    rag.prompts - project; the NOT_FOUND reply text

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Class name: Citation
    Purpose: One cited passage, as returned by the API.
    """

    source: str
    page: int | None
    section: str
    chunk_id: str
    label: str

    def to_dict(self) -> dict:
        """
        Function name: Citation.to_dict
        Purpose: Convert the citation to a plain dictionary for the JSON response.
        Parameters:
            None
        Returns:
            dict: The citation fields as a dictionary.
        """
        return asdict(self)


def evidence_is_weak(hits: Sequence[Hit], threshold: float) -> bool:
    """
    Function name: evidence_is_weak
    Purpose: Decide whether the retrieved evidence is too weak to answer. Hits without a
             re-ranker score (dense, bm25 and hybrid modes) are never judged weak here; the
             prompt's own not-found instruction still applies.
    Parameters:
        hits (Sequence[Hit]): The retrieved passages, best first.
        threshold (float): The lowest acceptable best re-ranker score.
    Returns:
        bool: True when nothing was retrieved or the best re-ranker score is below threshold.
    """
    if not hits:
        return True
    rerank = [h.scores["rerank"] for h in hits if "rerank" in h.scores]
    return bool(rerank) and max(rerank) < threshold


def is_not_found(answer: str) -> bool:
    """
    Function name: is_not_found
    Purpose: Check whether the model used the not-found reply, in exact or close wording.
    Parameters:
        answer (str): The raw answer text from the model.
    Returns:
        bool: True when the answer is the not-found reply.
    """
    return bool(_NOT_FOUND_RE.search(answer)) or answer.strip() == NOT_FOUND


def resolve_citations(answer: str, hits: Sequence[Hit]) -> tuple[str, list[Citation]]:
    """
    Function name: resolve_citations
    Purpose: Replace numeric and label citation markers with "[source, page]" labels and
             collect the unique citations. Markers that match no passage are dropped.
    Parameters:
        answer (str): The raw answer text from the model.
        hits (Sequence[Hit]): The passages given to the model, in prompt order.
    Returns:
        tuple[str, list[Citation]]: The display text and the unique citations, in first-use
                                    order.
    """
    cited: dict[str, Citation] = {}

    def cite(hit: Hit) -> str:
        """
        Function name: cite
        Purpose: Record a hit as cited (once per chunk) and return its display label.
        Parameters:
            hit (Hit): The cited passage.
        Returns:
            str: The label in square brackets, for example "[source.pdf, p. 3]".
        """
        c = hit.chunk
        cited.setdefault(c.chunk_id, Citation(c.source, c.page, c.section, c.chunk_id, c.label()))
        return f"[{c.label()}]"

    def numeric(m: re.Match) -> str:
        """
        Function name: numeric
        Purpose: Replacement for a numeric marker such as [1] or [2, 3]; numbers outside the
                 passage range are dropped, not invented.
        Parameters:
            m (re.Match): The match of the numeric marker pattern.
        Returns:
            str: The joined labels of the valid passage numbers, or an empty string.
        """
        labels = []
        for n in re.split(r"[,;]", m.group(1)):
            i = int(n.strip())
            if 1 <= i <= len(hits):
                labels.append(cite(hits[i - 1]))
        return "".join(labels)  # out-of-range numbers are dropped, not invented

    def label(m: re.Match) -> str:
        """
        Function name: label
        Purpose: Replacement for a label marker such as [source.pdf, p. 3]; it matches the
                 file name without case and the page when both sides have one.
        Parameters:
            m (re.Match): The match of the label marker pattern.
        Returns:
            str: The label of the first matching passage, or an empty string when none
                 matches.
        """
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
    """
    Function name: strip_citations
    Purpose: Produce the text to speak by removing every bracketed marker and tidying the
             spacing.
    Parameters:
        text (str): Text that may contain citation markers.
    Returns:
        str: The text without any bracketed content.
    """
    text = _ANY_BRACKET.sub("", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()
