"""Spoken-question clean-up: turn "uh, what's the, like, refund rule" into a search query.

Flow:
    rag.pipeline.AnswerPipeline.stream_answer()
        -> prepare_query(llm, question, mode)
            -> strip_fillers(question)          # deterministic, always applied
            -> needs_rewrite(question)          # mode "auto": only conversational input
            -> rewrite_query(llm, question)     # one short LLM call (max ~40 tokens)

The LLM call costs a few hundred milliseconds on a local 3B model, so "auto" skips it for
questions that are already clean (typed questions usually are).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rag.prompts import build_rewrite_messages

if TYPE_CHECKING:
    from rag.llm import LLMClient

log = logging.getLogger(__name__)

# "like" only counts as a filler when set off by commas or after "so"/"um", never in
# "structures like a garage".
_FILLERS = re.compile(
    r"\b(?:uh+|um+|erm|hmm+|you know|i mean|so like|like(?=,)|okay so|ok so|hey|hi|hello|"
    r"please|basically)\b,?",
    re.IGNORECASE,
)
_CHATTY = re.compile(
    r"\b(?:uh+|um+|erm|you know|i mean|so like|like(?=,)|kinda|sort of|i was wondering|"
    r"can you tell me|could you tell me|i wanna know|i want to know)\b",
    re.IGNORECASE,
)


def strip_fillers(text: str) -> str:
    """Remove filler words and collapse whitespace."""
    out = _FILLERS.sub(" ", text)
    out = re.sub(r"\s+([,?.!])", r"\1", out)
    out = re.sub(r"^[\s,]+", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def needs_rewrite(text: str) -> bool:
    """Heuristic: conversational fillers, or a long run-on utterance."""
    return bool(_CHATTY.search(text)) or len(text.split()) > 25


def rewrite_query(llm: LLMClient, text: str, max_tokens: int = 40) -> str:
    """One LLM call; falls back to the filler-stripped text on any failure."""
    try:
        out = llm.complete(build_rewrite_messages(text), temperature=0.0, max_tokens=max_tokens)
    except Exception as e:  # rewrite is an optimisation; never fail the request over it
        log.warning("rewrite failed, using original: %s", e)
        return strip_fillers(text)
    out = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
    return out or strip_fillers(text)


def prepare_query(llm: LLMClient, text: str, mode: str = "auto", max_tokens: int = 40) -> str:
    """Search query for text according to the rewrite mode (auto | always | never)."""
    if mode == "never":
        return text.strip()
    if mode == "always" or needs_rewrite(text):
        return rewrite_query(llm, text, max_tokens)
    return strip_fillers(text)
