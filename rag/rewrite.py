"""
File: rag/rewrite.py

Description:
    Spoken-question clean-up: turn "uh, what's the, like, refund rule" into a search query.

    The LLM call costs a few hundred milliseconds on a local 3B model, so "auto" skips it
    for questions that are already clean (typed questions usually are).

Flow:
    rag.pipeline.AnswerPipeline.stream_answer()
        -> prepare_query(llm, question, mode)
            -> strip_fillers(question)          # deterministic, always applied
            -> needs_rewrite(question)          # mode "auto": only conversational input
            -> rewrite_query(llm, question)     # one short LLM call (max ~40 tokens)

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    logging - standard library; warns when the rewrite call fails
    re - standard library; filler-word and chatty-phrase patterns
    typing - standard library; TYPE_CHECKING guard for the LLM client type
    rag.prompts - project; builds the rewrite prompt messages
    rag.llm - project; the LLMClient type (type checking only)

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
    """
    Function name: strip_fillers
    Purpose: Remove filler words and greetings, tidy the spacing before punctuation and
             collapse whitespace.
    Parameters:
        text (str): The spoken or typed question.
    Returns:
        str: The question without filler words.
    """
    out = _FILLERS.sub(" ", text)
    out = re.sub(r"\s+([,?.!])", r"\1", out)
    out = re.sub(r"^[\s,]+", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def needs_rewrite(text: str) -> bool:
    """
    Function name: needs_rewrite
    Purpose: Heuristic check for input that needs the LLM rewrite: conversational fillers,
             or a long run-on utterance of more than 25 words.
    Parameters:
        text (str): The spoken or typed question.
    Returns:
        bool: True when the question should be rewritten by the LLM.
    """
    return bool(_CHATTY.search(text)) or len(text.split()) > 25


def rewrite_query(llm: LLMClient, text: str, max_tokens: int = 40) -> str:
    """
    Function name: rewrite_query
    Purpose: Rewrite the question with one LLM call, keeping only the first line of the
             reply; falls back to the filler-stripped text on any failure or empty reply.
    Parameters:
        llm (LLMClient): The LLM client used for the rewrite.
        text (str): The spoken or typed question.
        max_tokens (int): The token limit for the rewrite reply.
    Returns:
        str: The rewritten search question.
    """
    try:
        out = llm.complete(build_rewrite_messages(text), temperature=0.0, max_tokens=max_tokens)
    except Exception as e:  # rewrite is an optimisation; never fail the request over it
        log.warning("rewrite failed, using original: %s", e)
        return strip_fillers(text)
    out = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
    return out or strip_fillers(text)


def prepare_query(llm: LLMClient, text: str, mode: str = "auto", max_tokens: int = 40) -> str:
    """
    Function name: prepare_query
    Purpose: Build the search query for the text according to the rewrite mode: "never"
             only trims, "always" calls the LLM, "auto" calls it only for conversational
             input and otherwise strips fillers.
    Parameters:
        llm (LLMClient): The LLM client used when a rewrite is needed.
        text (str): The spoken or typed question.
        mode (str): The rewrite mode: "auto", "always" or "never".
        max_tokens (int): The token limit for the rewrite reply.
    Returns:
        str: The search query.
    """
    if mode == "never":
        return text.strip()
    if mode == "always" or needs_rewrite(text):
        return rewrite_query(llm, text, max_tokens)
    return strip_fillers(text)
