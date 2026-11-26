"""
File: eval/dataset.py

Description:
    Golden question set and relevance matching.

    A chunk is relevant to a question when its source file matches, its page matches (when
    the label names one) and it contains the label's key phrase. Matching on content rather
    than chunk ids keeps the labels valid when chunking settings change.

Flow:
    eval.retrieval_eval / eval.answer_eval
        -> load_questions(path)            # eval/questions.jsonl
        -> is_relevant(chunk, question)    # source + page + key phrase match

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    json - standard library; parses each line of the JSONL golden set
    dataclasses - standard library; defines the frozen Question record
    pathlib - standard library; builds the repository, question and result paths
    common.types - project; the Chunk type that relevance is checked against

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
from dataclasses import dataclass, field
from pathlib import Path

from common.types import Chunk

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "eval" / "questions.jsonl"
RESULTS = REPO / "eval" / "results"


@dataclass(frozen=True)
class Question:
    """
    Class name: Question
    Purpose: One golden-set question with its id, text, whether the documents can answer it,
        an optional reference answer and the relevance labels used to judge retrieved chunks.
    """

    id: str
    question: str
    answerable: bool
    reference: str = ""
    relevant: list[dict] = field(default_factory=list)


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    """
    Function name: load_questions
    Purpose: Read the JSONL golden set and build one Question per non-blank line.
    Parameters:
        path (Path): Path of the JSONL file; defaults to eval/questions.jsonl.
    Returns:
        list[Question]: The questions in file order.
    """
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def _norm(s: str) -> str:
    """
    Function name: _norm
    Purpose: Normalise text for matching by lower-casing it and collapsing all whitespace to
        single spaces.
    Parameters:
        s (str): Text to normalise.
    Returns:
        str: The normalised text.
    """
    return " ".join(s.lower().split())


def is_relevant(chunk: Chunk, q: Question) -> bool:
    """
    Function name: is_relevant
    Purpose: Decide whether a chunk satisfies any of the question's relevance labels, that is,
        the source matches, the page matches when the label names one, and the normalised
        chunk text contains the label's key phrase.
    Parameters:
        chunk (Chunk): Retrieved chunk to check.
        q (Question): Question whose relevance labels are used.
    Returns:
        bool: True when at least one label matches the chunk, otherwise False.
    """
    for rel in q.relevant:
        if chunk.source != rel["source"]:
            continue
        if rel.get("page") is not None and chunk.page != rel["page"]:
            continue
        if _norm(rel["contains"]) in _norm(chunk.text):
            return True
    return False
