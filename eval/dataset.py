"""Golden question set and relevance matching.

Flow:
    eval.retrieval_eval / eval.answer_eval
        -> load_questions(path)            # eval/questions.jsonl
        -> is_relevant(chunk, question)    # source + page + key phrase match

A chunk is relevant to a question when its source file matches, its page matches (when
the label names one) and it contains the label's key phrase. Matching on content rather
than chunk ids keeps the labels valid when chunking settings change.
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
    id: str
    question: str
    answerable: bool
    reference: str = ""
    relevant: list[dict] = field(default_factory=list)


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    """Read the JSONL golden set."""
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def is_relevant(chunk: Chunk, q: Question) -> bool:
    """True when chunk satisfies any of the question's relevance labels."""
    for rel in q.relevant:
        if chunk.source != rel["source"]:
            continue
        if rel.get("page") is not None and chunk.page != rel["page"]:
            continue
        if _norm(rel["contains"]) in _norm(chunk.text):
            return True
    return False
