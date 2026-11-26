"""
File: rag/prompts.py

Description:
    Prompt builders: the grounded answer prompt and the spoken-question rewrite prompt.

    The model cites passage numbers ([1], [2]) rather than file names: small local models
    copy a number reliably but often mangle long file names. rag.citations maps the numbers
    back to "[source, page]" labels for display.

Flow:
    rag.pipeline.AnswerPipeline.stream_answer()
        -> build_answer_messages(question, hits, few_shot)
            -> format_passages(hits)        # "[1] (source, p. N) text"
    rag.rewrite.rewrite_query()
        -> build_rewrite_messages(utterance)

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    collections.abc - standard library; Sequence type for the list of hits
    common.types - project; the Hit type for retrieved passages

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

from collections.abc import Sequence

from common.types import Hit

# Prompt strings keep long lines on purpose: the model sees them verbatim.
# ruff: noqa: E501

NOT_FOUND = "I could not find that in the documents."

ANSWER_SYSTEM = f"""You are a voice assistant that answers questions about company documents.
Rules:
- Use only the numbered passages. Do not use outside knowledge.
- After every sentence that states a fact, cite the passage number in square brackets, for example [2].
- Answer in one to three short sentences that sound natural when read aloud. No lists, no markdown.
- If the passages do not answer the question, reply with exactly: {NOT_FOUND}"""

# Fictional, domain-neutral examples so the few-shot turns never leak facts into answers.
_FEW_SHOT: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Passages:\n"
            "[1] (zenda_library_rules.pdf, p. 2) Borrowing\n"
            "Members may borrow up to 8 books at a time. The loan period is 21 days.\n"
            "[2] (zenda_library_rules.pdf, p. 3) Late returns\n"
            "A late fee of 20 cents per book per day applies, up to 5 dollars per book.\n\n"
            "Question: How long can I keep a book, and what if I return it late?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "You can keep a book for 21 days [1]. After that there is a late fee of 20 cents "
            "per book per day, capped at 5 dollars per book [2]."
        ),
    },
    {
        "role": "user",
        "content": (
            "Passages:\n"
            "[1] (zenda_library_rules.pdf, p. 2) Borrowing\n"
            "Members may borrow up to 8 books at a time. The loan period is 21 days.\n\n"
            "Question: Does the library have a cafe?"
        ),
    },
    {"role": "assistant", "content": NOT_FOUND},
]

REWRITE_SYSTEM = """Rewrite the user's spoken question as one short, clear search question.
Remove filler words, greetings and repetitions. Keep every specific detail (names, amounts, product names).
Reply with the rewritten question only."""


def format_passages(hits: Sequence[Hit]) -> str:
    """
    Function name: format_passages
    Purpose: Build the numbered passage block for the prompt; numbering starts at 1 and
             matches the citation markers.
    Parameters:
        hits (Sequence[Hit]): The passages to show the model, in order.
    Returns:
        str: One line per passage in the form "[n] (label) text".
    """
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] ({h.chunk.label()}) {h.chunk.text.strip()}")
    return "\n".join(lines)


def build_answer_messages(
    question: str, hits: Sequence[Hit], few_shot: bool = True
) -> list[dict[str, str]]:
    """
    Function name: build_answer_messages
    Purpose: Build the chat messages for a grounded, cited answer: the system rules, the
             optional few-shot examples and the user turn with the passages and question.
    Parameters:
        question (str): The (rewritten) question to answer.
        hits (Sequence[Hit]): The passages the answer may use.
        few_shot (bool): Whether to include the fictional few-shot example turns.
    Returns:
        list[dict[str, str]]: The chat messages with "role" and "content" keys.
    """
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    if few_shot:
        messages.extend(_FEW_SHOT)
    messages.append(
        {
            "role": "user",
            "content": f"Passages:\n{format_passages(hits)}\n\nQuestion: {question.strip()}",
        }
    )
    return messages


def build_rewrite_messages(utterance: str) -> list[dict[str, str]]:
    """
    Function name: build_rewrite_messages
    Purpose: Build the chat messages for the spoken-question clean-up call, with one worked
             example before the real utterance.
    Parameters:
        utterance (str): The spoken or typed question to clean up.
    Returns:
        list[dict[str, str]]: The chat messages with "role" and "content" keys.
    """
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": "uh hi, so like, what's the, um, late fee on books?"},
        {"role": "assistant", "content": "What is the late fee for books?"},
        {"role": "user", "content": utterance.strip()},
    ]
