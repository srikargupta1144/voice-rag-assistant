"""Heading-aware chunking: one chunk per Markdown section, long sections split with overlap.

Flow:
    ingest.build.build_index()
        -> chunk_documents(documents, chunk_tokens, overlap_tokens)
            -> MarkdownNodeParser.get_nodes_from_documents()   # split at every heading
            -> _section_title(node)                            # nearest heading text
            -> SentenceSplitter.split_text()                   # only sections > chunk_tokens
        -> list[common.types.Chunk] with {source, page, section}

Token counts use count_tokens(), a word-and-punctuation approximation of subword tokens.
It avoids downloading a tokenizer at ingest time and is within about 20 percent of the
bge WordPiece count on English prose, which is close enough for a ~500 token budget.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from common.types import Chunk

_TOKEN = re.compile(r"\w+|[^\w\s]")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")


def tokenize(text: str) -> list[str]:
    """Word and punctuation tokens (the unit for chunk_tokens / overlap_tokens)."""
    return _TOKEN.findall(text)


def count_tokens(text: str) -> int:
    """Approximate token count of text."""
    return len(tokenize(text))


def chunk_documents(
    documents: Sequence[Document], chunk_tokens: int = 500, overlap_tokens: int = 50
) -> list[Chunk]:
    """Split documents into heading-scoped chunks no longer than about chunk_tokens."""
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    md_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=False)
    splitter = SentenceSplitter(
        chunk_size=chunk_tokens, chunk_overlap=overlap_tokens, tokenizer=tokenize
    )
    chunks: list[Chunk] = []
    counters: dict[tuple[str, int | None], int] = {}
    for node in md_parser.get_nodes_from_documents(list(documents)):
        source = node.metadata.get("source", "unknown")
        page = node.metadata.get("page")
        section = _section_title(node.get_content(), node.metadata.get("header_path", ""))
        body = _strip_heading_line(node.get_content()).strip()
        if not body:
            continue  # heading with no text under it
        pieces = [body] if count_tokens(body) <= chunk_tokens else splitter.split_text(body)
        for piece in pieces:
            key = (source, page)
            n = counters.get(key, 0)
            counters[key] = n + 1
            # The heading is repeated in every piece so each chunk stands on its own.
            text = f"{section}\n{piece.strip()}" if section else piece.strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{source}:{page or 0}:{n}",
                    text=text,
                    source=source,
                    page=page,
                    section=section,
                )
            )
    return chunks


def _section_title(content: str, header_path: str) -> str:
    """Heading of this node: its own first-line heading, else the last header_path entry."""
    first = content.lstrip().splitlines()[0] if content.strip() else ""
    m = _HEADING.match(first)
    if m:
        return m.group(1).strip()
    parts = [p for p in header_path.split("/") if p]
    return parts[-1] if parts else ""


def _strip_heading_line(content: str) -> str:
    """Drop the leading Markdown heading line (it is re-added as plain text)."""
    lines = content.lstrip().splitlines()
    if lines and _HEADING.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines)
