"""
File: ingest/loader.py

Description:
    Document loading: PDFs (one Document per page), Markdown and text files.

    Each Document carries Markdown text so that ingest.chunking can split every format the
    same way (LlamaIndex MarkdownNodeParser). PDFs and plain text have no Markdown markup,
    so short heading-like lines are promoted to "## " headings first.

Flow:
    ingest.build.build_index()
        -> load_documents(docs_dir)
            -> _load_pdf(path)      # pypdf, page text -> heading-promoted Markdown
            -> _load_text(path)     # .md as-is; .txt with heading-like lines promoted
        -> returns llama_index Document objects with metadata {source, page}

Libraries used:
    logging - standard library; logs each loaded source file
    re - standard library; sentence-end pattern for the heading test
    pathlib - standard library; Path for the documents folder and files
    llama_index.core - third party; Document record
    pypdf - third party; PdfReader to extract PDF page text

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
from pathlib import Path

from llama_index.core import Document
from pypdf import PdfReader

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}
_SENTENCE_END = re.compile(r"[.!?:;,]$")


def load_documents(docs_dir: Path) -> list[Document]:
    """
    Function name: load_documents
    Purpose: Load every supported file under docs_dir, including sub-folders. Files are
             sorted by path so that ingestion is deterministic.
    Parameters:
        docs_dir (Path): folder that holds the source documents
    Returns:
        list[Document]: one Document per PDF page with text, and one per Markdown or text file
    """
    docs: list[Document] = []
    for path in sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED):
        loaded = _load_pdf(path) if path.suffix.lower() == ".pdf" else _load_text(path)
        log.info("loaded source=%s documents=%d", path.name, len(loaded))
        docs.extend(loaded)
    return docs


def _load_pdf(path: Path) -> list[Document]:
    """
    Function name: _load_pdf
    Purpose: Create one Document per non-empty PDF page, with heading-like lines promoted to
             Markdown headings. Page numbers are 1-based.
    Parameters:
        path (Path): PDF file to read
    Returns:
        list[Document]: page documents with metadata {source: file name, page: page number}
    """
    out = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(
                Document(
                    text=promote_headings(text),
                    metadata={"source": path.name, "page": i},
                )
            )
    return out


def _load_text(path: Path) -> list[Document]:
    """
    Function name: _load_text
    Purpose: Load a Markdown or text file as one Document. Markdown is used as it is; plain
             text (.txt) has its heading-like lines promoted to Markdown headings.
    Parameters:
        path (Path): Markdown or text file to read (UTF-8)
    Returns:
        list[Document]: a single document with metadata {source: file name, page: None}
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".txt":
        text = promote_headings(text)
    return [Document(text=text, metadata={"source": path.name, "page": None})]


def is_heading_line(line: str) -> bool:
    """
    Function name: is_heading_line
    Purpose: Heuristic heading test for text without markup: the line is short (at most 70
             characters and 10 words), starts with a capital letter or digit, and has no
             end punctuation. It is only applied at a paragraph start (see
             promote_headings), where a body line would either be a full-width wrapped line
             or a short line ending in a period.
    Parameters:
        line (str): one line of text
    Returns:
        bool: True when the line looks like a heading
    """
    s = line.strip()
    if not s or s.startswith("#") or len(s) > 70 or _SENTENCE_END.search(s):
        return False
    return len(s.split()) <= 10 and (s[0].isupper() or s[0].isdigit())


def promote_headings(text: str) -> str:
    """
    Function name: promote_headings
    Purpose: Turn heading-like lines into "## " Markdown headings and join wrapped PDF lines
             into one line per paragraph, with a blank line between blocks.
    Parameters:
        text (str): plain text from a PDF page or a .txt file
    Returns:
        str: Markdown text that ends with a single newline
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
        """
        Function name: flush
        Purpose: Write the current paragraph to the output as one line followed by a blank
                 line, then clear it. Does nothing when the paragraph is empty.
        Parameters:
            None
        Returns:
            None
        """
        if para:
            out.append(" ".join(para))
            out.append("")
            para.clear()

    for i, ln in enumerate(lines):
        nxt = next((x.strip() for x in lines[i + 1 :] if x.strip()), "")
        if not ln.strip():
            flush()
        elif (
            is_heading_line(ln)
            and (not para or para[-1].endswith((".", "!", "?")))
            # a wrapped sentence continues in lower case; a heading is followed by a new one
            and not nxt[:1].islower()
        ):
            # PDF text often has no blank line between a paragraph and the next heading.
            flush()
            out.append(f"## {ln.strip()}")
            out.append("")
        else:
            para.append(ln.strip())
    flush()
    return "\n".join(out).strip() + "\n"
