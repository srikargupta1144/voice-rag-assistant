"""Document loading: PDFs (one Document per page), Markdown and text files.

Flow:
    ingest.build.build_index()
        -> load_documents(docs_dir)
            -> _load_pdf(path)      # pypdf, page text -> heading-promoted Markdown
            -> _load_text(path)     # .md as-is; .txt with heading-like lines promoted
        -> returns llama_index Document objects with metadata {source, page}

Each Document carries Markdown text so that ingest.chunking can split every format the
same way (LlamaIndex MarkdownNodeParser). PDFs and plain text have no Markdown markup,
so short heading-like lines are promoted to "## " headings first.
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
    """Load every supported file under docs_dir (sorted, so ingestion is deterministic)."""
    docs: list[Document] = []
    for path in sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED):
        loaded = _load_pdf(path) if path.suffix.lower() == ".pdf" else _load_text(path)
        log.info("loaded source=%s documents=%d", path.name, len(loaded))
        docs.extend(loaded)
    return docs


def _load_pdf(path: Path) -> list[Document]:
    """One Document per non-empty page; page numbers are 1-based."""
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
    """Markdown is used as-is; plain text gets heading-like lines promoted."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".txt":
        text = promote_headings(text)
    return [Document(text=text, metadata={"source": path.name, "page": None})]


def is_heading_line(line: str) -> bool:
    """Heuristic heading test for text without markup: short, capitalised, no end punctuation.

    Only applied at a paragraph start (see promote_headings), where a body line would either
    be a full-width wrapped line or a short line ending in a period.
    """
    s = line.strip()
    if not s or s.startswith("#") or len(s) > 70 or _SENTENCE_END.search(s):
        return False
    return len(s.split()) <= 10 and (s[0].isupper() or s[0].isdigit())


def promote_headings(text: str) -> str:
    """Turn heading-like lines into '## ' Markdown headings; collapse PDF line wraps."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    para: list[str] = []

    def flush() -> None:
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
