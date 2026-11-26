"""
File: tests/unit/test_chunking.py

Description:
    Unit tests for chunking and loading: heading splits, token budget, overlap and
    provenance.

Libraries used:
    pytest - third party; checks that invalid settings raise errors
    llama_index.core - third party; the Document type used as chunker input
    fpdf - third party; writes a two-page test PDF (imported inside the PDF test)
    ingest.chunking - project; chunking and token counting under test
    ingest.loader - project; document loading and heading detection under test

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import pytest
from llama_index.core import Document

from ingest.chunking import chunk_documents, count_tokens, tokenize
from ingest.loader import is_heading_line, load_documents, promote_headings


def _doc(text, source="a.md", page=None):
    """
    Function name: _doc
    Purpose: Build a Document with source and page metadata for the chunker.
    Parameters:
        text (str): Document text.
        source (str): Source file name stored in the metadata.
        page (int | None): Page number stored in the metadata; None for non-paged files.
    Returns:
        Document: The test document.
    """
    return Document(text=text, metadata={"source": source, "page": page})


def test_splits_at_every_heading_and_keeps_heading_text():
    """
    Function name: test_splits_at_every_heading_and_keeps_heading_text
    Purpose: Check that Markdown is split at every heading, that each chunk records its
        section and starts with the heading text, and that the # markers are removed.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    md = "# Title\n\nIntro text.\n\n## Fees\n\nThe fee is 8 dollars.\n\n## Wires\n\nWires cost 45."
    chunks = chunk_documents([_doc(md)])
    assert [c.section for c in chunks] == ["Title", "Fees", "Wires"]
    assert chunks[1].text.startswith("Fees\n") and "8 dollars" in chunks[1].text
    assert all("#" not in c.text for c in chunks)


def test_provenance_and_unique_ids():
    """
    Function name: test_provenance_and_unique_ids
    Purpose: Check that chunks keep their source and page, have unique ids, and produce a
        label with the page number.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    chunks = chunk_documents([_doc("## A\n\nOne.", "p.pdf", 3), _doc("## B\n\nTwo.", "p.pdf", 4)])
    assert [(c.source, c.page) for c in chunks] == [("p.pdf", 3), ("p.pdf", 4)]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].label() == "p.pdf, p. 3"


def test_long_section_is_split_within_budget_with_overlap():
    """
    Function name: test_long_section_is_split_within_budget_with_overlap
    Purpose: Check that a long section is split into several pieces that each repeat the
        heading, stay near the token budget, and overlap with the next piece.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    sentences = " ".join(f"Sentence number {i} talks about premium payments." for i in range(150))
    chunks = chunk_documents([_doc(f"## Long\n\n{sentences}")], chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 5
    # every piece repeats the heading and stays near the budget (+ heading tokens)
    assert all(c.text.startswith("Long\n") for c in chunks)
    assert max(count_tokens(c.text) for c in chunks) <= 100 + 5
    # overlap: consecutive pieces share some words
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert set(first[-10:]) & set(second[:30])


def test_overlap_must_be_smaller_than_chunk():
    """
    Function name: test_overlap_must_be_smaller_than_chunk
    Purpose: Check that an overlap equal to the chunk size raises ValueError.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    with pytest.raises(ValueError):
        chunk_documents([_doc("## A\n\ntext")], chunk_tokens=50, overlap_tokens=50)


def test_tokenize_counts_words_and_punctuation():
    """
    Function name: test_tokenize_counts_words_and_punctuation
    Purpose: Check that the tokenizer returns words, numbers and punctuation marks as
        separate tokens.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert tokenize("Fee: 8 dollars.") == ["Fee", ":", "8", "dollars", "."]


def test_heading_heuristic():
    """
    Function name: test_heading_heuristic
    Purpose: Check the heading rule: short title-case or upper-case lines are headings, while
        sentences, lower-case lines and very long lines are not.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert is_heading_line("About this policy")
    assert is_heading_line("WHAT INFORMATION KOVA COLLECTS")
    assert not is_heading_line("The fee is 8 dollars.")
    assert not is_heading_line("lower case start")
    assert not is_heading_line("X" * 90)


def test_promote_headings_on_pdf_like_text():
    """
    Function name: test_promote_headings_on_pdf_like_text
    Purpose: Check that heading lines in PDF-like text become Markdown headings and that a
        sentence wrapped over two lines is joined.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    text = "Title Line\nFirst sentence wraps\nonto a second line.\nNext Section\nBody text here."
    out = promote_headings(text)
    assert "## Title Line" in out and "## Next Section" in out
    assert "First sentence wraps onto a second line." in out


def test_load_documents_md_txt_pdf(tmp_path):
    """
    Function name: test_load_documents_md_txt_pdf
    Purpose: Check that Markdown, text and PDF files are loaded in name order with one
        document per PDF page, that unsupported files are skipped, and that headings in a
        text file are promoted.
    Parameters:
        tmp_path (Path): pytest temporary directory that holds the test files.
    Returns:
        None: The test passes or fails through assertions.
    """
    from fpdf import FPDF

    (tmp_path / "a.md").write_text("# A\n\n## Sec\n\nBody.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("HEADING ONE\n\nSome body text.", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("x,y", encoding="utf-8")
    pdf = FPDF()
    for body in ("Page one text.", "Page two text."):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 6, body)
    pdf.output(str(tmp_path / "c.pdf"))
    docs = load_documents(tmp_path)
    meta = [(d.metadata["source"], d.metadata["page"]) for d in docs]
    assert meta == [("a.md", None), ("b.txt", None), ("c.pdf", 1), ("c.pdf", 2)]
    assert "## HEADING ONE" in docs[1].text
