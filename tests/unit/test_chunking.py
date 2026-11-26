"""Chunking and loading: heading splits, token budget, overlap, provenance."""

import pytest
from llama_index.core import Document

from ingest.chunking import chunk_documents, count_tokens, tokenize
from ingest.loader import is_heading_line, load_documents, promote_headings


def _doc(text, source="a.md", page=None):
    return Document(text=text, metadata={"source": source, "page": page})


def test_splits_at_every_heading_and_keeps_heading_text():
    md = "# Title\n\nIntro text.\n\n## Fees\n\nThe fee is 8 dollars.\n\n## Wires\n\nWires cost 45."
    chunks = chunk_documents([_doc(md)])
    assert [c.section for c in chunks] == ["Title", "Fees", "Wires"]
    assert chunks[1].text.startswith("Fees\n") and "8 dollars" in chunks[1].text
    assert all("#" not in c.text for c in chunks)


def test_provenance_and_unique_ids():
    chunks = chunk_documents([_doc("## A\n\nOne.", "p.pdf", 3), _doc("## B\n\nTwo.", "p.pdf", 4)])
    assert [(c.source, c.page) for c in chunks] == [("p.pdf", 3), ("p.pdf", 4)]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].label() == "p.pdf, p. 3"


def test_long_section_is_split_within_budget_with_overlap():
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
    with pytest.raises(ValueError):
        chunk_documents([_doc("## A\n\ntext")], chunk_tokens=50, overlap_tokens=50)


def test_tokenize_counts_words_and_punctuation():
    assert tokenize("Fee: 8 dollars.") == ["Fee", ":", "8", "dollars", "."]


def test_heading_heuristic():
    assert is_heading_line("About this policy")
    assert is_heading_line("WHAT INFORMATION KOVA COLLECTS")
    assert not is_heading_line("The fee is 8 dollars.")
    assert not is_heading_line("lower case start")
    assert not is_heading_line("X" * 90)


def test_promote_headings_on_pdf_like_text():
    text = "Title Line\nFirst sentence wraps\nonto a second line.\nNext Section\nBody text here."
    out = promote_headings(text)
    assert "## Title Line" in out and "## Next Section" in out
    assert "First sentence wraps onto a second line." in out


def test_load_documents_md_txt_pdf(tmp_path):
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
