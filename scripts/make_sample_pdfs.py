"""
File: scripts/make_sample_pdfs.py

Description:
    Render the fictional sample PDFs in data/docs from their Markdown sources.

    One section per page keeps page numbers stable, so the golden questions in
    eval/questions.jsonl can name the page that holds each answer. The generated PDFs are
    committed; this script only needs re-running after editing a source file.

Flow:
    uv run python -m scripts.make_sample_pdfs
        -> for each data/pdf_sources/*.md: render_pdf(md_path, data/docs/<name>.pdf)
            -> "# " title on page 1, every "## " section starts a new page

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    pathlib - standard library; source and output paths
    fpdf - third party; writes the PDF pages (fpdf2)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO / "data" / "pdf_sources"
OUT = REPO / "data" / "docs"


def render_pdf(md_path: Path, pdf_path: Path) -> int:
    """
    Function name: render_pdf
    Purpose: Write pdf_path from a simple Markdown file of headings and paragraphs; the
             "# " title goes on page 1 and every "## " section after the first starts a new
             page.
    Parameters:
        md_path (Path): The Markdown source file.
        pdf_path (Path): The PDF file to write.
    Returns:
        int: The number of pages written.
    """
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(22, 22, 22)
    pdf.add_page()
    first_section = True
    for block in md_path.read_text(encoding="utf-8").split("\n\n"):
        block = " ".join(block.split())
        if not block:
            continue
        if block.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, block[2:])
            pdf.ln(4)
        elif block.startswith("## "):
            if not first_section:
                pdf.add_page()
            first_section = False
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, block[3:])
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, block)
            pdf.ln(3)
    pdf.output(str(pdf_path))
    return pdf.pages_count


def main() -> None:
    """
    Function name: main
    Purpose: Render every Markdown file in data/pdf_sources to a PDF in data/docs and print
             the page count of each.
    Parameters:
        None
    Returns:
        None
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for md in sorted(SOURCES.glob("*.md")):
        target = OUT / f"{md.stem}.pdf"
        print(f"{target.name}: {render_pdf(md, target)} pages")


if __name__ == "__main__":
    main()
