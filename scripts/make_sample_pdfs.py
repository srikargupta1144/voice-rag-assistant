"""Render the fictional sample PDFs in data/docs from their Markdown sources.

Flow:
    uv run python -m scripts.make_sample_pdfs
        -> for each data/pdf_sources/*.md: render_pdf(md_path, data/docs/<name>.pdf)
            -> "# " title on page 1, every "## " section starts a new page

One section per page keeps page numbers stable, so the golden questions in
eval/questions.jsonl can name the page that holds each answer. The generated PDFs are
committed; this script only needs re-running after editing a source file.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO / "data" / "pdf_sources"
OUT = REPO / "data" / "docs"


def render_pdf(md_path: Path, pdf_path: Path) -> int:
    """Write pdf_path from a simple Markdown file (headings + paragraphs); returns pages."""
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
    OUT.mkdir(parents=True, exist_ok=True)
    for md in sorted(SOURCES.glob("*.md")):
        target = OUT / f"{md.stem}.pdf"
        print(f"{target.name}: {render_pdf(md, target)} pages")


if __name__ == "__main__":
    main()
