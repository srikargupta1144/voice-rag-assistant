"""
File: eval/report.py

Description:
    Render measured results into the README between the RESULTS markers.

    Only numbers present in the result files are written; a missing file becomes a note.

Flow:
    uv run python -m eval.report
        -> eval/results/retrieval.json   (eval.retrieval_eval)
        -> eval/results/answers*.json    (eval.answer_eval; one row per variant tag)
        -> eval/results/e2e.json         (scripts.demo_e2e)
        -> render_markdown() -> README.md between <!-- RESULTS:START --> and <!-- RESULTS:END -->

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    json - standard library; reads the result files
    re - standard library; finds and replaces the RESULTS block in README.md
    pathlib - standard library; builds the repository, results and README paths

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
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "eval" / "results"
README = REPO / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(name: str) -> dict | None:
    """
    Function name: _load
    Purpose: Read one JSON result file from eval/results when it exists.
    Parameters:
        name (str): File name inside eval/results.
    Returns:
        dict | None: The parsed document, or None when the file does not exist.
    """
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt(v, digits: int = 3) -> str:
    """
    Function name: _fmt
    Purpose: Format a value for a Markdown table: floats to a fixed number of decimal places,
        None as "-", and any other value with str().
    Parameters:
        v (float | int | str | None): Value to format.
        digits (int): Decimal places for floats.
    Returns:
        str: The formatted value.
    """
    if v is None:
        return "-"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _ms(v) -> str:
    """
    Function name: _ms
    Purpose: Format a duration in milliseconds as "-" for None, "<1 ms" below one
        millisecond, seconds with one decimal place from 1000 ms, and whole milliseconds
        otherwise.
    Parameters:
        v (float | None): Duration in milliseconds.
    Returns:
        str: The formatted duration.
    """
    if v is None:
        return "-"
    if v < 1:
        return "<1 ms"
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def retrieval_section(r: dict | None) -> list[str]:
    """
    Function name: retrieval_section
    Purpose: Build the Markdown lines for the retrieval table (one row per mode) and the
        not-found gate sentence, or a note when the retrieval results are missing.
    Parameters:
        r (dict | None): Retrieval results document, or None when it was not generated.
    Returns:
        list[str]: Markdown lines for the section.
    """
    if not r:
        return ["Retrieval results not generated yet (`uv run python -m eval.retrieval_eval`)."]
    names = {
        "dense": "Dense only (FAISS, bge-small)",
        "bm25": "BM25 only",
        "hybrid": "Hybrid (dense + BM25, RRF)",
        "full": "Hybrid + cross-encoder re-rank",
    }
    lines = [
        f"Retrieval over {r['answerable']} answerable questions, {r['chunks']} chunks "
        f"(no LLM involved):",
        "",
        "| Retrieval | Recall@5 | MRR@10 | Median latency |",
        "| --- | --- | --- | --- |",
    ]
    for mode, m in r["modes"].items():
        lines.append(
            f"| {names.get(mode, mode)} | {_fmt(m['recall_at_5'])} | "
            f"{_fmt(m['mrr_at_10'])} | {_ms(m['median_latency_ms'])} |"
        )
    g = r.get("not_found_gate")
    if g:
        lines += [
            "",
            f"Not-found gate on the re-ranker score (threshold {g['threshold']}): "
            f"{_fmt(g['accuracy'])} of all {r['questions']} questions classified "
            f"correctly before any LLM call (lowest answerable score "
            f"{g['answerable_min_score']}, highest unanswerable score "
            f"{g['unanswerable_max_score']}).",
        ]
    return lines


def answers_section() -> list[str]:
    """
    Function name: answers_section
    Purpose: Build the Markdown lines for the answer-quality table, one row per
        eval/results/answers*.json file, plus a judge note taken from the first file; a note
        is returned when no answer results exist.
    Parameters:
        None
    Returns:
        list[str]: Markdown lines for the section.
    """
    files = sorted(RESULTS.glob("answers*.json"))
    if not files:
        return ["Answer-quality results not generated yet (`uv run python -m eval.answer_eval`)."]
    lines = [
        "| Variant | Faithfulness | Answer relevancy | Answered (answerable) "
        "| Declined (unanswerable) "
        "| Citation correct | Median first token | Median total |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        s = d["summary"]
        variant = (
            f"{d['llm_model']}, {'few-shot' if d['few_shot'] else 'zero-shot'}, "
            f"temp {d['temperature']}"
        )
        lines.append(
            f"| {variant} | {_fmt(s['faithfulness'])} | {_fmt(s['answer_relevancy'])} | "
            f"{_fmt(s['answered_rate'])} | {_fmt(s['unanswerable_declined'])} | "
            f"{_fmt(s['citation_correct'])} | {_ms(s['median_llm_first_token_ms'])} | "
            f"{_ms(s['median_total_ms'])} |"
        )
    d = json.loads(files[0].read_text(encoding="utf-8"))
    lines += [
        "",
        f"Judge: {d['judge_model']} through Ollama, over {d['questions']} questions. "
        "Faithfulness and relevancy are averaged over answered, answerable questions.",
    ]
    return lines


def latency_section(e: dict | None) -> list[str]:
    """
    Function name: latency_section
    Purpose: Build the Markdown lines for the per-stage latency table (audio and text
        questions) and the WebSocket streaming sentence, or a note when the end-to-end
        results are missing.
    Parameters:
        e (dict | None): End-to-end results document, or None when it was not generated.
    Returns:
        list[str]: Markdown lines for the section.
    """
    if not e:
        return ["End-to-end timings not generated yet (`uv run python -m scripts.demo_e2e`)."]
    a, t, w = e["audio"]["server_timings"], e["text"]["server_timings"], e["websocket"]
    rows = [
        ("Speech-to-text (faster-whisper small, int8)", a.get("stt_ms"), None),
        ("Query rewrite (LLM, only for casual speech)", a.get("rewrite_ms"), t.get("rewrite_ms")),
        ("Retrieval (hybrid + re-rank)", a.get("retrieval_ms"), t.get("retrieval_ms")),
        (
            "LLM first token (since request start)",
            a.get("llm_first_token_ms"),
            t.get("llm_first_token_ms"),
        ),
        ("LLM call (prompt evaluation + generation)", a.get("llm_ms"), t.get("llm_ms")),
        ("Text-to-speech (Piper, all sentences)", a.get("tts_ms"), None),
        ("Total server time", a.get("total_ms"), t.get("total_ms")),
    ]
    lines = [
        "| Stage | Audio question (/ask/audio) | Text question (/ask) |",
        "| --- | --- | --- |",
        *[f"| {n} | {_ms(x)} | {_ms(y)} |" for n, x, y in rows],
        "",
        f"Streaming (/ws/voice, same recording): first reply audio after "
        f"{_ms(w.get('client_first_audio_ms'))} measured at the client, full reply after "
        f"{_ms(w.get('client_total_ms'))}; {w.get('sentences_streamed')} sentence(s) streamed. "
        f"Word error rate of the transcript on the Piper-generated question: "
        f"{_fmt(e['audio'].get('wer'))}.",
    ]
    return lines


def render_markdown() -> str:
    """
    Function name: render_markdown
    Purpose: Join the retrieval, answers and latency sections into one Markdown block.
    Parameters:
        None
    Returns:
        str: The Markdown text placed between the RESULTS markers.
    """
    parts = [
        "### Retrieval",
        "",
        *retrieval_section(_load("retrieval.json")),
        "",
        "### Answers",
        "",
        *answers_section(),
        "",
        "### Latency (one run on the development laptop, CPU only)",
        "",
        *latency_section(_load("e2e.json")),
    ]
    return "\n".join(parts)


def main() -> None:
    """
    Function name: main
    Purpose: Replace the block between the RESULTS markers in README.md with freshly rendered
        Markdown and print it. Exits with an error when the markers are missing.
    Parameters:
        None
    Returns:
        None: README.md is rewritten and the Markdown is printed.
    """
    text = README.read_text(encoding="utf-8")
    block = f"{START}\n{render_markdown()}\n{END}"
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"README has no {START} ... {END} block")
    README.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    print(render_markdown())


if __name__ == "__main__":
    main()
