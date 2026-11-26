"""
File: eval/answer_eval.py

Description:
    Answer-quality evaluation with a local LLM judge (RAGAS-style metrics, no hosted API).

    The judge is the same local Ollama model unless --judge-model is given; a 3B judge is
    lenient and noisy, so treat these numbers as a smoke signal, not a benchmark.

Flow:
    uv run python -m eval.answer_eval [--limit N] [--no-few-shot] [--temperature T] [--tag NAME]
        -> rag.pipeline.build_pipeline(settings)
        -> per question: AnswerPipeline.answer(question)            # timings recorded
            answerable + answered:
                -> faithfulness(): answer split into sentences (statements); judge votes
                   YES / NO per statement against the passages the model saw;
                   score = supported / statements                           (RAGAS definition)
                -> answer_relevancy(): judge writes 3 questions the answer responds to;
                   score = mean cosine(bge(question), bge(generated))       (RAGAS definition)
                -> citation_correct: any cited chunk matches the golden relevance label
            every question:
                -> not-found correctness (declined iff unanswerable)
        -> eval/results/answers[_<tag>].json

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    argparse - standard library; command-line options for the evaluation run
    json - standard library; writes the results document
    logging - standard library; logs per-question scores
    os - standard library; sets KMP_DUPLICATE_LIB_OK
    statistics - standard library; median latency figures
    time - standard library; measures the evaluation time per question
    common.logging - project; configures logging from the settings
    configs - project; loads the application settings
    eval.dataset - project; golden questions, relevance matching and the results folder
    eval.metrics - project; cosine, mean and the numbered verdict parser
    rag.citations - project; removes citation markers before judging the answer
    rag.sentences - project; splits the answer into statements
    rag.llm - project; builds the judge LLM client (imported inside run)
    rag.pipeline - project; builds the answer pipeline under evaluation (imported inside run)
    retrieval.embeddings - project; the encoder for answer relevancy (imported inside run)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time

from common.logging import configure_logging
from configs import get_settings
from eval.dataset import RESULTS, is_relevant, load_questions
from eval.metrics import cosine, mean, parse_numbered_verdicts
from rag.citations import strip_citations
from rag.sentences import split_sentences

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
log = logging.getLogger(__name__)

# Statements come from deterministic sentence splitting, not from the judge: a 3B judge
# asked to "list the claims" tends to list facts from the context instead of the answer.
FAITHFULNESS_SYSTEM = """You check statements against a CONTEXT.
For each numbered statement, write one line "<number>: YES" if the CONTEXT supports it,
or "<number>: NO" if it does not. Write nothing else."""

RELEVANCY_SYSTEM = """Write 3 different questions that the given answer responds to.
Write one question per line and nothing else."""


def faithfulness(judge, context: str, answer: str) -> tuple[float | None, list]:
    """
    Function name: faithfulness
    Purpose: Share of answer sentences that the judge finds supported by the context. The
        answer is split into numbered statements and the judge votes YES or NO on each.
    Parameters:
        judge (LLMClient): LLM client used as the judge; its complete() method is called.
        context (str): Passage text the answering model saw.
        answer (str): Answer text with citation markers removed.
    Returns:
        tuple[float | None, list]: The score (None when the answer has no sentences or the
            judge output cannot be parsed) and a list of [verdict, statement] pairs.
    """
    statements = split_sentences(answer)
    if not statements:
        return None, []
    numbered = "\n".join(f"{i}. {st}" for i, st in enumerate(statements, start=1))
    text = judge.complete(
        [
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nSTATEMENTS:\n{numbered}"},
        ],
        temperature=0.0,
        max_tokens=60,
    )
    verdicts = parse_numbered_verdicts(text, len(statements))
    if not verdicts:
        return None, []
    claims = [[verdicts[i], statements[i - 1]] for i in sorted(verdicts)]
    return sum(verdicts.values()) / len(verdicts), claims


def answer_relevancy(judge, encoder, question: str, answer: str) -> tuple[float | None, list]:
    """
    Function name: answer_relevancy
    Purpose: Mean cosine similarity between the question and up to three questions that the
        judge generates back from the answer.
    Parameters:
        judge (LLMClient): LLM client used as the judge; its complete() method is called.
        encoder (Encoder): Embedding model; its embed_documents() method is called.
        question (str): Original golden question.
        answer (str): Answer text with citation markers removed.
    Returns:
        tuple[float | None, list]: The score (None when no question was generated) and the
            list of generated questions.
    """
    text = judge.complete(
        [
            {"role": "system", "content": RELEVANCY_SYSTEM},
            {"role": "user", "content": answer},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    generated = [ln.strip(" -*0123456789.)") for ln in text.splitlines() if "?" in ln][:3]
    if not generated:
        return None, []
    vecs = encoder.embed_documents([question, *generated])
    return sum(cosine(vecs[0], v) for v in vecs[1:]) / len(generated), generated


def run(limit: int | None, few_shot: bool, temperature: float | None, judge_model: str | None):
    """
    Function name: run
    Purpose: Build the answer pipeline, judge and encoder, answer each golden question and
        record one result row per question with timings, not-found correctness and, for
        answered answerable questions, citation correctness, faithfulness and relevancy.
    Parameters:
        limit (int | None): Number of questions to evaluate from the start; None or 0 uses all.
        few_shot (bool): Whether the answer prompt includes the few-shot examples.
        temperature (float | None): Answer temperature; None uses the configured value.
        judge_model (str | None): Judge model name; None uses the configured LLM model.
    Returns:
        tuple[Settings, list[dict]]: The settings used and the per-question result rows.
    """
    from rag.llm import build_llm
    from rag.pipeline import build_pipeline
    from retrieval.embeddings import get_encoder

    settings = get_settings()
    pipeline = build_pipeline(settings)
    pipeline.few_shot = few_shot
    judge_cfg = settings.llm.model_copy(update={"model": judge_model} if judge_model else {})
    judge = build_llm(judge_cfg)
    encoder = get_encoder(settings)
    questions = load_questions()[:limit] if limit else load_questions()

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        r = pipeline.answer(q.question, temperature=temperature)
        row = {
            "id": q.id,
            "question": q.question,
            "answerable": q.answerable,
            "query": r.query,
            "answer": r.answer,
            "found": r.found,
            "not_found_correct": r.found == q.answerable,
            "citations": [c.label for c in r.citations],
            "top_score": r.top_score,
            "timings": r.timings,
        }
        if q.answerable and r.found:
            by_id = {p["chunk_id"]: p for p in r.passages}
            row["citation_correct"] = any(
                is_relevant(pipeline.retriever.by_id[c.chunk_id], q)
                for c in r.citations
                if c.chunk_id in by_id
            )
            context = "\n".join(p["text"] for p in r.passages[: pipeline.context_k])
            spoken = strip_citations(r.answer)
            row["faithfulness"], row["claims"] = faithfulness(judge, context, spoken)
            row["answer_relevancy"], row["generated_questions"] = answer_relevancy(
                judge, encoder, q.question, spoken
            )
        row["eval_seconds"] = round(time.perf_counter() - t0, 1)
        log.info(
            "q=%s found=%s faith=%s rel=%s",
            q.id,
            r.found,
            row.get("faithfulness"),
            row.get("answer_relevancy"),
        )
        rows.append(row)
    return settings, rows


def summarize(rows: list[dict]) -> dict:
    """
    Function name: summarize
    Purpose: Aggregate the per-question rows into summary metrics: faithfulness, answer
        relevancy, answered rate, not-found accuracy, unanswerable decline rate, citation
        rate and correctness, and median latencies.
    Parameters:
        rows (list[dict]): Per-question result rows from run().
    Returns:
        dict: The summary metrics.
    """
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if r["found"]]
    faith = [r["faithfulness"] for r in answered if r.get("faithfulness") is not None]
    rel = [r["answer_relevancy"] for r in answered if r.get("answer_relevancy") is not None]
    first = [
        r["timings"]["llm_first_token_ms"] for r in rows if "llm_first_token_ms" in r["timings"]
    ]
    total = [r["timings"]["total_ms"] for r in rows]
    return {
        "faithfulness": mean(faith),
        "faithfulness_scored": len(faith),
        "answer_relevancy": round(mean(rel), 4),
        "answer_relevancy_scored": len(rel),
        "answered_rate": round(len(answered) / len(answerable), 4) if answerable else 0.0,
        "not_found_accuracy": mean([float(r["not_found_correct"]) for r in rows]),
        "unanswerable_declined": mean([float(not r["found"]) for r in unanswerable]),
        "citation_rate": mean([float(bool(r["citations"])) for r in answered]),
        "citation_correct": mean([float(r.get("citation_correct", False)) for r in answered]),
        "median_llm_first_token_ms": round(statistics.median(first), 1) if first else None,
        "median_total_ms": round(statistics.median(total), 1) if total else None,
    }


def main() -> None:
    """
    Function name: main
    Purpose: Parse the command-line options, run the evaluation, write the results JSON file
        (answers.json, or answers_<tag>.json when a tag is given) and print the summary.
    Parameters:
        None
    Returns:
        None: The results are written to a file and printed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Answer-quality evaluation with a local LLM judge (RAGAS-style metrics, no hosted API)."
        )
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--judge-model")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    settings, rows = run(args.limit, not args.no_few_shot, args.temperature, args.judge_model)
    out = {
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model
        if settings.llm.provider == "ollama"
        else settings.llm.openai_model,
        "judge_model": args.judge_model or settings.llm.model,
        "few_shot": not args.no_few_shot,
        "temperature": args.temperature
        if args.temperature is not None
        else settings.llm.temperature,
        "context_k": settings.retrieval.context_k,
        "not_found_threshold": settings.retrieval.not_found_threshold,
        "questions": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }
    name = f"answers_{args.tag}.json" if args.tag else "answers.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
