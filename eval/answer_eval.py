"""Answer-quality evaluation with a local LLM judge (RAGAS-style metrics, no hosted API).

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

The judge is the same local Ollama model unless --judge-model is given; a 3B judge is
lenient and noisy, so treat these numbers as a smoke signal, not a benchmark.
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
    """Share of answer sentences the judge finds supported by context (None if unparseable)."""
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
    """Mean cosine between the question and questions generated back from the answer."""
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
    """Aggregate metrics over the per-question rows."""
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
    parser = argparse.ArgumentParser(description=__doc__)
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
