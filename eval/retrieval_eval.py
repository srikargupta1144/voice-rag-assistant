"""
File: eval/retrieval_eval.py

Description:
    Retrieval evaluation (no LLM): Recall@5 and MRR per retrieval mode, plus the not-found
    gate.

    Questions are used as written (no LLM rewrite), so CI can run this with no LLM at all.

Flow:
    uv run python -m eval.retrieval_eval [--modes dense,bm25,hybrid,full]
        -> eval.dataset.load_questions()
        -> retrieval.retriever.load_retriever(settings)          # on-disk index from `vra ingest`
        -> per mode, per answerable question:
               HybridRetriever.retrieve(question, mode, top_k=10)
               -> eval.dataset.is_relevant() per hit -> recall_at_k(5), reciprocal_rank(10)
        -> full mode only: top re-ranker score of every question
               -> not-found gate accuracy at retrieval.not_found_threshold
        -> eval/results/retrieval.json        (CI gate reads modes.full.recall_at_5)

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    argparse - standard library; command-line options for the modes and output path
    json - standard library; writes the results document
    logging - standard library; logs per-mode metrics
    os - standard library; sets KMP_DUPLICATE_LIB_OK and creates the output folder
    time - standard library; measures retrieval latency
    common.logging - project; configures logging from the settings
    configs - project; loads the application settings
    eval.dataset - project; golden questions, relevance matching and the results folder
    eval.metrics - project; mean, recall_at_k and reciprocal_rank
    retrieval.retriever - project; the retrieval modes and the on-disk retriever loader

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
    2026-11-26  Srikar Gupta  No crash when a question returns no hits or the golden set has
                              no answerable question.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from common.logging import configure_logging
from configs import get_settings
from eval.dataset import RESULTS, is_relevant, load_questions
from eval.metrics import mean, recall_at_k, reciprocal_rank
from retrieval.retriever import MODES, load_retriever

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
log = logging.getLogger(__name__)


def evaluate(modes: list[str], depth: int = 10) -> dict:
    """
    Function name: evaluate
    Purpose: Run every retrieval mode over the answerable golden questions and build the
        results document with Recall@5, MRR, median latency and misses per mode. When the
        full mode is included, it also measures the not-found gate accuracy on the top
        re-ranker score of every question and copies the full-mode metrics to the top level
        for eval.gate.
    Parameters:
        modes (list[str]): Retrieval modes to evaluate, for example dense, bm25, hybrid, full.
        depth (int): Number of hits retrieved per question for the rank metrics.
    Returns:
        dict: The results document written to eval/results/retrieval.json.
    """
    settings = get_settings()
    questions = load_questions()
    answerable = [q for q in questions if q.answerable]
    retriever = load_retriever(settings)
    threshold = settings.retrieval.not_found_threshold
    out: dict = {
        "questions": len(questions),
        "answerable": len(answerable),
        "unanswerable": len(questions) - len(answerable),
        "chunks": len(retriever.chunks),
        "vector_store": retriever.store.name,
        "embedding_model": settings.embedding.model,
        "rerank_model": settings.retrieval.rerank_model,
        "modes": {},
    }
    for mode in modes:
        recalls, rrs, per_q, latencies = [], [], [], []
        for q in answerable:
            t0 = time.perf_counter()
            hits = retriever.retrieve(q.question, mode=mode, top_k=depth)
            latencies.append((time.perf_counter() - t0) * 1000)
            rel = [is_relevant(h.chunk, q) for h in hits]
            recalls.append(recall_at_k(rel, 5))
            rrs.append(reciprocal_rank(rel, depth))
            first = next((i for i, r in enumerate(rel, start=1) if r), None)
            per_q.append({"id": q.id, "first_relevant_rank": first})
        out["modes"][mode] = {
            "recall_at_5": mean(recalls),
            "mrr_at_10": mean(rrs),
            "median_latency_ms": (
                round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None
            ),
            "misses": [
                p["id"]
                for p in per_q
                if not p["first_relevant_rank"] or p["first_relevant_rank"] > 5
            ],
            "per_question": per_q,
        }
        log.info(
            "mode=%s recall_at_5=%.3f mrr=%.3f",
            mode,
            out["modes"][mode]["recall_at_5"],
            out["modes"][mode]["mrr_at_10"],
        )

    if "full" in modes:
        scores = []
        for q in questions:
            hits = retriever.retrieve(q.question, mode="full", top_k=1)
            top = hits[0].scores.get("rerank") if hits else None
            scores.append({"id": q.id, "answerable": q.answerable, "top_rerank": top})
        correct = sum(
            1
            for s in scores
            if (s["top_rerank"] is not None and s["top_rerank"] >= threshold) == s["answerable"]
        )
        # Questions with no hits have no score; leave them out of the min and max.
        ans = [s["top_rerank"] for s in scores if s["answerable"] and s["top_rerank"] is not None]
        una = [
            s["top_rerank"] for s in scores if not s["answerable"] and s["top_rerank"] is not None
        ]
        out["not_found_gate"] = {
            "threshold": threshold,
            "accuracy": round(correct / len(scores), 4),
            "answerable_min_score": round(min(ans), 3) if ans else None,
            "unanswerable_max_score": round(max(una), 3) if una else None,
            "scores": scores,
        }
    # Flat copy of the gated metric so eval.gate stays a one-key lookup.
    if "full" in out["modes"]:
        out["recall_at_5"] = out["modes"]["full"]["recall_at_5"]
        out["mrr_at_10"] = out["modes"]["full"]["mrr_at_10"]
    return out


def main() -> None:
    """
    Function name: main
    Purpose: Parse the command-line options, run the evaluation, write the results JSON file
        and print a summary table per mode plus the not-found gate line when present.
    Parameters:
        None
    Returns:
        None: The results are written to a file and printed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Retrieval evaluation (no LLM): Recall@5 and MRR per retrieval mode, "
            "plus the not-found gate."
        )
    )
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--out", default=str(RESULTS / "retrieval.json"))
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    result = evaluate([m.strip() for m in args.modes.split(",") if m.strip()])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"{'mode':8} {'Recall@5':>9} {'MRR@10':>8} {'p50 ms':>8}")
    for mode, m in result["modes"].items():
        print(
            f"{mode:8} {m['recall_at_5']:>9.3f} {m['mrr_at_10']:>8.3f} "
            f"{m['median_latency_ms']:>8.1f}"
        )
    if "not_found_gate" in result:
        g = result["not_found_gate"]
        print(
            f"not-found gate @ {g['threshold']}: accuracy {g['accuracy']:.3f} "
            f"(answerable min {g['answerable_min_score']}, "
            f"unanswerable max {g['unanswerable_max_score']})"
        )


if __name__ == "__main__":
    main()
