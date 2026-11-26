"""
File: tests/functional/test_retrieval_real.py

Description:
    Functional tests of retrieval with the real models on the committed index: golden
    questions find their labelled source (and page for PDFs) in the top 5 of the full
    pipeline, unanswerable questions score below the not-found threshold, and every
    retrieval mode returns hits with its own score.

    Run with: .venv\\Scripts\\python -m pytest -m models

Libraries used:
    pytest - third party; the models marker and parametrised cases
    rag.citations - project; evidence_is_weak, the not-found gate on re-ranker scores
    retrieval.retriever - project; the list of retrieval modes

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

import pytest

from rag.citations import evidence_is_weak
from retrieval.retriever import MODES

pytestmark = pytest.mark.models

# One or two questions per document, including a conversational one (q04) that dense search
# alone ranks below fifth.
GOLDEN_IDS = ["q01", "q04", "q07", "q13", "q22", "q25", "q28"]

# Questions the documents do not answer and whose best re-ranker score stays below the
# threshold. u01 and u03 are left out on purpose: they share words with real passages and
# score above it (documented in docs/TECHNICAL_REFERENCE.md, Known limits).
UNANSWERABLE = [
    "Does Kova offer car loans?",
    "Does Acme home insurance pay for my dog's vet bills?",
    "How much does Acme car insurance cost per month?",
    "Who is the CEO of Kova Bank?",
]

# The main score each mode sets on its hits.
MODE_SCORE = {"dense": "dense", "bm25": "bm25", "hybrid": "rrf", "full": "rerank"}


def is_relevant(hit, label: dict) -> bool:
    """
    Function name: is_relevant
    Purpose: Check a hit against one golden label the same way eval.retrieval_eval does:
        same source, same page for PDFs, and the key phrase present in the chunk text.
    Parameters:
        hit (Hit): A retrieved passage.
        label (dict): One entry of the question's "relevant" list.
    Returns:
        bool: True when the hit is the labelled passage.
    """
    c = hit.chunk
    if c.source != label["source"]:
        return False
    if label.get("page") is not None and c.page != label["page"]:
        return False
    return label["contains"].lower() in c.text.lower()


@pytest.mark.parametrize("qid", GOLDEN_IDS)
def test_golden_question_finds_labelled_passage_in_top_5(real_retriever, golden, qid):
    """
    Function name: test_golden_question_finds_labelled_passage_in_top_5
    Purpose: Check that the full pipeline (dense, BM25, RRF and re-rank) returns the
        labelled source, page and passage within the top 5 hits, and that every hit carries
        all four stage scores.
    Parameters:
        real_retriever (HybridRetriever): Retriever over the committed index.
        golden (dict[str, dict]): Golden questions by id.
        qid (str): Id of the golden question.
    Returns:
        None: The test passes or fails through assertions.
    """
    row = golden[qid]
    hits = real_retriever.retrieve(row["question"], mode="full", top_k=5)
    assert len(hits) == 5
    found = [h for h in hits if any(is_relevant(h, lab) for lab in row["relevant"])]
    got = [(h.chunk.source, h.chunk.page) for h in hits]
    assert found, f"{qid}: labelled passage not in top 5; got {got}"
    for h in hits:
        # Every fused hit came from the dense leg, the BM25 leg or both.
        assert {"rrf", "rerank"} <= set(h.scores)
        assert {"dense", "bm25"} & set(h.scores)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_answerable_golden_questions_pass_the_not_found_gate(real_retriever, real_settings):
    """
    Function name: test_answerable_golden_questions_pass_the_not_found_gate
    Purpose: Check that a clearly answerable question scores above the not-found threshold,
        so the gate does not block it.
    Parameters:
        real_retriever (HybridRetriever): Retriever over the committed index.
        real_settings (Settings): Session settings with the threshold.
    Returns:
        None: The test passes or fails through assertions.
    """
    hits = real_retriever.retrieve("What is the standard deductible on the Acme policy?")
    assert not evidence_is_weak(hits, real_settings.retrieval.not_found_threshold)


@pytest.mark.parametrize("question", UNANSWERABLE)
def test_unanswerable_question_scores_below_not_found_threshold(
    real_retriever, real_settings, question
):
    """
    Function name: test_unanswerable_question_scores_below_not_found_threshold
    Purpose: Check that the best re-ranker score of an unanswerable question is below the
        not-found threshold, so the assistant answers "not found" without the LLM.
    Parameters:
        real_retriever (HybridRetriever): Retriever over the committed index.
        real_settings (Settings): Session settings with the threshold.
        question (str): A question the documents do not answer.
    Returns:
        None: The test passes or fails through assertions.
    """
    threshold = real_settings.retrieval.not_found_threshold
    hits = real_retriever.retrieve(question, mode="full", top_k=5)
    top = max(h.scores["rerank"] for h in hits)
    assert top < threshold, f"{question!r} scored {top:.2f}, threshold {threshold}"
    assert evidence_is_weak(hits, threshold)


@pytest.mark.parametrize("mode", MODES)
def test_each_retrieval_mode_returns_hits(real_retriever, mode):
    """
    Function name: test_each_retrieval_mode_returns_hits
    Purpose: Check that dense, bm25, hybrid and full mode each return up to top_k hits,
        best first, each with the main score of that mode, and that the wire fee question
        finds the Kova FAQ in every mode.
    Parameters:
        real_retriever (HybridRetriever): Retriever over the committed index.
        mode (str): Retrieval mode under test.
    Returns:
        None: The test passes or fails through assertions.
    """
    hits = real_retriever.retrieve("How much is an outgoing international wire?", mode, 5)
    assert 0 < len(hits) <= 5
    key = MODE_SCORE[mode]
    assert all(key in h.scores for h in hits)
    main = [h.scores[key] for h in hits]
    assert main == sorted(main, reverse=True)
    assert "kova_bank_account_faq.md" in {h.chunk.source for h in hits}


def test_retriever_covers_committed_index(real_retriever):
    """
    Function name: test_retriever_covers_committed_index
    Purpose: Check that the loaded retriever holds all 36 committed chunks with unique ids
        and that the FAISS store has one vector per chunk.
    Parameters:
        real_retriever (HybridRetriever): Retriever over the committed index.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert len(real_retriever.chunks) == 36
    assert len(real_retriever.by_id) == 36
    assert real_retriever.encoder.dim == 384
    hits = real_retriever.retrieve("fee", mode="dense", top_k=100)
    assert len(hits) == 36
