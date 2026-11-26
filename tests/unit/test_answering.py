"""
File: tests/unit/test_answering.py

Description:
    Unit tests for prompt building, citation parsing, not-found logic, sentence streaming,
    query rewrite and the answer pipeline.

Libraries used:
    pytest - third party; test parametrisation
    common.types - project; Chunk and Hit used to build test hits
    rag.citations - project; citation resolution, stripping and not-found checks under test
    rag.pipeline - project; the answer pipeline under test
    rag.prompts - project; prompt builders and the NOT_FOUND reply under test
    rag.rewrite - project; filler detection and query preparation under test
    rag.sentences - project; sentence splitting and streaming under test
    tests.conftest - project; the FakeLLM test double

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import pytest

from common.types import Chunk, Hit
from rag.citations import evidence_is_weak, is_not_found, resolve_citations, strip_citations
from rag.pipeline import AnswerPipeline
from rag.prompts import NOT_FOUND, build_answer_messages, build_rewrite_messages, format_passages
from rag.rewrite import needs_rewrite, prepare_query, strip_fillers
from rag.sentences import SentenceSplitterStream, split_sentences
from tests.conftest import FakeLLM


def _hits():
    """
    Function name: _hits
    Purpose: Build two fixed hits for the tests: a PDF chunk on page 3 with re-rank score 5.0
        and a Markdown chunk with re-rank score 1.0.
    Parameters:
        None
    Returns:
        list[Hit]: The two test hits in rank order.
    """
    return [
        Hit(
            Chunk("p.pdf:3:0", "Fees\nLate fee is 30 dollars.", "p.pdf", 3, "Fees"),
            scores={"rerank": 5.0},
        ),
        Hit(
            Chunk("f.md:0:2", "Wires\nWires cost 45 dollars.", "f.md", None, "Wires"),
            scores={"rerank": 1.0},
        ),
    ]


# prompts


def test_passages_are_numbered_with_labels():
    """
    Function name: test_passages_are_numbered_with_labels
    Purpose: Check that formatted passages are numbered and carry their source labels, with
        the page for PDFs and the section for Markdown.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    block = format_passages(_hits())
    assert block.splitlines()[0].startswith("[1] (p.pdf, p. 3) Fees")
    assert "[2] (f.md, Wires)" in block


def test_answer_prompt_structure():
    """
    Function name: test_answer_prompt_structure
    Purpose: Check the answer prompt layout: a system message with the not-found reply, two
        few-shot pairs when enabled, the question last, and no real sample company names in
        the few-shot examples.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    msgs = build_answer_messages("What is the late fee?", _hits(), few_shot=True)
    assert msgs[0]["role"] == "system" and NOT_FOUND in msgs[0]["content"]
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"].endswith(
        "Question: What is the late fee?"
    )
    assert len(msgs) == 6  # system + 2 few-shot pairs + question
    assert len(build_answer_messages("q", _hits(), few_shot=False)) == 2
    # few-shot examples never mention the real sample companies
    assert "Acme" not in str(msgs[1:5]) and "Kova" not in str(msgs[1:5])


def test_rewrite_prompt_ends_with_utterance():
    """
    Function name: test_rewrite_prompt_ends_with_utterance
    Purpose: Check that the rewrite prompt ends with the trimmed user utterance.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert build_rewrite_messages(" uh hi ")[-1] == {"role": "user", "content": "uh hi"}


# citations and not-found


def test_numeric_citations_resolve_to_labels():
    """
    Function name: test_numeric_citations_resolve_to_labels
    Purpose: Check that numeric citations are replaced with source labels and that the cited
        chunks are returned once each, in order of first use, with their pages.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    text, cites = resolve_citations(
        "The late fee is 30 dollars [1]. Wires cost 45 [2][1].", _hits()
    )
    assert text == (
        "The late fee is 30 dollars [p.pdf, p. 3]. Wires cost 45 [f.md, Wires][p.pdf, p. 3]."
    )
    assert [c.chunk_id for c in cites] == ["p.pdf:3:0", "f.md:0:2"]
    assert cites[0].page == 3 and cites[1].page is None


def test_grouped_and_out_of_range_numbers():
    """
    Function name: test_grouped_and_out_of_range_numbers
    Purpose: Check that a grouped citation such as [1, 2] resolves to both chunks and that an
        out-of-range number is removed from the text.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    text, cites = resolve_citations("Fact [1, 2]. Other [7].", _hits())
    assert len(cites) == 2
    assert "[7]" not in text and text.endswith("Other.")


def test_label_style_citation_is_accepted():
    """
    Function name: test_label_style_citation_is_accepted
    Purpose: Check that a citation written as a source label, such as [p.pdf, page 3],
        resolves to the matching chunk.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    _, cites = resolve_citations("Late fee is 30 dollars [p.pdf, page 3].", _hits())
    assert [c.chunk_id for c in cites] == ["p.pdf:3:0"]


def test_strip_citations_for_speech():
    """
    Function name: test_strip_citations_for_speech
    Purpose: Check that label and numeric citations are removed from text meant for speech.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert strip_citations("Fee is 30 dollars [p.pdf, p. 3]. Wires [2] cost 45.") == (
        "Fee is 30 dollars. Wires cost 45."
    )


def test_not_found_detection():
    """
    Function name: test_not_found_detection
    Purpose: Check that the standard not-found reply and a paraphrase of it are detected, and
        that a real answer is not.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert is_not_found(NOT_FOUND)
    assert is_not_found("Sorry, I couldn't find that in the documents.")
    assert not is_not_found("The fee is 30 dollars [1].")


def test_weak_evidence_threshold():
    """
    Function name: test_weak_evidence_threshold
    Purpose: Check the weak-evidence rule: no hits is weak, the best re-rank score is compared
        with the threshold, and hits without a re-rank score are never judged weak.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert evidence_is_weak([], 0.0)
    assert not evidence_is_weak(_hits(), 1.5)  # best rerank 5.0
    assert evidence_is_weak(_hits(), 6.0)
    no_rerank = [Hit(_hits()[0].chunk, scores={"rrf": 0.03})]
    assert not evidence_is_weak(no_rerank, 6.0)  # only judged when a rerank score exists


# sentences


def test_stream_emits_sentences_as_they_complete():
    """
    Function name: test_stream_emits_sentences_as_they_complete
    Purpose: Check that the streaming splitter emits each sentence once it is complete, does
        not split on decimals or abbreviations, keeps a trailing citation, and returns the
        remaining text on flush.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    s = SentenceSplitterStream()
    out = []
    for piece in [
        "The fee is 3",
        ".80 percent [1]",
        ". Wires cost",
        " 45 dollars at 3 p.m. today. ",
        "End",
    ]:
        out += s.feed(piece)
    assert out == ["The fee is 3.80 percent [1].", "Wires cost 45 dollars at 3 p.m. today."]
    assert s.flush() == ["End"]


def test_split_sentences_keeps_trailing_citation():
    """
    Function name: test_split_sentences_keeps_trailing_citation
    Purpose: Check that a citation placed after a full stop stays with the preceding sentence.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert split_sentences("First fact here. [1] Second fact here.") == [
        "First fact here. [1]",
        "Second fact here.",
    ]


# rewrite


def test_filler_detection_and_stripping():
    """
    Function name: test_filler_detection_and_stripping
    Purpose: Check that speech fillers trigger a rewrite, that the word "like" in a normal
        question does not, and that filler words are stripped from the text.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    assert needs_rewrite("uh so like what is the late fee")
    assert not needs_rewrite("What is the limit for structures like a garage?")
    assert strip_fillers("um, what is the, uh, late fee?") == "what is the, late fee?"


def test_prepare_query_modes():
    """
    Function name: test_prepare_query_modes
    Purpose: Check the rewrite modes: auto rewrites casual speech with the LLM, auto skips the
        LLM for a clean question, and never only trims the text.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    llm = FakeLLM(rewrite="What is the late fee?")
    assert prepare_query(llm, "um what's the late fee thing", "auto") == "What is the late fee?"
    assert prepare_query(llm, "What is the late fee?", "auto") == "What is the late fee?"
    assert len(llm.calls) == 1  # clean question skipped the LLM
    assert prepare_query(llm, " raw ", "never") == "raw"


def test_rewrite_falls_back_on_llm_error():
    """
    Function name: test_rewrite_falls_back_on_llm_error
    Purpose: Check that when the LLM fails during a rewrite, the query falls back to the text
        with fillers stripped.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """

    class Broken(FakeLLM):
        """
        Class name: Broken
        Purpose: Fake LLM whose stream always raises, to simulate an LLM failure.
        """

        def stream(self, *a, **k):
            """
            Function name: Broken.stream
            Purpose: Raise a RuntimeError instead of streaming a reply.
            Parameters:
                a (tuple): Positional arguments, ignored.
                k (dict): Keyword arguments, ignored.
            Returns:
                Iterator[str]: never returns; always raises RuntimeError.
            """
            raise RuntimeError("down")

    assert prepare_query(Broken(), "um what is the fee", "always") == "what is the fee"


# pipeline


def _pipeline(retriever, reply, threshold=-1.0):
    """
    Function name: _pipeline
    Purpose: Build an AnswerPipeline over the test retriever with a FakeLLM that gives the
        scripted reply, top_k 3, context_k 2 and query rewrite turned off.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
        reply (str): Scripted LLM reply.
        threshold (float): Not-found threshold on the best re-rank score.
    Returns:
        AnswerPipeline: The pipeline under test.
    """
    return AnswerPipeline(
        retriever,
        FakeLLM(reply),
        top_k=3,
        context_k=2,
        not_found_threshold=threshold,
        rewrite_mode="never",
    )


def test_pipeline_answers_with_citations_and_timings(retriever):
    """
    Function name: test_pipeline_answers_with_citations_and_timings
    Purpose: Check the streamed answer events: query first and result last, sentences without
        citations, a found result with resolved citations and stage timings, and only
        context_k passages in the prompt.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    p = _pipeline(retriever, "International wires cost 45 dollars [1]. That is all [9].")
    events = list(p.stream_answer("How much do international wires cost?"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "query" and kinds[-1] == "result"
    sentences = [e["text"] for e in events if e["type"] == "sentence"]
    assert sentences == ["International wires cost 45 dollars.", "That is all."]
    r = events[-1]["result"]
    assert r.found and r.citations[0].chunk_id == "faq.md:0:1"
    assert r.answer.startswith("International wires cost 45 dollars [faq.md, Wires].")
    assert {"retrieval_ms", "llm_first_token_ms", "llm_ms", "total_ms"} <= set(r.timings)
    # only context_k passages reach the prompt
    assert "[3]" not in p.llm.calls[0][-1]["content"]


def test_pipeline_skips_llm_on_weak_evidence(retriever):
    """
    Function name: test_pipeline_skips_llm_on_weak_evidence
    Purpose: Check that weak retrieval evidence returns the not-found reply without calling
        the LLM and without an LLM timing.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    p = _pipeline(retriever, "should not be used", threshold=50.0)
    r = p.answer("What is the capital of France?")
    assert not r.found and r.answer == NOT_FOUND and r.citations == []
    assert p.llm.calls == []
    assert "llm_first_token_ms" not in r.timings


def test_pipeline_respects_model_not_found(retriever):
    """
    Function name: test_pipeline_respects_model_not_found
    Purpose: Check that a not-found reply from the model gives a not-found result with no
        citations.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    r = _pipeline(retriever, NOT_FOUND).answer("How much do international wires cost?")
    assert not r.found and r.citations == [] and r.spoken == NOT_FOUND


@pytest.mark.parametrize("reply", ["", "   "])
def test_empty_model_reply_is_not_found(retriever, reply):
    """
    Function name: test_empty_model_reply_is_not_found
    Purpose: Check that an empty or blank model reply is treated as not found.
    Parameters:
        retriever (HybridRetriever): Test retriever fixture.
        reply (str): Scripted empty or blank LLM reply (parametrised).
    Returns:
        None: The test passes or fails through assertions.
    """
    assert not _pipeline(retriever, reply).answer("wires cost").found
