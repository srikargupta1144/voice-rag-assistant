"""Prompt building, citation parsing, not-found logic, sentence streaming, rewrite, pipeline."""

import pytest

from common.types import Chunk, Hit
from rag.citations import evidence_is_weak, is_not_found, resolve_citations, strip_citations
from rag.pipeline import AnswerPipeline
from rag.prompts import NOT_FOUND, build_answer_messages, build_rewrite_messages, format_passages
from rag.rewrite import needs_rewrite, prepare_query, strip_fillers
from rag.sentences import SentenceSplitterStream, split_sentences
from tests.conftest import FakeLLM


def _hits():
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
    block = format_passages(_hits())
    assert block.splitlines()[0].startswith("[1] (p.pdf, p. 3) Fees")
    assert "[2] (f.md, Wires)" in block


def test_answer_prompt_structure():
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
    assert build_rewrite_messages(" uh hi ")[-1] == {"role": "user", "content": "uh hi"}


# citations and not-found


def test_numeric_citations_resolve_to_labels():
    text, cites = resolve_citations(
        "The late fee is 30 dollars [1]. Wires cost 45 [2][1].", _hits()
    )
    assert text == (
        "The late fee is 30 dollars [p.pdf, p. 3]. Wires cost 45 [f.md, Wires][p.pdf, p. 3]."
    )
    assert [c.chunk_id for c in cites] == ["p.pdf:3:0", "f.md:0:2"]
    assert cites[0].page == 3 and cites[1].page is None


def test_grouped_and_out_of_range_numbers():
    text, cites = resolve_citations("Fact [1, 2]. Other [7].", _hits())
    assert len(cites) == 2
    assert "[7]" not in text and text.endswith("Other.")


def test_label_style_citation_is_accepted():
    _, cites = resolve_citations("Late fee is 30 dollars [p.pdf, page 3].", _hits())
    assert [c.chunk_id for c in cites] == ["p.pdf:3:0"]


def test_strip_citations_for_speech():
    assert strip_citations("Fee is 30 dollars [p.pdf, p. 3]. Wires [2] cost 45.") == (
        "Fee is 30 dollars. Wires cost 45."
    )


def test_not_found_detection():
    assert is_not_found(NOT_FOUND)
    assert is_not_found("Sorry, I couldn't find that in the documents.")
    assert not is_not_found("The fee is 30 dollars [1].")


def test_weak_evidence_threshold():
    assert evidence_is_weak([], 0.0)
    assert not evidence_is_weak(_hits(), 1.5)  # best rerank 5.0
    assert evidence_is_weak(_hits(), 6.0)
    no_rerank = [Hit(_hits()[0].chunk, scores={"rrf": 0.03})]
    assert not evidence_is_weak(no_rerank, 6.0)  # only judged when a rerank score exists


# sentences


def test_stream_emits_sentences_as_they_complete():
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
    assert split_sentences("First fact here. [1] Second fact here.") == [
        "First fact here. [1]",
        "Second fact here.",
    ]


# rewrite


def test_filler_detection_and_stripping():
    assert needs_rewrite("uh so like what is the late fee")
    assert not needs_rewrite("What is the limit for structures like a garage?")
    assert strip_fillers("um, what is the, uh, late fee?") == "what is the, late fee?"


def test_prepare_query_modes():
    llm = FakeLLM(rewrite="What is the late fee?")
    assert prepare_query(llm, "um what's the late fee thing", "auto") == "What is the late fee?"
    assert prepare_query(llm, "What is the late fee?", "auto") == "What is the late fee?"
    assert len(llm.calls) == 1  # clean question skipped the LLM
    assert prepare_query(llm, " raw ", "never") == "raw"


def test_rewrite_falls_back_on_llm_error():
    class Broken(FakeLLM):
        def stream(self, *a, **k):
            raise RuntimeError("down")

    assert prepare_query(Broken(), "um what is the fee", "always") == "what is the fee"


# pipeline


def _pipeline(retriever, reply, threshold=-1.0):
    return AnswerPipeline(
        retriever,
        FakeLLM(reply),
        top_k=3,
        context_k=2,
        not_found_threshold=threshold,
        rewrite_mode="never",
    )


def test_pipeline_answers_with_citations_and_timings(retriever):
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
    p = _pipeline(retriever, "should not be used", threshold=50.0)
    r = p.answer("What is the capital of France?")
    assert not r.found and r.answer == NOT_FOUND and r.citations == []
    assert p.llm.calls == []
    assert "llm_first_token_ms" not in r.timings


def test_pipeline_respects_model_not_found(retriever):
    r = _pipeline(retriever, NOT_FOUND).answer("How much do international wires cost?")
    assert not r.found and r.citations == [] and r.spoken == NOT_FOUND


@pytest.mark.parametrize("reply", ["", "   "])
def test_empty_model_reply_is_not_found(retriever, reply):
    assert not _pipeline(retriever, reply).answer("wires cost").found
