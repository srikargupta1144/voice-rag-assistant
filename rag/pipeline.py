"""Question -> cited answer: rewrite, retrieve, not-found gate, grounded generation, citations.

Flow:
    api.routes, ingest.cli ask, eval.answer_eval
        -> build_pipeline(settings)  (retriever + LLM client)
        -> AnswerPipeline.stream_answer(question, mode, timer)       # generator of events
            -> rag.rewrite.prepare_query()                            # stage "rewrite"
            -> retrieval.retriever.HybridRetriever.retrieve()         # stage "retrieval"
            -> rag.citations.evidence_is_weak()  -> not-found, LLM skipped
            -> rag.prompts.build_answer_messages()
            -> llm.stream()                                           # llm_first_token mark
                -> rag.sentences.SentenceSplitterStream.feed()        # "sentence" events
            -> rag.citations.is_not_found() / resolve_citations()
            -> final "result" event carrying AnswerResult
        -> AnswerPipeline.answer() consumes the stream for non-streaming callers

Sentence events carry the text to speak (citations stripped), so voice callers can start
TTS on the first sentence while the model is still writing the rest.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from common.timing import StageTimer
from common.types import Hit
from rag.citations import (
    Citation,
    evidence_is_weak,
    is_not_found,
    resolve_citations,
    strip_citations,
)
from rag.prompts import NOT_FOUND, build_answer_messages
from rag.rewrite import prepare_query
from rag.sentences import SentenceSplitterStream

if TYPE_CHECKING:
    from configs import Settings
    from rag.llm import LLMClient
    from retrieval.retriever import HybridRetriever

log = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Everything an API response or evaluation needs about one answer."""

    question: str
    query: str
    answer: str  # display text with [source, page] citations
    spoken: str  # text that is read aloud (no citations)
    found: bool
    citations: list[Citation]
    passages: list[dict[str, Any]]
    top_score: float | None
    timings: dict[str, float] = field(default_factory=dict)
    raw_answer: str = ""  # model output before citation rewriting

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d


def _passage_dict(rank: int, h: Hit) -> dict[str, Any]:
    c = h.chunk
    return {
        "rank": rank,
        "chunk_id": c.chunk_id,
        "source": c.source,
        "page": c.page,
        "section": c.section,
        "label": c.label(),
        "text": c.text,
        "scores": {k: round(v, 4) for k, v in h.scores.items()},
    }


class AnswerPipeline:
    """Holds the retriever and LLM; one instance serves all requests."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        top_k: int = 5,
        context_k: int = 3,
        not_found_threshold: float = 1.5,
        rewrite_mode: str = "auto",
        rewrite_max_tokens: int = 40,
        few_shot: bool = True,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.context_k = context_k
        self.not_found_threshold = not_found_threshold
        self.rewrite_mode = rewrite_mode
        self.rewrite_max_tokens = rewrite_max_tokens
        self.few_shot = few_shot

    def stream_answer(
        self,
        question: str,
        mode: str = "full",
        timer: StageTimer | None = None,
        rewrite_mode: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield {"type": "query"|"sentence"|"result", ...} events; see module docstring."""
        timer = timer or StageTimer()
        with timer.stage("rewrite"):
            query = prepare_query(
                self.llm, question, rewrite_mode or self.rewrite_mode, self.rewrite_max_tokens
            )
        yield {"type": "query", "query": query}

        with timer.stage("retrieval"):
            hits = self.retriever.retrieve(query, mode=mode, top_k=self.top_k)
        passages = [_passage_dict(i, h) for i, h in enumerate(hits, start=1)]
        top = max((h.scores["rerank"] for h in hits if "rerank" in h.scores), default=None)

        if evidence_is_weak(hits, self.not_found_threshold):
            log.info("not_found reason=weak_evidence top_score=%s", top)
            yield {"type": "sentence", "index": 0, "text": NOT_FOUND}
            yield {
                "type": "result",
                "result": AnswerResult(
                    question,
                    query,
                    NOT_FOUND,
                    NOT_FOUND,
                    False,
                    [],
                    passages,
                    top,
                    timer.as_dict(),
                    NOT_FOUND,
                ),
            }
            return

        context = hits[: self.context_k]  # citation numbers refer to these passages
        messages = build_answer_messages(query, context, few_shot=self.few_shot)
        splitter = SentenceSplitterStream()
        pieces: list[str] = []
        n_sent = 0
        with timer.stage("llm"):
            for piece in self.llm.stream(messages, temperature=temperature):
                timer.mark("llm_first_token")
                pieces.append(piece)
                for sentence in splitter.feed(piece):
                    spoken = strip_citations(sentence)
                    if spoken:
                        yield {"type": "sentence", "index": n_sent, "text": spoken}
                        n_sent += 1
        for sentence in splitter.flush():
            spoken = strip_citations(sentence)
            if spoken:
                yield {"type": "sentence", "index": n_sent, "text": spoken}
                n_sent += 1

        raw = "".join(pieces).strip()
        if not raw or is_not_found(raw):
            answer, citations, found = NOT_FOUND, [], False
        else:
            answer, citations = resolve_citations(raw, context)
            found = True
        result = AnswerResult(
            question=question,
            query=query,
            answer=answer,
            spoken=strip_citations(answer),
            found=found,
            citations=citations,
            passages=passages,
            top_score=top,
            timings=timer.as_dict(),
            raw_answer=raw,
        )
        yield {"type": "result", "result": result}

    def answer(
        self,
        question: str,
        mode: str = "full",
        timer: StageTimer | None = None,
        rewrite_mode: str | None = None,
        temperature: float | None = None,
    ) -> AnswerResult:
        """Run stream_answer() to completion and return the result."""
        for event in self.stream_answer(question, mode, timer, rewrite_mode, temperature):
            if event["type"] == "result":
                return event["result"]
        raise RuntimeError("pipeline ended without a result")


def build_pipeline(settings: Settings, retriever: HybridRetriever | None = None) -> AnswerPipeline:
    """Pipeline from settings: on-disk index + configured LLM backend."""
    from rag.llm import build_llm
    from retrieval.retriever import load_retriever

    r = settings.retrieval
    return AnswerPipeline(
        retriever=retriever or load_retriever(settings),
        llm=build_llm(settings.llm),
        top_k=r.top_k,
        context_k=r.context_k,
        not_found_threshold=r.not_found_threshold,
        rewrite_mode=settings.rewrite.mode,
        rewrite_max_tokens=settings.rewrite.max_tokens,
        few_shot=settings.llm.few_shot,
    )
