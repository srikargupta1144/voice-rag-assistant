"""Incremental sentence splitting over a token stream, so speech can start early.

Flow:
    rag.pipeline.AnswerPipeline.stream_answer()
        -> splitter = SentenceSplitterStream()
        -> for piece in llm.stream(...): splitter.feed(piece) -> completed sentences
        -> splitter.flush()                                     -> the remainder
    api /ws/voice and /ask/audio
        -> voice.tts.synthesize(sentence) per completed sentence

A boundary is ., ! or ? followed by whitespace (so decimals like 3.80 never split), except
after common abbreviations (p., a.m., e.g.). Citation markers right after the period
("fee. [2]") stay with their sentence. A boundary is only confirmed once the following
whitespace has arrived, so a half-streamed number is never cut.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = (
    "p.",
    "pp.",
    "a.m.",
    "p.m.",
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "dr.",
    "no.",
    "vs.",
    "etc.",
    "approx.",
    "sec.",
    "st.",
)
_BOUNDARY = re.compile(r"[.!?](?:\s*\[[^\]]*\])*(?=\s)")


class SentenceSplitterStream:
    """Feed text pieces; get back sentences as soon as they are complete."""

    def __init__(self, min_chars: int = 12):
        self._buf = ""
        self._min = min_chars  # avoid speaking fragments like "Yes."-sized pieces alone

    def feed(self, piece: str) -> list[str]:
        """Add a piece of text; return any sentences completed by it."""
        self._buf += piece
        out = []
        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            sentence, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> list[str]:
        """Return whatever is left as a final sentence."""
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _find_boundary(self) -> int | None:
        for m in _BOUNDARY.finditer(self._buf):
            end = m.end()
            if end < self._min:
                continue
            head = self._buf[: m.start() + 1]
            last_word = head.split()[-1].lower() if head.split() else ""
            if last_word.endswith(_ABBREVIATIONS):
                continue
            return end
        return None


def split_sentences(text: str) -> list[str]:
    """Split a complete text into sentences with the same rules as the stream."""
    s = SentenceSplitterStream()
    return s.feed(text + " ") + s.flush()
