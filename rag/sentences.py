"""
File: rag/sentences.py

Description:
    Incremental sentence splitting over a token stream, so speech can start early.

    A boundary is ., ! or ? followed by whitespace (so decimals like 3.80 never split),
    except after common abbreviations (p., a.m., e.g.). Citation markers right after the
    period ("fee. [2]") stay with their sentence. A boundary is only confirmed once the
    following whitespace has arrived, so a half-streamed number is never cut.

Flow:
    rag.pipeline.AnswerPipeline.stream_answer()
        -> splitter = SentenceSplitterStream()
        -> for piece in llm.stream(...): splitter.feed(piece) -> completed sentences
        -> splitter.flush()                                     -> the remainder
    api /ws/voice and /ask/audio
        -> voice.tts.synthesize(sentence) per completed sentence

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    re - standard library; the sentence boundary pattern

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Class name: SentenceSplitterStream
    Purpose: Feed text pieces; get back sentences as soon as they are complete.
    """

    def __init__(self, min_chars: int = 12):
        """
        Function name: SentenceSplitterStream.__init__
        Purpose: Start with an empty buffer and set the shortest sentence that may be cut.
        Parameters:
            min_chars (int): The lowest buffer position at which a boundary is accepted,
                             so very short pieces are not spoken alone.
        Returns:
            None
        """
        self._buf = ""
        self._min = min_chars  # avoid speaking fragments like "Yes."-sized pieces alone

    def feed(self, piece: str) -> list[str]:
        """
        Function name: SentenceSplitterStream.feed
        Purpose: Add a piece of text to the buffer and return any sentences completed by it.
        Parameters:
            piece (str): The next piece of streamed text.
        Returns:
            list[str]: The completed sentences, possibly empty.
        """
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
        """
        Function name: SentenceSplitterStream.flush
        Purpose: Return whatever is left in the buffer as a final sentence and clear it.
        Parameters:
            None
        Returns:
            list[str]: The remaining text as one sentence, or an empty list when none is left.
        """
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _find_boundary(self) -> int | None:
        """
        Function name: SentenceSplitterStream._find_boundary
        Purpose: Find the first confirmed sentence end in the buffer, skipping ends before
                 the minimum length and ends after a known abbreviation.
        Parameters:
            None
        Returns:
            int | None: The buffer index just after the sentence end (including any citation
                        markers), or None when no boundary is confirmed yet.
        """
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
    """
    Function name: split_sentences
    Purpose: Split a complete text into sentences with the same rules as the stream.
    Parameters:
        text (str): The complete text.
    Returns:
        list[str]: The sentences in order.
    """
    s = SentenceSplitterStream()
    return s.feed(text + " ") + s.flush()
