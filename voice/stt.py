"""Speech-to-text with faster-whisper (small, int8, CPU) and Silero voice activity detection.

Flow:
    api.routes /ask/audio and /ws/voice
        -> get_stt(settings).transcribe(audio_bytes)
            -> WhisperModel.transcribe(BytesIO, vad_filter=True, beam_size=1, language="en")
            -> Transcript(text, duration_s, language)

Any container PyAV can decode is accepted (WAV from the API contract, or WebM/Opus
straight from a browser). VAD drops leading and trailing silence and long pauses, which
both speeds up decoding and stops Whisper inventing words in silence.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from configs import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    duration_s: float
    language: str


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes) -> Transcript: ...


class WhisperSTT:
    """faster-whisper wrapper; the model loads once and is shared across requests."""

    def __init__(
        self,
        model: str,
        compute_type: str = "int8",
        cpu_threads: int = 0,
        language: str = "en",
        beam_size: int = 1,
        vad: bool = True,
        initial_prompt: str = "",
    ):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
        )
        self.language = language
        self.beam_size = beam_size
        self.vad = vad
        self.initial_prompt = initial_prompt or None
        # CTranslate2 models are not safe to call from several threads at once.
        self._lock = threading.Lock()

    def transcribe(self, audio: bytes) -> Transcript:
        """Transcribe one utterance."""
        with self._lock:
            segments, info = self._model.transcribe(
                io.BytesIO(audio),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text=text, duration_s=float(info.duration), language=info.language)


_stt: WhisperSTT | None = None
_stt_lock = threading.Lock()


def get_stt(settings: Settings) -> WhisperSTT:
    """Process-wide STT; uses models/faster-whisper-<size> when present, else downloads."""
    global _stt
    with _stt_lock:
        if _stt is None:
            s = settings.stt
            local = settings.paths.resolve(s.model_dir)
            model = str(local) if (local / "model.bin").exists() else s.model_size
            log.info("loading whisper model=%s compute=%s", model, s.compute_type)
            _stt = WhisperSTT(
                model,
                s.compute_type,
                s.cpu_threads,
                s.language,
                s.beam_size,
                s.vad,
                s.initial_prompt,
            )
        return _stt
