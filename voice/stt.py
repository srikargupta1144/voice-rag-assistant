"""
File: voice/stt.py

Description:
    Speech-to-text with faster-whisper (small, int8, CPU) and Silero voice activity detection.

    Any container PyAV can decode is accepted (WAV from the API contract, or WebM/Opus
    straight from a browser). VAD drops leading and trailing silence and long pauses, which
    both speeds up decoding and stops Whisper inventing words in silence.

Flow:
    api.routes /ask/audio and /ws/voice
        -> get_stt(settings).transcribe(audio_bytes)
            -> WhisperModel.transcribe(BytesIO, vad_filter=True, beam_size=1, language="en")
            -> Transcript(text, duration_s, language)

Libraries used:
    io - standard library; wraps audio bytes in a file-like buffer for Whisper
    logging - standard library; module logger for model loading
    threading - standard library; locks for the model and the process-wide instance
    dataclasses - standard library; dataclass for the Transcript record
    typing - standard library; TYPE_CHECKING and Protocol
    faster_whisper - third party; WhisperModel for speech recognition (imported on first use)
    configs - project; Settings type (type checking only)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Class name: Transcript
    Purpose: Result of one transcription: the text, the audio duration in seconds and the
             language.
    """

    text: str
    duration_s: float
    language: str


class SpeechToText(Protocol):
    """
    Class name: SpeechToText
    Purpose: Interface for any speech-to-text component (the real model or a test fake).
    """

    def transcribe(self, audio: bytes) -> Transcript:
        """
        Function name: SpeechToText.transcribe
        Purpose: Transcribe one recorded utterance.
        Parameters:
            audio (bytes): Encoded audio file contents.
        Returns:
            Transcript: The recognised text with duration and language.
        """
        ...


class WhisperSTT:
    """
    Class name: WhisperSTT
    Purpose: faster-whisper wrapper; the model loads once and is shared across requests.
    """

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
        """
        Function name: WhisperSTT.__init__
        Purpose: Load the Whisper model on the CPU and store the decoding options.
        Parameters:
            model (str): Local model folder or model size name to download.
            compute_type (str): CTranslate2 compute type, e.g. "int8".
            cpu_threads (int): Number of CPU threads; 0 uses the library default.
            language (str): Language code of the speech.
            beam_size (int): Beam size; 1 means greedy decoding.
            vad (bool): Whether to apply voice activity detection.
            initial_prompt (str): Domain vocabulary shown to Whisper; "" means none.
        Returns:
            None
        """
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
        """
        Function name: WhisperSTT.transcribe
        Purpose: Transcribe one utterance under the model lock and join the segment texts.
        Parameters:
            audio (bytes): Encoded audio file contents in any format PyAV can decode.
        Returns:
            Transcript: The recognised text, audio duration and detected language.
        """
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
    """
    Function name: get_stt
    Purpose: Return the process-wide speech-to-text model, creating it on first call. It uses
             the local model folder (models/faster-whisper-<size>) when it holds model.bin,
             and otherwise downloads the model by size name.
    Parameters:
        settings (Settings): Application settings (stt and paths groups are used).
    Returns:
        WhisperSTT: The shared speech-to-text model.
    """
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
