"""
File: voice/tts.py

Description:
    Text-to-speech with Piper (ONNX voice, CPU), one sentence at a time.

    synthesize() returns a complete WAV per sentence, so the WebSocket client can play each
    one as soon as it arrives while the next sentence is still being generated.

Flow:
    api.routes /ask/audio   -> for each sentence: get_tts(settings).synthesize(sentence)
                            -> voice.audio.concat_wav(...)
    api.routes /ws/voice    -> synthesize(sentence) as each sentence leaves the LLM
    scripts.demo_e2e        -> synthesize(question) to create the test recording

Libraries used:
    logging - standard library; module logger for voice loading
    threading - standard library; locks for the voice and the process-wide instance
    typing - standard library; TYPE_CHECKING and Protocol
    piper - third party; PiperVoice and SynthesisConfig for speech (imported on first use)
    configs - project; Settings type (type checking only)
    voice.audio - project; pcm_to_wav wraps the synthesized audio in a WAV container

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Protocol

from voice.audio import pcm_to_wav

if TYPE_CHECKING:
    from configs import Settings

log = logging.getLogger(__name__)


class TextToSpeech(Protocol):
    """
    Class name: TextToSpeech
    Purpose: Interface for any text-to-speech component (the real voice or a test fake).
    """

    sample_rate: int

    def synthesize(self, text: str) -> bytes:
        """
        Function name: TextToSpeech.synthesize
        Purpose: Convert text to speech audio.
        Parameters:
            text (str): Text to speak.
        Returns:
            bytes: WAV audio of the spoken text.
        """
        ...


class PiperTTS:
    """
    Class name: PiperTTS
    Purpose: piper-tts voice wrapper producing 16-bit mono WAV bytes.
    """

    def __init__(self, voice_path: str, length_scale: float = 1.0):
        """
        Function name: PiperTTS.__init__
        Purpose: Load the Piper voice and set up the synthesis options and sample rate.
        Parameters:
            voice_path (str): Path to the Piper .onnx voice file.
            length_scale (float): Speaking speed factor; values above 1 speak slower.
        Returns:
            None
        """
        from piper import PiperVoice, SynthesisConfig

        self._voice = PiperVoice.load(voice_path)
        self._config = SynthesisConfig(length_scale=length_scale)
        self.sample_rate = int(self._voice.config.sample_rate)
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        """
        Function name: PiperTTS.synthesize
        Purpose: Synthesize text under the voice lock and return it as WAV bytes. Empty text
                 gives an empty WAV.
        Parameters:
            text (str): Text to speak.
        Returns:
            bytes: 16-bit mono WAV audio.
        """
        with self._lock:
            pcm = b"".join(
                chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, self._config)
            )
        return pcm_to_wav(pcm, self.sample_rate)


_tts: PiperTTS | None = None
_tts_lock = threading.Lock()


def get_tts(settings: Settings) -> PiperTTS:
    """
    Function name: get_tts
    Purpose: Return the process-wide text-to-speech voice for the configured voice
             (models/piper/<voice>.onnx), creating it on first call.
    Parameters:
        settings (Settings): Application settings (tts and paths groups are used).
    Returns:
        PiperTTS: The shared text-to-speech voice. Raises FileNotFoundError when the voice
                  file is missing.
    """
    global _tts
    with _tts_lock:
        if _tts is None:
            path = settings.paths.resolve(settings.tts.voice_dir) / f"{settings.tts.voice}.onnx"
            if not path.exists():
                raise FileNotFoundError(
                    f"Piper voice missing at {path}; run: uv run python -m scripts.download_models"
                )
            log.info("loading piper voice=%s", path.name)
            _tts = PiperTTS(str(path), settings.tts.length_scale)
        return _tts
