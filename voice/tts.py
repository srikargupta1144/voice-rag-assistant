"""Text-to-speech with Piper (ONNX voice, CPU), one sentence at a time.

Flow:
    api.routes /ask/audio   -> for each sentence: get_tts(settings).synthesize(sentence)
                            -> voice.audio.concat_wav(...)
    api.routes /ws/voice    -> synthesize(sentence) as each sentence leaves the LLM
    scripts.demo_e2e        -> synthesize(question) to create the test recording

synthesize() returns a complete WAV per sentence, so the WebSocket client can play each
one as soon as it arrives while the next sentence is still being generated.
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
    sample_rate: int

    def synthesize(self, text: str) -> bytes: ...


class PiperTTS:
    """piper-tts voice wrapper producing 16-bit mono WAV bytes."""

    def __init__(self, voice_path: str, length_scale: float = 1.0):
        from piper import PiperVoice, SynthesisConfig

        self._voice = PiperVoice.load(voice_path)
        self._config = SynthesisConfig(length_scale=length_scale)
        self.sample_rate = int(self._voice.config.sample_rate)
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        """WAV bytes for text (empty text gives an empty WAV)."""
        with self._lock:
            pcm = b"".join(
                chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, self._config)
            )
        return pcm_to_wav(pcm, self.sample_rate)


_tts: PiperTTS | None = None
_tts_lock = threading.Lock()


def get_tts(settings: Settings) -> PiperTTS:
    """Process-wide TTS for the configured voice (models/piper/<voice>.onnx)."""
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
