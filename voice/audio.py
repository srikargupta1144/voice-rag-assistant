"""WAV helpers shared by TTS, the API and the end-to-end demo script.

Flow:
    voice.tts.PiperTTS.synthesize()      -> pcm_to_wav(pcm, rate)
    api.routes /ask/audio                -> concat_wav([...sentence wavs]) for one reply file
    scripts.demo_e2e                     -> wav_duration_s() for reporting
"""

from __future__ import annotations

import io
import wave


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """(pcm, sample_rate, channels, sample_width) of a WAV file in memory."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()


def concat_wav(wavs: list[bytes], silence_ms: int = 150) -> bytes:
    """Join WAV files of identical format, with a short pause between sentences."""
    if not wavs:
        return b""
    pcm_parts = []
    rate = channels = width = 0
    for data in wavs:
        pcm, rate, channels, width = read_wav(data)
        pcm_parts.append(pcm)
    gap = b"\x00" * int(rate * silence_ms / 1000) * channels * width
    return pcm_to_wav(gap.join(pcm_parts), rate, channels, width)


def wav_duration_s(data: bytes) -> float:
    """Length of a WAV file in seconds."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())
