"""
File: voice/audio.py

Description:
    WAV helpers shared by TTS, the API and the end-to-end demo script.

Flow:
    voice.tts.PiperTTS.synthesize()      -> pcm_to_wav(pcm, rate)
    api.routes /ask/audio                -> concat_wav([...sentence wavs]) for one reply file
    scripts.demo_e2e                     -> wav_duration_s() for reporting

Libraries used:
    io - standard library; in-memory byte buffers for WAV data
    wave - standard library; reads and writes WAV containers

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
import wave


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Function name: pcm_to_wav
    Purpose: Wrap raw little-endian PCM audio in a WAV container.
    Parameters:
        pcm (bytes): Raw PCM sample data.
        sample_rate (int): Samples per second.
        channels (int): Number of audio channels.
        sample_width (int): Bytes per sample (2 for 16-bit audio).
    Returns:
        bytes: The complete WAV file.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """
    Function name: read_wav
    Purpose: Read a WAV file held in memory and return its PCM data and format.
    Parameters:
        data (bytes): The complete WAV file.
    Returns:
        tuple[bytes, int, int, int]: (pcm, sample_rate, channels, sample_width).
    """
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()


def concat_wav(wavs: list[bytes], silence_ms: int = 150) -> bytes:
    """
    Function name: concat_wav
    Purpose: Join WAV files of identical format, with a short silence between sentences.
             The format of the last file is used for the output.
    Parameters:
        wavs (list[bytes]): WAV files to join, in order.
        silence_ms (int): Length of the silence inserted between files, in milliseconds.
    Returns:
        bytes: One WAV file, or b"" when the list is empty.
    """
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
    """
    Function name: wav_duration_s
    Purpose: Return the length of a WAV file in seconds.
    Parameters:
        data (bytes): The complete WAV file.
    Returns:
        float: Duration in seconds.
    """
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())
