"""
File: tests/unit/test_voice_models.py

Description:
    Opt-in check with the real voice models: Piper speaks a sentence, faster-whisper hears
    it.

    Run with `uv run pytest -m models` after `uv run python -m scripts.download_models`.

Libraries used:
    pytest - third party; the models marker that makes this test opt-in
    configs - project; the Settings class used to locate the models
    voice.audio - project; WAV duration helper
    voice.stt - project; the real speech-to-text model (imported inside the test)
    voice.tts - project; the real text-to-speech model (imported inside the test)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import pytest

from configs import Settings
from voice.audio import wav_duration_s

pytestmark = pytest.mark.models


def test_piper_to_whisper_roundtrip():
    """
    Function name: test_piper_to_whisper_roundtrip
    Purpose: Check that Piper speaks a question as more than one second of audio and that
        faster-whisper transcribes it back with the key words "late fee" and "card".
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    from voice.stt import get_stt
    from voice.tts import get_tts

    settings = Settings(_env_file=None)
    wav = get_tts(settings).synthesize("What is the late fee on the Kova card?")
    assert wav_duration_s(wav) > 1.0
    text = get_stt(settings).transcribe(wav).text.lower()
    assert "late fee" in text and "card" in text
