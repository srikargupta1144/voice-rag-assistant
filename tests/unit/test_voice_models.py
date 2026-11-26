"""Opt-in check with the real voice models: Piper speaks a sentence, faster-whisper hears it.

Run with `uv run pytest -m models` after `uv run python -m scripts.download_models`.
"""

import pytest

from configs import Settings
from voice.audio import wav_duration_s

pytestmark = pytest.mark.models


def test_piper_to_whisper_roundtrip():
    from voice.stt import get_stt
    from voice.tts import get_tts

    settings = Settings(_env_file=None)
    wav = get_tts(settings).synthesize("What is the late fee on the Kova card?")
    assert wav_duration_s(wav) > 1.0
    text = get_stt(settings).transcribe(wav).text.lower()
    assert "late fee" in text and "card" in text
