"""
File: tests/functional/test_voice_real.py

Description:
    Functional test of the real voice round trip: Piper speaks golden questions, faster-
    whisper transcribes them, and the word error rate against the spoken text stays low.
    Unlike tests/unit/test_voice_models.py (one sentence, key words only), this test scores
    whole transcripts on several questions, checks the WAV format Piper produces and checks
    that the transcript still retrieves the labelled document.

    Run with: .venv\\Scripts\\python -m pytest -m models

Libraries used:
    pytest - third party; the models marker and parametrised cases
    scripts.demo_e2e - project; word_error_rate, the same metric the demo script reports
    voice.audio - project; pcm_to_wav, read_wav and wav_duration_s

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

import pytest

from scripts.demo_e2e import word_error_rate
from voice.audio import pcm_to_wav, read_wav, wav_duration_s

pytestmark = pytest.mark.models

# Highest word error rate accepted for one question. Piper adds random variation, so the
# same text is heard slightly differently on each run: in testing Whisper small heard
# "account" as "count" (0.09) and "outgoing" as "alt going" (0.22) now and then.
MAX_WER = 0.3

CASES = [
    ("q22", "kova_bank_account_faq.md"),
    ("q07", "acme_home_insurance_policy.pdf"),
    ("q28", "kova_privacy_notice.txt"),
]


@pytest.mark.parametrize(("qid", "source"), CASES)
def test_piper_whisper_round_trip_has_low_wer(
    real_tts, real_stt, real_retriever, golden, qid, source
):
    """
    Function name: test_piper_whisper_round_trip_has_low_wer
    Purpose: Speak a golden question with Piper, transcribe it with faster-whisper, and
        check the word error rate, the detected language and duration, and that the
        transcript still retrieves the labelled source first.
    Parameters:
        real_tts (PiperTTS): Session text-to-speech model.
        real_stt (WhisperSTT): Session speech-to-text model.
        real_retriever (HybridRetriever): Retriever over the committed index.
        golden (dict[str, dict]): Golden questions by id.
        qid (str): Id of the golden question.
        source (str): Labelled source document of the question.
    Returns:
        None: The test passes or fails through assertions.
    """
    question = golden[qid]["question"]
    wav = real_tts.synthesize(question)
    transcript = real_stt.transcribe(wav)
    wer = word_error_rate(question, transcript.text)
    assert wer <= MAX_WER, f"WER {wer} for {transcript.text!r}"
    assert transcript.language == "en"
    assert transcript.duration_s == pytest.approx(wav_duration_s(wav), abs=0.1)
    top = real_retriever.retrieve(transcript.text, mode="full", top_k=1)[0]
    assert top.chunk.source == source


def test_piper_output_is_16_bit_mono_wav(real_tts):
    """
    Function name: test_piper_output_is_16_bit_mono_wav
    Purpose: Check that Piper returns a readable WAV with speech-length audio, which is the
        format the API sends to the browser.
    Parameters:
        real_tts (PiperTTS): Session text-to-speech model.
    Returns:
        None: The test passes or fails through assertions.
    """
    wav = real_tts.synthesize("Outgoing international wires cost 45 dollars.")
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    pcm, rate, channels, width = read_wav(wav)
    assert rate >= 16000
    assert (channels, width) == (1, 2)
    assert len(pcm) > 0
    assert 1.0 < wav_duration_s(wav) < 10.0


def test_whisper_hears_no_words_in_silence(real_stt):
    """
    Function name: test_whisper_hears_no_words_in_silence
    Purpose: Check that one second of silence gives an empty transcript (voice activity
        detection removes it), which the API turns into "no speech detected".
    Parameters:
        real_stt (WhisperSTT): Session speech-to-text model.
    Returns:
        None: The test passes or fails through assertions.
    """
    silence = pcm_to_wav(b"\x00\x00" * 16000, 16000)
    assert real_stt.transcribe(silence).text == ""
