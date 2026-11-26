"""
File: tests/functional/test_api_real.py

Description:
    Functional tests of the HTTP and WebSocket API with the real retriever, the real
    faster-whisper and the real Piper voice, but a fake LLM with a fixed, cited answer, so
    the results are deterministic and no Ollama is needed.

    Run with: .venv\\Scripts\\python -m pytest -m models

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    base64 - standard library; decodes the spoken answer of /ask/audio
    json - standard library; decodes WebSocket text frames
    pytest - third party; the models marker and fixtures
    fastapi.testclient - third party; in-process HTTP and WebSocket client
    api.main - project; create_app
    api.services - project; the Services class the test services extend
    rag.pipeline - project; AnswerPipeline built over the real retriever
    rag.prompts - project; the NOT_FOUND reply
    scripts.demo_e2e - project; word_error_rate
    tests.conftest - project; FakeLLM, the scripted LLM
    tests.functional.conftest - project; make_settings
    voice.audio - project; wav_duration_s

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import Services
from rag.pipeline import AnswerPipeline
from rag.prompts import NOT_FOUND
from scripts.demo_e2e import word_error_rate
from tests.conftest import FakeLLM
from tests.functional.conftest import make_settings
from voice.audio import wav_duration_s

pytestmark = pytest.mark.models

QUESTION = "How much is an outgoing international wire at Kova?"
REPLY = "An outgoing international wire costs 45 dollars [1]. Incoming wires are free [1]."
NOT_FOUND_QUESTION = "Does Kova offer car loans?"
FAQ = "kova_bank_account_faq.md"

# Piper adds random variation, so the same text is heard slightly differently on each run;
# "outgoing" was once heard as "alt going" (two word errors in nine, WER 0.22).
MAX_WER = 0.3


def transcript_is_close(text: str) -> bool:
    """
    Function name: transcript_is_close
    Purpose: Check that a transcript of QUESTION is close enough to answer it: a word error
        rate of at most MAX_WER and the key words "international", "wire" and "Kova".
    Parameters:
        text (str): Transcript returned by the API.
    Returns:
        bool: True when the transcript is close to the spoken question.
    """
    words = text.lower()
    keys = all(k in words for k in ("international", "wire", "kova"))
    return keys and word_error_rate(QUESTION, text) <= MAX_WER


class RealModelServices(Services):
    """
    Class name: RealModelServices
    Purpose: Services that hold an already built pipeline and the session STT and TTS, so
        each test reuses the loaded models instead of loading them again.
    """

    def __init__(self, settings, pipeline, stt, tts):
        """
        Function name: RealModelServices.__init__
        Purpose: Store the settings and the prebuilt components.
        Parameters:
            settings (Settings): Application settings.
            pipeline (AnswerPipeline): Pipeline with the real retriever and a fake LLM.
            stt (WhisperSTT): Real speech-to-text model.
            tts (PiperTTS): Real text-to-speech model.
        Returns:
            None: Initialises the instance.
        """
        super().__init__(settings)
        self._pipeline = pipeline
        self._stt = stt
        self._tts = tts


@pytest.fixture
def api(real_retriever, real_stt, real_tts):
    """
    Function name: api
    Purpose: Fixture that builds the application around the real retriever, STT and TTS and
        a fresh FakeLLM with a cited reply. Rewriting is off so the LLM is called only for
        the answer.
    Parameters:
        real_retriever (HybridRetriever): Retriever over the committed index.
        real_stt (WhisperSTT): Session speech-to-text model.
        real_tts (PiperTTS): Session text-to-speech model.
    Returns:
        tuple[TestClient, FakeLLM]: The test client and the fake LLM behind it.
    """
    settings = make_settings(api={"warm_up": False})
    llm = FakeLLM(reply=REPLY)
    r = settings.retrieval
    pipeline = AnswerPipeline(
        real_retriever,
        llm,
        top_k=r.top_k,
        context_k=r.context_k,
        not_found_threshold=r.not_found_threshold,
        rewrite_mode="never",
    )
    svc = RealModelServices(settings, pipeline, real_stt, real_tts)
    with TestClient(create_app(settings=settings, services=svc)) as client:
        yield client, llm


@pytest.fixture(scope="module")
def question_wav(real_tts) -> bytes:
    """
    Function name: question_wav
    Purpose: Fixture with the test question spoken by Piper, used as the recording.
    Parameters:
        real_tts (PiperTTS): Session text-to-speech model.
    Returns:
        bytes: WAV audio of QUESTION.
    """
    return real_tts.synthesize(QUESTION)


def test_ask_returns_passages_with_all_stage_scores(api):
    """
    Function name: test_ask_returns_passages_with_all_stage_scores
    Purpose: Check that POST /ask with the real retriever returns five passages, each with
        RRF and re-ranker scores, that dense and BM25 scores both appear, and that the [1]
        marker is turned into a citation of the Kova FAQ.
    Parameters:
        api (tuple[TestClient, FakeLLM]): Test client and fake LLM.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, llm = api
    r = client.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found"] is True
    assert len(llm.calls) == 1
    passages = body["passages"]
    assert len(passages) == 5
    assert [p["rank"] for p in passages] == [1, 2, 3, 4, 5]
    for p in passages:
        assert {"rrf", "rerank"} <= set(p["scores"])
    assert any("dense" in p["scores"] for p in passages)
    assert any("bm25" in p["scores"] for p in passages)
    assert {"dense", "bm25", "rrf", "rerank"} <= set(passages[0]["scores"])
    assert passages[0]["source"] == FAQ
    assert body["top_score"] == pytest.approx(passages[0]["scores"]["rerank"], abs=1e-3)
    assert [c["source"] for c in body["citations"]] == [FAQ]
    assert f"[{body['citations'][0]['label']}]" in body["answer"]
    assert "[1]" not in body["answer"] and "[" not in body["spoken"]
    assert body["timings"]["retrieval_ms"] > 0


def test_ask_not_found_skips_the_llm(api):
    """
    Function name: test_ask_not_found_skips_the_llm
    Purpose: Check that a question the documents do not answer returns the not-found reply
        with no citations and never calls the LLM, while still listing the passages.
    Parameters:
        api (tuple[TestClient, FakeLLM]): Test client and fake LLM.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, llm = api
    r = client.post("/ask", json={"question": NOT_FOUND_QUESTION})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found"] is False
    assert body["answer"] == NOT_FOUND
    assert body["citations"] == []
    assert llm.calls == []
    assert body["top_score"] < make_settings().retrieval.not_found_threshold
    assert len(body["passages"]) == 5


def test_ask_audio_with_piper_recording(api, question_wav):
    """
    Function name: test_ask_audio_with_piper_recording
    Purpose: Check that POST /ask/audio with a Piper recording returns a transcript close to
        the question, a cited answer from the Kova FAQ and a spoken answer WAV of a
        plausible length.
    Parameters:
        api (tuple[TestClient, FakeLLM]): Test client and fake LLM.
        question_wav (bytes): The question spoken by Piper.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = api
    files = {"file": ("question.wav", question_wav, "audio/wav")}
    r = client.post("/ask/audio", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert transcript_is_close(body["transcript"]), body["transcript"]
    assert body["question"] == body["transcript"]
    assert body["found"] is True
    assert [c["source"] for c in body["citations"]] == [FAQ]
    wav = base64.b64decode(body["audio_wav_base64"])
    assert wav[:4] == b"RIFF"
    assert body["audio_seconds"] == pytest.approx(wav_duration_s(wav), abs=0.01)
    assert body["audio_seconds"] > 2.0  # two spoken sentences
    assert body["timings"]["stt_ms"] > 0 and body["timings"]["tts_ms"] > 0


def test_websocket_turn_streams_sentence_audio(api, question_wav):
    """
    Function name: test_websocket_turn_streams_sentence_audio
    Purpose: Check one WebSocket voice turn with a Piper recording: the server sends start,
        transcript and query messages, then one JSON sentence message followed by one WAV
        frame per spoken sentence, then the result with passages and citations.
    Parameters:
        api (tuple[TestClient, FakeLLM]): Test client and fake LLM.
        question_wav (bytes): The question spoken by Piper.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = api
    kinds: list[str] = []
    sentences: list[str] = []
    frames: list[bytes] = []
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_bytes(question_wav)
        while True:
            message = ws.receive()
            if message.get("bytes") is not None:
                kinds.append("audio")
                frames.append(message["bytes"])
                continue
            data = json.loads(message["text"])
            kinds.append(data["type"])
            assert data["type"] != "error", data
            if data["type"] == "transcript":
                assert transcript_is_close(data["text"]), data["text"]
            if data["type"] == "sentence":
                sentences.append(data["text"])
            if data["type"] == "result":
                result = data
                break
    assert kinds[:3] == ["start", "transcript", "query"]
    assert len(sentences) == 2
    assert len(frames) == len(sentences)
    # Every sentence message is followed by its audio frame.
    for i, kind in enumerate(kinds):
        if kind == "sentence":
            assert kinds[i + 1] == "audio"
    for wav in frames:
        assert wav[:4] == b"RIFF" and wav_duration_s(wav) > 0.5
    assert result["found"] is True
    assert [c["source"] for c in result["citations"]] == [FAQ]
    assert len(result["passages"]) == 5
    assert "rerank" in result["passages"][0]["scores"]
