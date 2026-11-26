"""
File: tests/unit/test_api.py

Description:
    Unit tests for the API routes with the LLM, STT and TTS mocked (no models, no network).

Libraries used:
    base64 - standard library; decodes the base64 WAV reply
    json - standard library; parses WebSocket text messages (imported inside _drain)
    time - standard library; delays the slow fake LLM to trigger the timeout
    pytest - third party; fixture definitions
    fastapi.testclient - third party; HTTP and WebSocket test client
    api.main - project; the application factory under test
    api.services - project; the Services class that the fake services extend
    rag.llm - project; LLMError raised by the failing fake LLM
    rag.pipeline - project; the answer pipeline wired into the app
    tests.conftest - project; the FakeLLM test double
    voice.audio - project; WAV building and reading helpers
    voice.stt - project; the Transcript type returned by the fake STT

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  Page test checks the new page text.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import base64
import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import Services
from rag.llm import LLMError
from rag.pipeline import AnswerPipeline
from tests.conftest import FakeLLM
from voice.audio import pcm_to_wav, read_wav
from voice.stt import Transcript

TONE = pcm_to_wav(b"\x01\x00" * 1600, 16000)  # 0.1 s of near-silence


class FakeSTT:
    """
    Class name: FakeSTT
    Purpose: Speech-to-text test double that returns a fixed transcript.
    """

    def __init__(self, text="How much do international wires cost?"):
        """
        Function name: FakeSTT.__init__
        Purpose: Store the transcript text to return.
        Parameters:
            text (str): Transcript text returned by transcribe().
        Returns:
            None: Initialises the instance.
        """
        self.text = text

    def transcribe(self, audio: bytes) -> Transcript:
        """
        Function name: FakeSTT.transcribe
        Purpose: Check that audio bytes were given and return the fixed transcript.
        Parameters:
            audio (bytes): Uploaded audio; must not be empty.
        Returns:
            Transcript: The fixed text with duration 1.0 and language "en".
        """
        assert audio
        return Transcript(self.text, 1.0, "en")


class FakeTTS:
    """
    Class name: FakeTTS
    Purpose: Text-to-speech test double that records each sentence and returns
        0.05 seconds of silent 16 kHz WAV audio.
    """

    sample_rate = 16000

    def __init__(self):
        """
        Function name: FakeTTS.__init__
        Purpose: Start an empty list of spoken sentences.
        Parameters:
            None
        Returns:
            None: Initialises the instance.
        """
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> bytes:
        """
        Function name: FakeTTS.synthesize
        Purpose: Record the sentence and return 0.05 seconds of silent WAV audio.
        Parameters:
            text (str): Sentence to speak.
        Returns:
            bytes: A WAV file at the fake sample rate.
        """
        self.spoken.append(text)
        return pcm_to_wav(b"\x00\x00" * 800, self.sample_rate)  # 0.05 s per sentence


class FakeServices(Services):
    """
    Class name: FakeServices
    Purpose: Services with an injected pipeline and fake STT and TTS, so no model is
        loaded; ingest only reports the chunk count.
    """

    def __init__(self, settings, pipeline, stt=None, tts=None):
        """
        Function name: FakeServices.__init__
        Purpose: Initialise the base services and inject the given pipeline, STT and TTS.
        Parameters:
            settings (Settings): Test settings.
            pipeline (AnswerPipeline): Pipeline to serve.
            stt (FakeSTT | None): Speech-to-text double; None uses a default FakeSTT.
            tts (FakeTTS | None): Text-to-speech double; None uses a new FakeTTS.
        Returns:
            None: Initialises the instance.
        """
        super().__init__(settings)
        self._pipeline = pipeline
        self._stt = stt or FakeSTT()
        self._tts = tts or FakeTTS()

    def ingest(self):
        """
        Function name: FakeServices.ingest
        Purpose: Report the number of chunks in the retriever instead of rebuilding the index.
        Parameters:
            None
        Returns:
            dict: A manifest with the "chunks" count.
        """
        return {"chunks": len(self._pipeline.retriever.chunks)}


REPLY = "International wires cost 45 dollars [1]. Domestic wires are not covered here [2]."


@pytest.fixture
def make_client(settings, retriever):
    """
    Function name: make_client
    Purpose: Fixture that provides a factory for a test client wired to an answer pipeline
        over the test retriever, with optional LLM, STT and request timeout overrides.
    Parameters:
        settings (Settings): Test settings fixture.
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        Callable: A factory _make(reply, llm, stt, timeout) that returns
            (TestClient, FakeServices).
    """

    def _make(reply=REPLY, llm=None, stt=None, timeout=None):
        """
        Function name: _make
        Purpose: Build the pipeline, fake services and FastAPI test client. When a timeout is
            given, the API request timeout setting is replaced with it.
        Parameters:
            reply (str): Scripted LLM reply; defaults to REPLY.
            llm (FakeLLM | None): LLM double; None uses FakeLLM(reply).
            stt (FakeSTT | None): Speech-to-text double; None uses a default FakeSTT.
            timeout (float | None): Request timeout in seconds; None keeps the setting.
        Returns:
            tuple[TestClient, FakeServices]: The test client and the services behind it.
        """
        pipeline = AnswerPipeline(
            retriever,
            llm or FakeLLM(reply),
            top_k=3,
            context_k=2,
            not_found_threshold=-1.0,
            rewrite_mode="never",
        )
        s = settings
        if timeout is not None:
            s = settings.model_copy(
                update={"api": settings.api.model_copy(update={"request_timeout_s": timeout})}
            )
        svc = FakeServices(s, pipeline, stt=stt)
        return TestClient(create_app(settings=s, services=svc)), svc

    return _make


def test_health(make_client):
    """
    Function name: test_health
    Purpose: Check that /health reports status ok, the chunk count and a reachable LLM, and
        sets a request id header.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["chunks"] == 4 and body["llm_reachable"] is True
    assert r.headers["X-Request-ID"]


def test_request_id_is_echoed(make_client):
    """
    Function name: test_request_id_is_echoed
    Purpose: Check that a safe X-Request-ID header is echoed back and an unsafe one is replaced.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    assert (
        client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"]
        == "abc-123"
    )
    # unsafe ids are replaced
    assert (
        client.get("/health", headers={"X-Request-ID": "bad id!"}).headers["X-Request-ID"]
        != "bad id!"
    )


def test_ask_returns_answer_citations_and_timings(make_client):
    """
    Function name: test_ask_returns_answer_citations_and_timings
    Purpose: Check that /ask returns a found answer with labelled citations, a spoken text
        without brackets, stage timings and the request id.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.post("/ask", json={"question": "How much do international wires cost?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["citations"][0]["source"] == "faq.md"
    assert "[faq.md, Wires]" in body["answer"]
    assert "[" not in body["spoken"]
    assert {"retrieval_ms", "llm_first_token_ms", "total_ms"} <= set(body["timings"])
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_ask_validation(make_client):
    """
    Function name: test_ask_validation
    Purpose: Check that /ask rejects an empty question and an unknown mode with 422.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422


def test_ask_llm_down_is_503(make_client):
    """
    Function name: test_ask_llm_down_is_503
    Purpose: Check that /ask returns 503 when the LLM raises an LLMError.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """

    class Down(FakeLLM):
        """
        Class name: Down
        Purpose: Fake LLM whose stream raises LLMError, to simulate an unreachable LLM server.
        """

        def stream(self, *a, **k):
            """
            Function name: Down.stream
            Purpose: Raise LLMError instead of streaming a reply.
            Parameters:
                a (tuple): Positional arguments, ignored.
                k (dict): Keyword arguments, ignored.
            Returns:
                Iterator[str]: never returns; always raises LLMError.
            """
            raise LLMError("connection refused")

    client, _ = make_client(llm=Down())
    assert client.post("/ask", json={"question": "wires cost"}).status_code == 503


def test_ask_timeout_is_504(make_client):
    """
    Function name: test_ask_timeout_is_504
    Purpose: Check that /ask returns 504 when the LLM is slower than the request timeout.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """

    class Slow(FakeLLM):
        """
        Class name: Slow
        Purpose: Fake LLM that waits 0.5 seconds before its first piece, to trigger the request
            timeout.
        """

        def stream(self, *a, **k):
            """
            Function name: Slow.stream
            Purpose: Wait 0.5 seconds and then yield one piece of text.
            Parameters:
                a (tuple): Positional arguments, ignored.
                k (dict): Keyword arguments, ignored.
            Returns:
                Iterator[str]: yields the single piece "late".
            """
            time.sleep(0.5)
            yield "late"

    client, _ = make_client(llm=Slow(), timeout=0.1)
    assert client.post("/ask", json={"question": "wires cost"}).status_code == 504


def test_ask_audio_returns_transcript_answer_and_wav(make_client):
    """
    Function name: test_ask_audio_returns_transcript_answer_and_wav
    Purpose: Check that /ask/audio returns the transcript, a found answer with citations, a
        mono 16 kHz WAV reply, one spoken sentence per answer sentence without citations,
        and the STT and TTS timings.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, svc = make_client()
    r = client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "How much do international wires cost?"
    assert body["citations"] and body["found"]
    wav = base64.b64decode(body["audio_wav_base64"])
    _, rate, channels, _ = read_wav(wav)
    assert rate == 16000 and channels == 1 and body["audio_seconds"] > 0.1
    assert svc._tts.spoken == [
        "International wires cost 45 dollars.",
        "Domestic wires are not covered here.",
    ]
    assert {"stt_ms", "tts_ms", "tts_first_sentence_done_ms"} <= set(body["timings"])


def test_ask_audio_rejects_empty_and_silent(make_client):
    """
    Function name: test_ask_audio_rejects_empty_and_silent
    Purpose: Check that /ask/audio returns 400 for an empty upload and 422 for an empty transcript.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    assert client.post("/ask/audio", files={"file": ("q.wav", b"", "audio/wav")}).status_code == 400
    client, _ = make_client(stt=FakeSTT(text=""))
    assert (
        client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")}).status_code == 422
    )


def _drain(ws):
    """
    Function name: _drain
    Purpose: Collect WebSocket messages until the result (or an error) arrives. Binary
        frames are recorded as audio and text frames are parsed as JSON events.
    Parameters:
        ws (WebSocketTestSession): Open test WebSocket connection.
    Returns:
        list[tuple]: ("audio", bytes) or (event type, event dict) pairs in arrival order.
    """
    out = []
    while True:
        m = ws.receive()
        if m.get("bytes") is not None:
            out.append(("audio", m["bytes"]))
            continue
        import json

        event = json.loads(m["text"])
        out.append((event["type"], event))
        if event["type"] in ("result", "error"):
            return out


def test_ws_voice_streams_sentence_audio_for_text(make_client):
    """
    Function name: test_ws_voice_streams_sentence_audio_for_text
    Purpose: Check that a text question over /ws/voice streams start, query, then a sentence
        and its audio for each sentence, and a found result with citations and the
        first audio timing.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "How much do international wires cost?"})
        msgs = _drain(ws)
    kinds = [k for k, _ in msgs]
    assert kinds == ["start", "query", "sentence", "audio", "sentence", "audio", "result"]
    result = msgs[-1][1]
    assert result["found"] and result["citations"]
    assert "first_audio_ms" in result["timings"]


def test_ws_voice_accepts_audio_bytes(make_client):
    """
    Function name: test_ws_voice_accepts_audio_bytes
    Purpose: Check that audio bytes sent over /ws/voice give a transcript event with the
        spoken question and an STT timing in the result.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_bytes(TONE)
        msgs = _drain(ws)
    assert msgs[1][0] == "transcript"
    assert msgs[1][1]["text"] == "How much do international wires cost?"
    assert "stt_ms" in msgs[-1][1]["timings"]


def test_ws_voice_bad_message_reports_error(make_client):
    """
    Function name: test_ws_voice_bad_message_reports_error
    Purpose: Check that a JSON message without a question ends with an error event.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"nothing": True})
        msgs = _drain(ws)
    assert msgs[-1][0] == "error"


def test_static_page_served(make_client):
    """
    Function name: test_static_page_served
    Purpose: Check that the root path serves the voice page with its microphone prompt text.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.get("/")
    assert r.status_code == 200 and "Tap the microphone" in r.text
