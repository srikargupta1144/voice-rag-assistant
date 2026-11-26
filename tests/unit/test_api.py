"""API routes with the LLM, STT and TTS mocked (no models, no network)."""

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
    def __init__(self, text="How much do international wires cost?"):
        self.text = text

    def transcribe(self, audio: bytes) -> Transcript:
        assert audio
        return Transcript(self.text, 1.0, "en")


class FakeTTS:
    sample_rate = 16000

    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return pcm_to_wav(b"\x00\x00" * 800, self.sample_rate)  # 0.05 s per sentence


class FakeServices(Services):
    def __init__(self, settings, pipeline, stt=None, tts=None):
        super().__init__(settings)
        self._pipeline = pipeline
        self._stt = stt or FakeSTT()
        self._tts = tts or FakeTTS()

    def ingest(self):
        return {"chunks": len(self._pipeline.retriever.chunks)}


REPLY = "International wires cost 45 dollars [1]. Domestic wires are not covered here [2]."


@pytest.fixture
def make_client(settings, retriever):
    def _make(reply=REPLY, llm=None, stt=None, timeout=None):
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
    client, _ = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["chunks"] == 4 and body["llm_reachable"] is True
    assert r.headers["X-Request-ID"]


def test_request_id_is_echoed(make_client):
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
    client, _ = make_client()
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422


def test_ask_llm_down_is_503(make_client):
    class Down(FakeLLM):
        def stream(self, *a, **k):
            raise LLMError("connection refused")

    client, _ = make_client(llm=Down())
    assert client.post("/ask", json={"question": "wires cost"}).status_code == 503


def test_ask_timeout_is_504(make_client):
    class Slow(FakeLLM):
        def stream(self, *a, **k):
            time.sleep(0.5)
            yield "late"

    client, _ = make_client(llm=Slow(), timeout=0.1)
    assert client.post("/ask", json={"question": "wires cost"}).status_code == 504


def test_ask_audio_returns_transcript_answer_and_wav(make_client):
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
    client, _ = make_client()
    assert client.post("/ask/audio", files={"file": ("q.wav", b"", "audio/wav")}).status_code == 400
    client, _ = make_client(stt=FakeSTT(text=""))
    assert (
        client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")}).status_code == 422
    )


def _drain(ws):
    """Collect messages until the result (or an error) arrives."""
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
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_bytes(TONE)
        msgs = _drain(ws)
    assert msgs[1][0] == "transcript"
    assert msgs[1][1]["text"] == "How much do international wires cost?"
    assert "stt_ms" in msgs[-1][1]["timings"]


def test_ws_voice_bad_message_reports_error(make_client):
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"nothing": True})
        msgs = _drain(ws)
    assert msgs[-1][0] == "error"


def test_static_page_served(make_client):
    client, _ = make_client()
    r = client.get("/")
    assert r.status_code == 200 and "Tap the microphone" in r.text
