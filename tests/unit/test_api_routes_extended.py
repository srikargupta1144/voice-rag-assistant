"""
File: tests/unit/test_api_routes_extended.py

Description:
    Extra unit tests for api/routes.py: the WebSocket /ws/voice path, the /ask/audio error
    cases, /client-log, /ingest, /health, request timeouts and the request id header. The
    LLM, STT and TTS are fakes, so no model is loaded and there is no network access.

Libraries used:
    base64 - standard library; decodes the base64 WAV reply
    json - standard library; parses WebSocket text messages
    logging - standard library; handler that collects the api.routes log messages
    threading - standard library; waits for the answer worker threads
    time - standard library; delays the slow fake LLM to trigger the timeout
    pytest - third party; fixtures, parametrisation and log capture
    fastapi.testclient - third party; HTTP and WebSocket test client
    starlette.requests - third party; Request class patched to break the body read
    api.main - project; the application factory under test
    rag.llm - project; LLMError raised by the failing fake LLM
    rag.pipeline - project; the answer pipeline wired into the app
    tests.conftest - project; the FakeLLM test double
    tests.unit.test_api - project; FakeSTT, FakeTTS, FakeServices, TONE and REPLY fakes
    voice.audio - project; WAV reading helpers

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

import base64
import json
import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from api.main import create_app
from rag.llm import LLMError
from rag.pipeline import AnswerPipeline
from tests.conftest import FakeLLM
from tests.unit.test_api import REPLY, TONE, FakeServices, FakeSTT, FakeTTS
from voice.audio import read_wav


class DownLLM(FakeLLM):
    """
    Class name: DownLLM
    Purpose: Fake LLM whose stream raises LLMError, to simulate an unreachable LLM server.
    """

    def stream(self, *a, **k):
        """
        Function name: DownLLM.stream
        Purpose: Raise LLMError instead of streaming a reply.
        Parameters:
            a (tuple): Positional arguments, ignored.
            k (dict): Keyword arguments, ignored.
        Returns:
            Iterator[str]: never returns; always raises LLMError.
        """
        raise LLMError("connection refused")


class SlowLLM(FakeLLM):
    """
    Class name: SlowLLM
    Purpose: Fake LLM that waits 0.5 seconds before its first piece, to trigger the timeout.
    """

    def stream(self, *a, **k):
        """
        Function name: SlowLLM.stream
        Purpose: Wait 0.5 seconds and then yield one piece of text.
        Parameters:
            a (tuple): Positional arguments, ignored.
            k (dict): Keyword arguments, ignored.
        Returns:
            Iterator[str]: yields the single piece "late".
        """
        time.sleep(0.5)
        yield "late"


class MissingVoiceTTS(FakeTTS):
    """
    Class name: MissingVoiceTTS
    Purpose: Fake TTS that raises FileNotFoundError, as when the Piper voice file is missing.
    """

    def synthesize(self, text: str) -> bytes:
        """
        Function name: MissingVoiceTTS.synthesize
        Purpose: Raise FileNotFoundError instead of speaking.
        Parameters:
            text (str): Sentence to speak, ignored.
        Returns:
            bytes: never returns; always raises FileNotFoundError.
        """
        raise FileNotFoundError("voice missing")


@pytest.fixture
def make_client(settings, retriever):
    """
    Function name: make_client
    Purpose: Fixture that provides a factory for a test client wired to an answer pipeline
        over the test retriever, with optional LLM, STT, TTS, timeout and upload limit.
    Parameters:
        settings (Settings): Test settings fixture.
        retriever (HybridRetriever): Test retriever fixture.
    Returns:
        Callable: A factory _make(reply, llm, stt, tts, timeout, max_upload_mb) that returns
            (TestClient, FakeServices).
    """

    def _make(reply=REPLY, llm=None, stt=None, tts=None, timeout=None, max_upload_mb=None):
        """
        Function name: _make
        Purpose: Build the pipeline, fake services and FastAPI test client, replacing the API
            settings that were given.
        Parameters:
            reply (str): Scripted LLM reply; defaults to REPLY.
            llm (FakeLLM | None): LLM double; None uses FakeLLM(reply).
            stt (FakeSTT | None): Speech-to-text double; None uses a default FakeSTT.
            tts (FakeTTS | None): Text-to-speech double; None uses a new FakeTTS.
            timeout (float | None): Request timeout in seconds; None keeps the setting.
            max_upload_mb (int | None): Upload limit in MB; None keeps the setting.
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
        update = {}
        if timeout is not None:
            update["request_timeout_s"] = timeout
        if max_upload_mb is not None:
            update["max_upload_mb"] = max_upload_mb
        s = settings.model_copy(update={"api": settings.api.model_copy(update=update)})
        svc = FakeServices(s, pipeline, stt=stt, tts=tts)
        return TestClient(create_app(settings=s, services=svc)), svc

    return _make


@pytest.fixture
def route_logs():
    """
    Function name: route_logs
    Purpose: Fixture that collects the messages logged by the api.routes logger. A handler on
        that logger is used because create_app replaces the root handlers, which would remove
        the pytest log capture handler.
    Parameters:
        None
    Returns:
        list[str]: The formatted messages, filled while the test runs.
    """
    messages: list[str] = []

    class Collect(logging.Handler):
        """
        Class name: Collect
        Purpose: Logging handler that appends each formatted message to a list.
        """

        def emit(self, record):
            """
            Function name: Collect.emit
            Purpose: Store the formatted message of one log record.
            Parameters:
                record (LogRecord): The log record.
            Returns:
                None
            """
            messages.append(record.getMessage())

    logger = logging.getLogger("api.routes")
    handler = Collect()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield messages
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def _join_workers():
    """
    Function name: _join_workers
    Purpose: Wait for the answer worker threads of the WebSocket route, so that a slow worker
        does not finish after the test client has closed its event loop.
    Parameters:
        None
    Returns:
        None
    """
    for t in threading.enumerate():
        if t.name.startswith("voice-"):
            t.join(timeout=5)


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
        event = json.loads(m["text"])
        out.append((event["type"], event))
        if event["type"] in ("result", "error"):
            return out


def test_ws_audio_full_event_sequence(make_client):
    """
    Function name: test_ws_audio_full_event_sequence
    Purpose: Check the exact event order for an audio message: start, transcript, query, then
        a sentence and a WAV frame per sentence, and a final result with the request id.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, svc = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_bytes(TONE)
        msgs = _drain(ws)
    kinds = [k for k, _ in msgs]
    assert kinds == [
        "start",
        "transcript",
        "query",
        "sentence",
        "audio",
        "sentence",
        "audio",
        "result",
    ]
    rid = msgs[0][1]["request_id"]
    assert len(rid) == 12 and msgs[-1][1]["request_id"] == rid
    assert msgs[1][1]["text"] == "How much do international wires cost?"
    assert [m[1]["index"] for m in msgs if m[0] == "sentence"] == [0, 1]
    assert svc._tts.spoken == [
        "International wires cost 45 dollars.",
        "Domestic wires are not covered here.",
    ]
    assert "raw_answer" not in msgs[-1][1]
    assert {"stt_ms", "tts_ms", "first_audio_ms"} <= set(msgs[-1][1]["timings"])


def test_ws_audio_frames_are_valid_wav(make_client):
    """
    Function name: test_ws_audio_frames_are_valid_wav
    Purpose: Check that each binary frame is a mono 16 kHz WAV file.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "wires cost"})
        msgs = _drain(ws)
    frames = [m[1] for m in msgs if m[0] == "audio"]
    assert len(frames) == 2
    for frame in frames:
        pcm, rate, channels, width = read_wav(frame)
        assert (rate, channels, width, len(pcm)) == (16000, 1, 2, 1600)


def test_ws_text_question_skips_stt(make_client):
    """
    Function name: test_ws_text_question_skips_stt
    Purpose: Check that a text question gives no transcript event and no stt timing, and that
        the question text is trimmed.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "  How much do international wires cost?  "})
        msgs = _drain(ws)
    kinds = [k for k, _ in msgs]
    assert "transcript" not in kinds and kinds[0] == "start" and kinds[-1] == "result"
    assert "stt_ms" not in msgs[-1][1]["timings"]
    assert msgs[-1][1]["found"] is True


def test_ws_empty_transcript_reports_no_speech(make_client):
    """
    Function name: test_ws_empty_transcript_reports_no_speech
    Purpose: Check that an empty transcript gives a transcript event with empty text followed
        by the error "no speech detected" and no audio.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(stt=FakeSTT(text=""))
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_bytes(TONE)
        msgs = _drain(ws)
    assert [k for k, _ in msgs] == ["start", "transcript", "error"]
    assert msgs[1][1]["text"] == ""
    assert msgs[2][1]["detail"] == "no speech detected"
    assert msgs[2][1]["request_id"] == msgs[0][1]["request_id"]


@pytest.mark.parametrize(
    "payload",
    [{"nothing": True}, {"question": ""}, {"question": "   "}],
)
def test_ws_invalid_json_message_reports_error(make_client, payload):
    """
    Function name: test_ws_invalid_json_message_reports_error
    Purpose: Check that a JSON message with a missing, empty or blank question ends with an
        error event that explains the expected message format.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
        payload (dict): The JSON message to send.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json(payload)
        msgs = _drain(ws)
    assert [k for k, _ in msgs] == ["start", "error"]
    assert msgs[1][1]["detail"] == 'send audio bytes or {"question": "..."}'


def test_ws_not_json_text_reports_error(make_client):
    """
    Function name: test_ws_not_json_text_reports_error
    Purpose: Check that a text message that is not JSON ends with an error event (the
        JSON decode error is a ValueError).
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_text("not json at all")
        msgs = _drain(ws)
    assert msgs[-1][0] == "error" and msgs[-1][1]["detail"]


def test_ws_timeout_reports_timed_out(make_client):
    """
    Function name: test_ws_timeout_reports_timed_out
    Purpose: Check that a turn slower than the request timeout ends with the error
        "timed out".
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(llm=SlowLLM(), timeout=0.1)
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "wires cost"})
        msgs = _drain(ws)
        _join_workers()
    assert msgs[-1][0] == "error" and msgs[-1][1]["detail"] == "timed out"


def test_ws_llm_error_reports_error(make_client):
    """
    Function name: test_ws_llm_error_reports_error
    Purpose: Check that an LLMError raised while streaming is sent as an error event.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(llm=DownLLM())
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "wires cost"})
        msgs = _drain(ws)
    assert msgs[-1][0] == "error" and "connection refused" in msgs[-1][1]["detail"]


def test_ws_missing_voice_reports_error(make_client):
    """
    Function name: test_ws_missing_voice_reports_error
    Purpose: Check that a FileNotFoundError from the TTS is sent as an error event.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(tts=MissingVoiceTTS())
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "wires cost"})
        msgs = _drain(ws)
    assert msgs[-1][0] == "error" and msgs[-1][1]["detail"] == "voice missing"


def test_ws_serves_several_turns_with_new_request_ids(make_client):
    """
    Function name: test_ws_serves_several_turns_with_new_request_ids
    Purpose: Check that one socket can carry three turns, each with its own request id, and
        that an error in one turn does not close the socket.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"question": "wires cost"})
        first = _drain(ws)
        ws.send_json({"nothing": True})
        second = _drain(ws)
        ws.send_json({"question": "wires cost"})
        third = _drain(ws)
    assert first[-1][0] == "result" and second[-1][0] == "error" and third[-1][0] == "result"
    ids = {first[0][1]["request_id"], second[0][1]["request_id"], third[0][1]["request_id"]}
    assert len(ids) == 3


def test_ask_audio_empty_upload_is_400(make_client):
    """
    Function name: test_ask_audio_empty_upload_is_400
    Purpose: Check that an empty upload gives 400 with the detail "empty audio upload".
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.post("/ask/audio", files={"file": ("q.wav", b"", "audio/wav")})
    assert r.status_code == 400 and r.json()["detail"] == "empty audio upload"


def test_ask_audio_too_large_is_413(make_client):
    """
    Function name: test_ask_audio_too_large_is_413
    Purpose: Check that an upload one byte over the limit gives 413 and one exactly at the
        limit passes the size check.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(max_upload_mb=1, stt=FakeSTT())
    big = b"\x00" * (1024 * 1024 + 1)
    r = client.post("/ask/audio", files={"file": ("q.wav", big, "audio/wav")})
    assert r.status_code == 413 and r.json()["detail"] == "audio larger than 1 MB"
    exact = b"\x00" * (1024 * 1024)
    ok = client.post("/ask/audio", files={"file": ("q.wav", exact, "audio/wav")})
    assert ok.status_code == 200


def test_ask_audio_no_speech_is_422(make_client):
    """
    Function name: test_ask_audio_no_speech_is_422
    Purpose: Check that an empty transcript gives 422 with a "no speech" detail.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(stt=FakeSTT(text=""))
    r = client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")})
    assert r.status_code == 422 and "no speech" in r.json()["detail"]


def test_ask_audio_success_fields(make_client):
    """
    Function name: test_ask_audio_success_fields
    Purpose: Check the transcript, the base64 WAV and the rounded audio length of a good
        /ask/audio response.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")})
    body = r.json()
    assert r.status_code == 200
    assert body["transcript"] == "How much do international wires cost?"
    wav = base64.b64decode(body["audio_wav_base64"])
    pcm, rate, _, _ = read_wav(wav)
    # two 0.05 s sentences with a 0.15 s gap
    assert rate == 16000 and len(pcm) // 2 == 800 + 2400 + 800
    assert body["audio_seconds"] == 0.25
    assert "raw_answer" not in body


def test_ask_audio_missing_voice_is_409(make_client):
    """
    Function name: test_ask_audio_missing_voice_is_409
    Purpose: Check that a missing voice file gives 409 with the error text.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(tts=MissingVoiceTTS())
    r = client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")})
    assert r.status_code == 409 and r.json()["detail"] == "voice missing"


def test_ask_audio_llm_down_is_503_and_timeout_is_504(make_client):
    """
    Function name: test_ask_audio_llm_down_is_503_and_timeout_is_504
    Purpose: Check that /ask/audio maps an LLM error to 503 and a slow LLM to 504.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(llm=DownLLM())
    r = client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")})
    assert r.status_code == 503 and "LLM unavailable" in r.json()["detail"]
    client, _ = make_client(llm=SlowLLM(), timeout=0.1)
    r = client.post("/ask/audio", files={"file": ("q.wav", TONE, "audio/wav")})
    assert r.status_code == 504


def test_ask_timeout_detail_and_missing_file_is_409(make_client):
    """
    Function name: test_ask_timeout_detail_and_missing_file_is_409
    Purpose: Check the 504 detail text of /ask and that a FileNotFoundError from the
        pipeline (for example a missing index) gives 409.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client(llm=SlowLLM(), timeout=0.1)
    r = client.post("/ask", json={"question": "wires cost"})
    assert r.status_code == 504 and r.json()["detail"] == "timed out after 0 s"

    client, svc = make_client()

    def boom(*a, **k):
        """
        Function name: boom
        Purpose: Raise FileNotFoundError as a missing index would.
        Parameters:
            a (tuple): Positional arguments, ignored.
            k (dict): Keyword arguments, ignored.
        Returns:
            None: never returns; always raises FileNotFoundError.
        """
        raise FileNotFoundError("index missing")

    svc._pipeline.answer = boom
    r = client.post("/ask", json={"question": "wires cost"})
    assert r.status_code == 409 and r.json()["detail"] == "index missing"


def test_ask_passes_mode_and_rewrite(make_client):
    """
    Function name: test_ask_passes_mode_and_rewrite
    Purpose: Check that /ask hands the question, mode and rewrite option to the pipeline.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, svc = make_client()
    seen = {}
    real = svc._pipeline.answer

    def spy(question, mode, timer, rewrite):
        """
        Function name: spy
        Purpose: Record the arguments and call the real pipeline.
        Parameters:
            question (str): The question.
            mode (str): Retrieval mode.
            timer (StageTimer): Stage timer.
            rewrite (str | None): Rewrite option.
        Returns:
            AnswerResult: The real answer.
        """
        seen.update(question=question, mode=mode, rewrite=rewrite)
        return real(question, mode, timer, rewrite)

    svc._pipeline.answer = spy
    r = client.post("/ask", json={"question": "wires cost", "mode": "dense", "rewrite": "always"})
    assert r.status_code == 200
    assert seen == {"question": "wires cost", "mode": "dense", "rewrite": "always"}


def test_health_reports_status_fields(make_client):
    """
    Function name: test_health_reports_status_fields
    Purpose: Check the readiness fields returned by /health for a loaded pipeline.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pipeline_loaded"] is True and body["stt_loaded"] is True
    assert body["tts_loaded"] is True
    assert body["llm_model"] == "fake-model" and body["chunks"] == 4
    assert body["llm_provider"] and body["vector_store"]


def test_request_id_header_on_error_and_new_each_time(make_client):
    """
    Function name: test_request_id_header_on_error_and_new_each_time
    Purpose: Check that the X-Request-ID header is also set on error responses, is a
        12-character id by default and differs between requests.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    a = client.post("/ask", json={"question": ""})
    b = client.get("/health")
    c = client.get("/health")
    assert a.status_code == 422 and a.headers["X-Request-ID"]
    assert len(b.headers["X-Request-ID"]) == 12
    assert b.headers["X-Request-ID"] != c.headers["X-Request-ID"]


def test_client_log_returns_204_and_logs_body(make_client, route_logs):
    """
    Function name: test_client_log_returns_204_and_logs_body
    Purpose: Check that /client-log answers 204 with no body and logs the body text with
        newlines replaced by spaces.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
        route_logs (list[str]): Messages logged by api.routes.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.post("/client-log", content='{"event":\n"mic_blocked"}')
    assert r.status_code == 204 and r.content == b""
    assert 'client_event {"event": "mic_blocked"}' in " ".join(route_logs)


def test_client_log_truncates_to_500_bytes(make_client, route_logs):
    """
    Function name: test_client_log_truncates_to_500_bytes
    Purpose: Check that only the first 500 bytes of a long body are logged.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
        route_logs (list[str]): Messages logged by api.routes.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    r = client.post("/client-log", content="a" * 500 + "Z" * 100)
    assert r.status_code == 204
    assert "a" * 500 in " ".join(route_logs) and "Z" not in " ".join(route_logs)


def test_client_log_handles_invalid_utf8_and_empty_body(make_client, route_logs):
    """
    Function name: test_client_log_handles_invalid_utf8_and_empty_body
    Purpose: Check that invalid UTF-8 bytes are replaced and an empty body still gives 204.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
        route_logs (list[str]): Messages logged by api.routes.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()
    assert client.post("/client-log", content=b"\xff\xfe").status_code == 204
    assert client.post("/client-log").status_code == 204
    assert "�" in " ".join(route_logs)


def test_client_log_broken_body_does_not_fail(make_client, route_logs, monkeypatch):
    """
    Function name: test_client_log_broken_body_does_not_fail
    Purpose: Check that an error while reading the body gives 204 and logs "(unreadable)".
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
        route_logs (list[str]): Messages logged by api.routes.
        monkeypatch (MonkeyPatch): pytest patching helper.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, _ = make_client()

    async def broken(self):
        """
        Function name: broken
        Purpose: Replace Request.body with a version that raises.
        Parameters:
            self (Request): The request.
        Returns:
            bytes: never returns; always raises RuntimeError.
        """
        raise RuntimeError("stream closed")

    monkeypatch.setattr(Request, "body", broken)
    r = client.post("/client-log", content="x")
    assert r.status_code == 204 and "client_event (unreadable)" in " ".join(route_logs)


def test_ingest_returns_manifest(make_client):
    """
    Function name: test_ingest_returns_manifest
    Purpose: Check that POST /ingest calls the services ingest once and returns its manifest
        with the request id.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, svc = make_client()
    calls = []
    svc.ingest = lambda: calls.append(1) or {"chunks": 7, "files": ["a.md"]}
    r = client.post("/ingest")
    assert r.status_code == 200
    assert r.json()["manifest"] == {"chunks": 7, "files": ["a.md"]}
    assert r.json()["request_id"] == r.headers["X-Request-ID"] and calls == [1]


def test_ingest_missing_docs_is_409(make_client):
    """
    Function name: test_ingest_missing_docs_is_409
    Purpose: Check that a FileNotFoundError during ingest gives 409.
    Parameters:
        make_client (Callable): Fixture that builds a test client and fake services.
    Returns:
        None: The test passes or fails through assertions.
    """
    client, svc = make_client()

    def boom():
        """
        Function name: boom
        Purpose: Raise FileNotFoundError as a missing docs folder would.
        Parameters:
            None
        Returns:
            None: never returns; always raises FileNotFoundError.
        """
        raise FileNotFoundError("no docs")

    svc.ingest = boom
    r = client.post("/ingest")
    assert r.status_code == 409 and r.json()["detail"] == "no docs"
