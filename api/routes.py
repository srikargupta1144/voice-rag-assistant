"""
File: api/routes.py

Description:
    HTTP and WebSocket routes: /health, /client-log, /ingest, /ask, /ask/audio, /ws/voice.

    All heavy work runs in the thread pool under asyncio timeouts (api.request_timeout_s);
    every response carries per-stage timings (stt, rewrite, retrieval, llm_first_token, llm,
    tts) and every log line carries the request id.

Flow:
    GET  /health     -> Services.status()
    POST /ingest     -> Services.ingest() -> ingest.build.build_index() + retriever reload
    POST /ask        -> AnswerPipeline.answer(question)                       (text in, JSON out)
    POST /ask/audio  -> stt.transcribe(upload) -> AnswerPipeline.answer(transcript)
                     -> rag.sentences.split_sentences(spoken) -> tts.synthesize() per sentence
                     -> voice.audio.concat_wav() -> JSON with base64 WAV   (record, then send)
    WS   /ws/voice   -> per message (audio bytes or {"question": ...}):
                        stt -> AnswerPipeline.stream_answer() in a worker thread
                        -> each "sentence" event: tts.synthesize() -> JSON + binary WAV frame
                        -> final "result" JSON with citations and timings

Libraries used:
    asyncio - standard library; timeouts, the event loop and the sentence event queue
    base64 - standard library; encodes the answer WAV for the JSON response
    json - standard library; parses text WebSocket messages
    logging - standard library; module logger for transcripts and results
    threading - standard library; worker thread that streams the LLM answer
    typing - standard library; TYPE_CHECKING and Any for type hints
    fastapi - third party; router, request, upload, WebSocket and HTTP error types
    starlette - third party; WebSocketDisconnect exception
    api.middleware - project; new_request_id for WebSocket turns
    api.schemas - project; request and response models
    api.services - project; Services type (type checking only)
    common.logging - project; request_id_var so log lines carry the request id
    common.timing - project; StageTimer for per-stage timings
    rag.llm - project; LLMError raised when the LLM is unavailable
    rag.pipeline - project; AnswerResult type (type checking only)
    rag.sentences - project; split_sentences splits the spoken answer for TTS
    voice.audio - project; concat_wav and wav_duration_s for the reply audio

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  POST /client-log for page events; transcripts logged; WebSocket
                              result keeps passages.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from api.middleware import new_request_id
from api.schemas import AskRequest, AskResponse, AudioAskResponse, IngestResponse
from common.logging import request_id_var
from common.timing import StageTimer
from rag.llm import LLMError
from rag.sentences import split_sentences
from voice.audio import concat_wav, wav_duration_s

if TYPE_CHECKING:
    from api.services import Services
    from rag.pipeline import AnswerResult

log = logging.getLogger(__name__)
router = APIRouter()


def _services(request: Request | WebSocket) -> Services:
    """
    Function name: _services
    Purpose: Return the Services object stored on the application state.
    Parameters:
        request (Request | WebSocket): The current HTTP request or WebSocket connection.
    Returns:
        Services: The shared services of the application.
    """
    return request.app.state.services


async def _in_thread(fn, timeout_s: float):
    """
    Function name: _in_thread
    Purpose: Run a blocking function in the thread pool with a timeout and map failures to
             HTTP errors: timeout to 504, LLMError to 503 and FileNotFoundError to 409.
    Parameters:
        fn (Callable): Function with no arguments to run in the thread pool.
        timeout_s (float): Time limit in seconds.
    Returns:
        Any: The value returned by fn.
    """
    try:
        return await asyncio.wait_for(run_in_threadpool(fn), timeout=timeout_s)
    except TimeoutError as e:
        # The worker thread cannot be killed; it finishes in the background.
        raise HTTPException(504, f"timed out after {timeout_s:.0f} s") from e
    except LLMError as e:
        raise HTTPException(503, f"LLM unavailable: {e}") from e
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e


def _response_fields(rid: str, result: AnswerResult, timer: StageTimer) -> dict[str, Any]:
    """
    Function name: _response_fields
    Purpose: Build the common response fields from an answer result: drop the raw LLM
             answer and add the timings and the request id.
    Parameters:
        rid (str): Request id.
        result (AnswerResult): Answer returned by the pipeline.
        timer (StageTimer): Timer holding the stage timings of this request.
    Returns:
        dict[str, Any]: Fields for AskResponse or the WebSocket "result" message.
    """
    d = result.to_dict()
    d.pop("raw_answer", None)
    d["timings"] = timer.as_dict()
    d["request_id"] = rid
    return d


def _log_done(kind: str, result: AnswerResult, timer: StageTimer) -> None:
    """
    Function name: _log_done
    Purpose: Write one log line for a finished answer with the found flag, the citation
             count and the stage timings.
    Parameters:
        kind (str): Name of the route that produced the answer, e.g. "ask".
        result (AnswerResult): Answer returned by the pipeline.
        timer (StageTimer): Timer holding the stage timings of this request.
    Returns:
        None
    """
    log.info(
        "%s done found=%s citations=%d %s",
        kind,
        result.found,
        len(result.citations),
        timer.log_line(),
    )


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """
    Function name: health
    Purpose: Report liveness plus component readiness. It does not load any models.
    Parameters:
        request (Request): The current HTTP request.
    Returns:
        dict[str, Any]: {"status": "ok"} merged with Services.status().
    """
    svc = _services(request)
    status = await run_in_threadpool(svc.status)
    return {"status": "ok", **status}


@router.post("/client-log", status_code=204)
async def client_log(request: Request) -> None:
    """
    Function name: client_log
    Purpose: Write one event from the web page (button clicks, microphone results) to the
             server log. Used to debug recording problems that never reach the server, such
             as a blocked or silent microphone. The body is a small JSON object; it is
             truncated to 500 bytes before logging, and an unreadable body never fails.
    Parameters:
        request (Request): The current HTTP request carrying the event body.
    Returns:
        None: The route answers with status 204 and no body.
    """
    try:
        body = (await request.body())[:500].decode("utf-8", errors="replace")
    except Exception:  # a broken log call must never fail the page
        body = "(unreadable)"
    log.info("client_event %s", body.replace("\n", " "))


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    """
    Function name: ingest
    Purpose: Re-index data/docs and reload the retriever. The time limit is five times the
             normal request timeout.
    Parameters:
        request (Request): The current HTTP request.
    Returns:
        IngestResponse: The request id and the index build manifest.
    """
    svc = _services(request)
    manifest = await _in_thread(svc.ingest, svc.settings.api.request_timeout_s * 5)
    return IngestResponse(request_id=request.state.request_id, manifest=manifest)


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """
    Function name: ask
    Purpose: Answer a text question with a cited answer, using the requested retrieval mode
             and rewrite option.
    Parameters:
        body (AskRequest): The question, retrieval mode and rewrite option.
        request (Request): The current HTTP request.
    Returns:
        AskResponse: The answer, citations, passages and stage timings.
    """
    svc = _services(request)
    timer = StageTimer()
    result = await _in_thread(
        lambda: svc.pipeline().answer(body.question, body.mode, timer, body.rewrite),
        svc.settings.api.request_timeout_s,
    )
    _log_done("ask", result, timer)
    return AskResponse(**_response_fields(request.state.request_id, result, timer))


@router.post("/ask/audio", response_model=AudioAskResponse)
async def ask_audio(request: Request, file: UploadFile = File(...)) -> AudioAskResponse:
    """
    Function name: ask_audio
    Purpose: Answer a recorded question (WAV, or any format ffmpeg/PyAV decodes) with the
             transcript, the cited answer and the spoken answer as one WAV. Empty uploads
             give 400, uploads over api.max_upload_mb give 413 and silence gives 422.
    Parameters:
        request (Request): The current HTTP request.
        file (UploadFile): The uploaded audio recording.
    Returns:
        AudioAskResponse: The answer fields plus the transcript and base64 WAV audio.
    """
    svc = _services(request)
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio upload")
    if len(audio) > svc.settings.api.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"audio larger than {svc.settings.api.max_upload_mb} MB")
    timer = StageTimer()

    def work() -> tuple[str, AnswerResult, bytes]:
        """
        Function name: work
        Purpose: Blocking part of ask_audio: transcribe the audio, answer the transcript,
                 synthesize each spoken sentence and join the sentence WAVs.
        Parameters:
            None
        Returns:
            tuple[str, AnswerResult, bytes]: The transcript, the answer and the joined WAV.
        """
        with timer.stage("stt"):
            transcript = svc.stt().transcribe(audio).text
        log.info("transcript %r", transcript)  # what Whisper heard, to check not-found answers
        if not transcript:
            raise HTTPException(422, "no speech detected in the recording")
        result = svc.pipeline().answer(transcript, "full", timer)
        wavs = []
        tts = svc.tts()
        for i, sentence in enumerate(split_sentences(result.spoken)):
            with timer.stage("tts"):
                wavs.append(tts.synthesize(sentence))
            if i == 0:
                timer.mark("tts_first_sentence_done")
        return transcript, result, concat_wav(wavs)

    transcript, result, wav = await _in_thread(work, svc.settings.api.request_timeout_s)
    _log_done("ask_audio", result, timer)
    return AudioAskResponse(
        **_response_fields(request.state.request_id, result, timer),
        transcript=transcript,
        audio_wav_base64=base64.b64encode(wav).decode("ascii"),
        audio_seconds=round(wav_duration_s(wav), 2) if wav else 0.0,
    )


@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    """
    Function name: ws_voice
    Purpose: Handle streaming voice turns on one WebSocket; see the module docstring for the
             message protocol. Each message gets its own request id and time limit, and
             timeouts and known errors are sent to the client as "error" messages.
    Parameters:
        ws (WebSocket): The client WebSocket connection.
    Returns:
        None
    """
    await ws.accept()
    svc = _services(ws)
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            rid = new_request_id()
            token = request_id_var.set(rid)
            try:
                async with asyncio.timeout(svc.settings.api.request_timeout_s):
                    await _voice_turn(ws, svc, message, rid)
            except TimeoutError:
                await ws.send_json({"type": "error", "request_id": rid, "detail": "timed out"})
            except (LLMError, FileNotFoundError, ValueError) as e:
                await ws.send_json({"type": "error", "request_id": rid, "detail": str(e)})
            finally:
                request_id_var.reset(token)
    except WebSocketDisconnect:
        return


async def _voice_turn(ws: WebSocket, svc: Services, message: dict, rid: str) -> None:
    """
    Function name: _voice_turn
    Purpose: Handle one question on the socket: transcribe audio bytes or read the JSON
             question, stream each answer sentence as JSON and WAV audio, then send the
             final result.
    Parameters:
        ws (WebSocket): The client WebSocket connection.
        svc (Services): The shared services of the application.
        message (dict): The received WebSocket message (audio bytes or JSON text).
        rid (str): Request id of this turn.
    Returns:
        None
    """
    timer = StageTimer()
    await ws.send_json({"type": "start", "request_id": rid})
    if message.get("bytes"):
        audio = message["bytes"]

        def stt() -> str:
            """
            Function name: stt
            Purpose: Transcribe the received audio bytes, timed as the "stt" stage.
            Parameters:
                None
            Returns:
                str: The transcribed text.
            """
            with timer.stage("stt"):
                return svc.stt().transcribe(audio).text

        question = await run_in_threadpool(stt)
        log.info("transcript %r", question)  # what Whisper heard, to check not-found answers
        await ws.send_json({"type": "transcript", "text": question})
        if not question:
            await ws.send_json({"type": "error", "request_id": rid, "detail": "no speech detected"})
            return
    else:
        question = str(json.loads(message.get("text") or "{}").get("question", "")).strip()
        if not question:
            raise ValueError('send audio bytes or {"question": "..."}')

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    pipeline = svc.pipeline()

    def produce() -> None:
        """
        Function name: produce
        Purpose: Run the streaming answer and put each event on the asyncio queue. An
                 exception is passed on as an "exception" event, and None marks the end.
        Parameters:
            None
        Returns:
            None
        """
        # Runs in a plain thread so LLM generation continues while TTS speaks earlier sentences.
        try:
            for event in pipeline.stream_answer(question, "full", timer):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:  # surfaced to the client as an error event
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "exception", "error": e})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=produce, daemon=True, name=f"voice-{rid}").start()
    tts = svc.tts()
    while (event := await queue.get()) is not None:
        kind = event["type"]
        if kind == "exception":
            raise event["error"]
        if kind == "query":
            await ws.send_json({"type": "query", "query": event["query"]})
        elif kind == "sentence":
            text = event["text"]
            await ws.send_json({"type": "sentence", "index": event["index"], "text": text})

            def speak(t: str = text) -> bytes:
                """
                Function name: speak
                Purpose: Synthesize one sentence to WAV, timed as the "tts" stage.
                Parameters:
                    t (str): The sentence text; defaults to the current sentence.
                Returns:
                    bytes: WAV audio for the sentence.
                """
                with timer.stage("tts"):
                    return tts.synthesize(t)

            wav = await run_in_threadpool(speak)
            await ws.send_bytes(wav)
            timer.mark("first_audio")
        elif kind == "result":
            result: AnswerResult = event["result"]
            payload = _response_fields(rid, result, timer)
            # Passages (with dense, BM25, RRF and re-ranker scores) are kept so the web page
            # can show what retrieval found; five short passages add only a few KB.
            await ws.send_json({"type": "result", **payload})
            _log_done("ws_voice", result, timer)
