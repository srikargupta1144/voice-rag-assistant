"""
File: scripts/demo_e2e.py

Description:
    End-to-end check against a running API: text question, audio question, WebSocket turn.

    Client-side times include HTTP overhead; server-side per-stage timings come from the
    responses themselves.

Flow:
    uv run vra serve          (in another terminal; warms up models first)
    uv run python -m scripts.demo_e2e [--url http://127.0.0.1:8000]
        -> GET  /health
        -> POST /ask          text question                      -> answer, citations, timings
        -> voice.tts.get_tts().synthesize(spoken question)       # Piper makes the test recording
        -> POST /ask/audio    that WAV                           -> transcript, answer, reply WAV
        -> WS   /ws/voice     same WAV; client-side time to the first audio frame
        -> word_error_rate(spoken question, transcript)
        -> eval/results/e2e.json  and reply audio under data/demo/

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    argparse - standard library; command-line options (--url)
    base64 - standard library; decodes the reply WAV from /ask/audio
    json - standard library; parses WebSocket events and writes the report
    os - standard library; sets KMP_DUPLICATE_LIB_OK before models load
    time - standard library; client-side timings
    httpx - third party; HTTP calls to the running API
    websockets - third party; synchronous WebSocket client for /ws/voice
    configs - project; repository root path and settings
    voice.tts - project; Piper text-to-speech for the test recording

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format;
                              short literal --help description.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time

import httpx

from configs import REPO, get_settings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

TEXT_QUESTION = "What is the late payment fee on the Kova Rewards card?"
SPOKEN_QUESTION = "Um, how much does Kova charge for an outgoing international wire?"


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Function name: word_error_rate
    Purpose: Compute the word error rate: word-level edit distance divided by the reference
             length, with punctuation and case ignored.
    Parameters:
        reference (str): The text that was actually spoken.
        hypothesis (str): The transcript to score.
    Returns:
        float: The word error rate rounded to four decimal places.
    """
    norm = lambda s: "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in s).split()  # noqa: E731
    r, h = norm(reference), norm(hypothesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return round(d[len(h)] / max(len(r), 1), 4)


def ws_turn(url: str, wav: bytes) -> dict:
    """
    Function name: ws_turn
    Purpose: Send one recording over /ws/voice, collect the audio frames and events until
             the result or error event, and time the first audio frame and the whole turn.
    Parameters:
        url (str): The HTTP base URL of the API; it is turned into a WebSocket URL.
        wav (bytes): The question recording as WAV bytes.
    Returns:
        dict: Client timings, sentence count, transcript, answer, citation labels, server
              timings and the raw audio frames.
    """
    from websockets.sync.client import connect

    ws_url = url.replace("http", "ws", 1) + "/ws/voice"
    frames, events = [], []
    with connect(ws_url, max_size=50 * 1024 * 1024, open_timeout=30) as ws:
        t0 = time.perf_counter()
        ws.send(wav)
        first_audio = None
        while True:
            msg = ws.recv(timeout=300)
            if isinstance(msg, bytes):
                frames.append(msg)
                first_audio = first_audio or (time.perf_counter() - t0) * 1000
                continue
            event = json.loads(msg)
            events.append(event)
            if event["type"] in ("result", "error"):
                break
        total = (time.perf_counter() - t0) * 1000
    result = events[-1]
    return {
        "client_first_audio_ms": round(first_audio, 1) if first_audio else None,
        "client_total_ms": round(total, 1),
        "sentences_streamed": len(frames),
        "transcript": next((e["text"] for e in events if e["type"] == "transcript"), ""),
        "answer": result.get("answer"),
        "citations": [c["label"] for c in result.get("citations", [])],
        "server_timings": result.get("timings"),
        "frames": frames,
    }


def main() -> None:
    """
    Function name: main
    Purpose: Run the text, audio and WebSocket checks against the API, print each part and
             write the report to eval/results/e2e.json and the audio to data/demo/.
    Parameters:
        None
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end check against a running API: text question, audio question, WebSocket turn."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    out_dir = REPO / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.url, timeout=300)
    report: dict = {"health": client.get("/health").json()}

    t0 = time.perf_counter()
    r = client.post("/ask", json={"question": TEXT_QUESTION}).json()
    report["text"] = {
        "question": TEXT_QUESTION,
        "answer": r["answer"],
        "citations": [c["label"] for c in r["citations"]],
        "server_timings": r["timings"],
        "client_total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "request_id": r["request_id"],
    }
    print("TEXT ", json.dumps(report["text"], indent=1))

    from voice.tts import get_tts

    question_wav = get_tts(get_settings()).synthesize(SPOKEN_QUESTION)
    (out_dir / "question.wav").write_bytes(question_wav)

    t0 = time.perf_counter()
    r = client.post("/ask/audio", files={"file": ("question.wav", question_wav, "audio/wav")})
    r.raise_for_status()
    body = r.json()
    (out_dir / "answer.wav").write_bytes(base64.b64decode(body["audio_wav_base64"]))
    report["audio"] = {
        "spoken_question": SPOKEN_QUESTION,
        "transcript": body["transcript"],
        "wer": word_error_rate(SPOKEN_QUESTION, body["transcript"]),
        "query": body["query"],
        "answer": body["answer"],
        "citations": [c["label"] for c in body["citations"]],
        "reply_audio_seconds": body["audio_seconds"],
        "server_timings": body["timings"],
        "client_total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "request_id": body["request_id"],
    }
    print("AUDIO", json.dumps(report["audio"], indent=1))

    ws = ws_turn(args.url, question_wav)
    for i, frame in enumerate(ws.pop("frames")):
        (out_dir / f"ws_sentence_{i}.wav").write_bytes(frame)
    report["websocket"] = ws
    print("WS   ", json.dumps(ws, indent=1))

    path = REPO / "eval" / "results" / "e2e.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
