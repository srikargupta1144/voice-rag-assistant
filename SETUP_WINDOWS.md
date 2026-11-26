# Voice RAG Assistant on this laptop: setup and run workflow

Purpose: how this copy was set up on Windows with the installed Python 3.12 (no uv), how to start and use it, how to read the logs, and what was changed to make it work.

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:

- 2026-11-26, Srikar Gupta: First version. Python pin widened to 3.12; PyAV pinned below 19.
- 2026-11-26, Srikar Gupta: Web page redesigned. Recording is tap to start, then Stop and send (hold-to-talk sent 0.18 s clips on a click, so every question failed with "no speech detected"). Raw audio capture with a level meter, timer, length and silence checks; typed questions are spoken; the WebSocket result includes the retrieved passages.
- 2026-11-26, Srikar Gupta: "Stop and send" and "Cancel" buttons, a player for the user's own recording, and a microphone picker.
- 2026-11-26, Srikar Gupta: Page served with Cache-Control: no-store, because the browser kept an old copy and showed its "EncodingError: Unable to decode audio data" error after the update.
- 2026-11-26, Srikar Gupta: Page events logged through POST /client-log; every transcript logged. Rewritten as a step-by-step workflow.

## 1. Every time: start it

Run in PowerShell from the project folder:

```powershell
cd voice-rag-assistant
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

What the script does, in order:

1. Sets the Windows settings the models need (`KMP_DUPLICATE_LIB_OK`, offline Hugging Face).
2. Starts Ollama if it is not answering, and pulls `llama3.2:3b` if it is missing.
3. Stops with a clear message if port 8000 is already in use (another copy is running).
4. Starts the server. Models load first; wait for `Application startup complete` (about 30 seconds).

Keep this PowerShell window open. Press Ctrl+C in it to stop the server.

## 2. Every time: use the page

1. Open <http://127.0.0.1:8000> in a normal Edge or Chrome window, not the browser panel inside VS Code.
2. Check it is the current page: a round microphone, a "Microphone" picker and "Server connected" at the top right.
3. Warm up: press a sample question such as "What is the late fee on the Kova card?" and wait for the spoken answer. The first answer after a restart takes longer.
4. Voice question: tap the microphone, allow access if asked, speak, watch the red bars move, press **Stop and send**.
5. Check the microphone: play "Your recording". If the bars did not move, pick another device in **Microphone**.
6. Show the evidence: "Sources", "Where the time went", and "Retrieved passages and their scores".

Questions that work well:

| Question | What it shows |
| --- | --- |
| What is the late fee on the Kova card? | Cited answer from a PDF page |
| How much does Kova charge for an outgoing international wire? | Cited answer from a Markdown section |
| Does Kova offer car loans? | "Not in documents" in about 1.5 seconds, the model is not called |
| Um, how much does Kova charge for an outgoing international wire? (spoken) | Transcription, filler removal, spoken answer |

The documents cover only Acme home insurance and Kova bank accounts, cards and privacy. Any other question is answered with "I could not find that in the documents", which is the intended behaviour.

## 3. One question without the page

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1 -Ask "What is the late fee on the Kova card?"
```

Prints the full result as JSON: answer, citations, passages with scores, and timings.

## 4. Reading the server log

Every voice turn appears in the server window in this order:

| Log line | Meaning |
| --- | --- |
| `client_event {"event":"page_loaded", ...}` | The browser loaded the current page |
| `client_event {"event":"mic_opened","device":"...","muted":false}` | Microphone access granted; shows which device |
| `client_event {"event":"rec_stop","seconds":5.0,"peakRms":0.12}` | Recording length and loudness; peakRms near 0 means a silent microphone |
| `Processing audio with duration ...` | Whisper received the audio |
| `transcript '...'` | What Whisper heard |
| `ws_voice done found=True citations=1 ... total_ms=...` | Answer finished (streamed mode); `ask_audio done` in single mode |
| `client_event {"event":"turn_failed", ...}` | Something failed; the detail says what |

Harmless lines at startup: "Could not load library with AVX2 support" (FAISS uses its standard build), "cache_dir argument is deprecated", the symlink notice, and `favicon.ico 404`.

## 5. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[Errno 10048] ... ('127.0.0.1', 8000)` or "Port 8000 is already in use" | Another copy of the server is running | Close the other window, or `taskkill /PID <number> /T /F` |
| The page shows "Hold to talk" | An old copy of the page | Close the tab, open a new Edge window, press Ctrl+F5 |
| "Microphone access is blocked" | Permission denied once | Lock icon next to the address, allow Microphone, reload |
| "The microphone picked up only silence" | Wrong input device or muted | Pick another device in Microphone; check Windows Settings > Privacy > Microphone |
| "That recording was only 0.x seconds" | Stop pressed too early | Speak the whole question, then press Stop and send |
| Answer takes 30 seconds or more | Model cold, or the CPU is busy | Ask a warm-up question; close heavy apps |
| "Not in documents" for a question the documents cover | Whisper misheard | Check the `transcript '...'` line in the log |

## 6. One-time setup (already done on this laptop)

1. `python -m venv .venv` with the installed Python 3.12.
2. `.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU-only torch).
3. `.venv\Scripts\python -m pip install -e . pytest ruff fpdf2` (PyAV is pinned below 19 in `pyproject.toml`).
4. `winget install --id Ollama.Ollama -e`, then `ollama pull llama3.2:3b`.
5. `.venv\Scripts\python -m scripts.download_models`, then `.venv\Scripts\vra ingest`.
6. `.venv\Scripts\python -m pytest -q` (74 tests pass).

## 7. What was changed for this laptop

| File | Change |
| --- | --- |
| `pyproject.toml`, `.python-version` | Python 3.12 allowed; PyAV pinned below 19 |
| `start.ps1` | New: one-command start, Ollama check, port check |
| `web/index.html` | Redesigned page: tap to record, Stop and send, Cancel, level meter, timer, own-recording player, microphone picker, progress steps, sources, timings, passages, replay, click logging |
| `api/main.py` | Page served with `Cache-Control: no-store` |
| `api/routes.py` | `POST /client-log` for page events; transcripts logged; WebSocket result keeps the passages |
| `tests/unit/test_api.py` | Page test checks the new page text |
| `README.md` | Option A install with pip, start script, how to use the page |
| `eval/results/e2e_this_laptop_py312.json` | End-to-end timings measured on this laptop; the README tables keep the original run |

## 8. Checks run on 2026-11-26

| Check | Result |
| --- | --- |
| Unit tests | 74 passed |
| Ruff lint | Passed |
| Retrieval evaluation | Same as the README: Recall@5 0.933 dense, 1.000 hybrid plus re-ranker; not-found gate 0.914 |
| Voice question in Edge with a simulated microphone | 6.3 s recorded, exact transcript, spoken cited answer |
| Voice question in Edge with the laptop microphone, streamed | 5.0 s recorded, found with a citation, first audio after 12.8 s |
| Voice question in Edge with the laptop microphone, single reply | 6.1 s recorded, answered in 4.1 s |
| Sample question, streamed | Found with a citation, first audio after 4.3 s |
