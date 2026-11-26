# Technical reference

Purpose: the detailed reference for the Voice RAG Assistant: architecture diagram, call flow, every folder, both install options, the API, configuration, evaluation method, CI, known limits, testing and code conventions. The short overview is in the [README](../README.md); diagrams are in [ARCHITECTURE.md](ARCHITECTURE.md).

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:

- 2026-11-26, Srikar Gupta: Moved here from the README so the README stays short. Content unchanged apart from relative links.
- 2026-11-26, Srikar Gupta: Testing section added: the three test levels (fast unit suite, functional tests with the real models, end-to-end tests with Ollama), coverage and the CI coverage report.

## Architecture

Diagrams of the whole system, ingestion, one voice question step by step, the not-found decision and the committed parsed documents are in [docs/ARCHITECTURE.md](ARCHITECTURE.md). Windows setup, the run workflow and troubleshooting are in [SETUP_WINDOWS.md](../SETUP_WINDOWS.md).

```text
 Browser page (web/index.html)            FastAPI (api/)                          Local services
┌──────────────────────────────┐   ┌─────────────────────────────────────┐   ┌──────────────────────┐
│ Hold to talk -> WAV 16 kHz   │──►│ POST /ask/audio   (record, then send)│   │ Ollama               │
│ Text box                     │──►│ POST /ask         (text)             │   │  llama3.2:3b         │
│ Plays each sentence's audio  │◄─►│ WS   /ws/voice    (streams sentences)│   │  (or hosted API via  │
└──────────────────────────────┘   │ GET  /health  POST /ingest           │   │   VRA_LLM__PROVIDER) │
                                   │  request ids, timeouts, stage timings│   └──────────▲───────────┘
                                   └──────────────┬──────────────────────┘              │
                                                  ▼                                     │
   speech ─► faster-whisper (small, int8, VAD) ─► query rewrite (LLM, casual speech only)─┤
                                                  ▼                                     │
            hybrid retrieval: FAISS (bge-small) ─┐                                      │
                              BM25 ──────────────┼─► RRF ─► cross-encoder re-rank        │
                                                 │            │                         │
                                                 │   top score < threshold ─► "not found"
                                                 │            ▼                         │
                                                 │   grounded prompt, top 3 passages ───┘
                                                 │            ▼
                                                 │   streamed tokens ─► sentences ─► Piper TTS ─► audio
                                                 │            ▼
                                                 │   [n] markers ─► [source, page] citations
            ingestion: data/docs (PDF, Markdown, text) ─► LlamaIndex MarkdownNodeParser
                       (one chunk per heading, long sections split at ~500 tokens with overlap)
                       ─► bge-small embeddings ─► FAISS (or Pinecone) + chunks.jsonl (BM25 at load)
```

## Call flow (one spoken question)

Every module documents its own flow in its file-level docstring; this is the end-to-end chain:

```text
POST /ask/audio  (or one message on WS /ws/voice)
└─ api.routes.ask_audio()
   ├─ api.middleware.RequestIdMiddleware            # X-Request-ID, request id on every log line
   ├─ voice.stt.WhisperSTT.transcribe(wav)          # stage stt
   ├─ rag.pipeline.AnswerPipeline.stream_answer(transcript)
   │  ├─ rag.rewrite.prepare_query()                # stage rewrite: filler strip, LLM only if casual
   │  ├─ retrieval.retriever.HybridRetriever.retrieve(mode="full")      # stage retrieval
   │  │  ├─ retrieval.embeddings (bge query) -> retrieval.vectorstore.FaissStore.search()
   │  │  ├─ retrieval.bm25.BM25Index.search()
   │  │  ├─ retrieval.fusion.reciprocal_rank_fusion()
   │  │  └─ retrieval.rerank.CrossEncoderReranker.score()
   │  ├─ rag.citations.evidence_is_weak() ──► not-found reply, LLM skipped
   │  ├─ rag.prompts.build_answer_messages(top 3 passages)
   │  ├─ rag.llm.OllamaClient.stream() / OpenAICompatClient.stream()    # llm_first_token mark
   │  │  └─ rag.sentences.SentenceSplitterStream.feed() -> "sentence" events (citations stripped)
   │  └─ rag.citations.is_not_found() / resolve_citations()  -> AnswerResult
   ├─ voice.tts.PiperTTS.synthesize(sentence) per sentence           # stage tts
   └─ JSON: transcript, answer, citations, passages, timings, base64 WAV
```

On `/ws/voice` the pipeline runs in a worker thread and each sentence is synthesized and sent as a binary WAV frame as soon as it leaves the LLM, so playback starts before the answer is complete.

## Project layout

| Path | What it holds |
| --- | --- |
| `configs/` | `settings.py` (typed pydantic-settings tree) and `default.yaml`; override anything with `VRA_<GROUP>__<FIELD>` |
| `common/` | Shared records (`Chunk`, `Hit`), key=value logging with request ids, per-stage timer |
| `ingest/` | `loader.py` (PDF per page, Markdown, text), `chunking.py` (LlamaIndex heading split + sentence splitter), `build.py`, `cli.py` (`vra` command) |
| `retrieval/` | `embeddings.py`, `vectorstore.py` (VectorStore interface, FAISS and Pinecone), `bm25.py`, `fusion.py` (RRF), `rerank.py`, `retriever.py` |
| `rag/` | `llm.py` (Ollama and OpenAI-compatible streaming clients), `prompts.py`, `rewrite.py`, `citations.py` (citations and not-found logic), `sentences.py`, `pipeline.py` |
| `voice/` | `stt.py` (faster-whisper with VAD), `tts.py` (Piper), `audio.py` (WAV helpers) |
| `api/` | `main.py` (app factory), `routes.py`, `services.py` (lazy components), `schemas.py`, `middleware.py` |
| `web/` | `index.html`: record button, text box, streamed playback |
| `eval/` | `questions.jsonl` (35-question golden set), `retrieval_eval.py`, `answer_eval.py`, `metrics.py`, `gate.py`, `report.py`, `baseline.json`, `results/` |
| `scripts/` | `download_models.py`, `make_sample_pdfs.py`, `demo_e2e.py` |
| `data/docs/` | The sample documents that get indexed; `data/pdf_sources/` holds the Markdown the PDFs are rendered from |
| `data/index/` | The parsed result, committed so it can be read without running anything: `chunks.jsonl` (36 chunks with source, page and section), the FAISS index, its id map and `manifest.json` |
| `start.ps1`, `SETUP_WINDOWS.md` | One-command start on Windows and the step-by-step run and troubleshooting guide |
| `tests/unit/` | Unit tests with fake encoder, re-ranker, LLM, STT and TTS; `-m models` adds a real Piper-to-Whisper round trip |
| `deploy/README.md` | Azure Container Apps and AWS ECS deployment steps with the hosted-LLM setting |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | CPU image, api + ollama stack, CI with the Recall@5 gate |

## Setup on Windows

Requirements: Windows 10 or 11, about 6 to 8 GB of free disk (virtual environment about 2.5 GB, models about 0.7 GB, Llama 3.2 3B about 2 GB, Ollama itself), Python 3.11 or 3.12, and Ollama. No GPU is needed.

There are two ways to install. Option A uses the Python already on the machine with pip; Option B uses uv and its lock file.

### Option A: installed Python 3.12 with pip (used on the development laptop)

```powershell
cd voice-rag-assistant

# 1. Virtual environment and packages. torch comes from the CPU index first, so pip does not
#    download about 5 GB of CUDA libraries.
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e . pytest ruff fpdf2

# 2. Ollama and the model
winget install --id Ollama.Ollama -e
ollama pull llama3.2:3b

# 3. Local models into models/ and the search index from data/docs
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
.venv\Scripts\python -m scripts.download_models
.venv\Scripts\vra ingest

# 4. Check
.venv\Scripts\python -m pytest -q
```

pip does not read `uv.lock`, so `pyproject.toml` pins the one package that broke without it: PyAV must stay below 19, because PyAV 19 removed an argument faster-whisper 1.2 passes and every audio request then failed.

### Option B: uv with the lock file

```powershell
# 1. uv (skip if installed). winget installs it as a portable app that may not be on PATH yet;
#    the full path is %LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe
winget install --id astral-sh.uv -e

# 2. Python dependencies (CPU-only torch comes from the PyTorch CPU index)
cd voice-rag-assistant
$env:VIRTUAL_ENV = $null          # avoid a stale outer virtual environment
uv sync                           # add --extra pinecone for the Pinecone backend

# 3. Ollama and the model
winget install --id Ollama.Ollama -e
ollama pull llama3.2:3b           # optional: ollama pull llama3.1:8b

# 4. Local models into models/ (faster-whisper small, Piper en_US-lessac-medium, bge-small, MiniLM re-ranker)
uv run python -m scripts.download_models

# 5. Build the index from data/docs
uv run vra ingest
```

`KMP_DUPLICATE_LIB_OK=TRUE` is set by every entrypoint because torch and CTranslate2 each ship an OpenMP runtime on Windows.

## Running

The simplest way on Windows (Option A install) is the start script. It checks that Ollama is running and has the model, refuses to start if port 8000 is already taken, and then serves the API and the web page:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1                                   # server + web page
powershell -ExecutionPolicy Bypass -File .\start.ps1 -Ask "What is the late fee on the Kova card?"   # one question, no server
```

The same with uv (Option B):

```powershell
uv run vra serve                  # API + web page on http://127.0.0.1:8000 (models warm up first)
uv run vra ask "What is the late fee on the Kova card?"    # one question without the API
uv run python -m scripts.demo_e2e # text, audio and WebSocket checks against the running API
```

### Using the web page

1. Wait for `Application startup complete`, then open <http://127.0.0.1:8000> in a normal Edge or Chrome window. Do not use the browser panel inside VS Code: it kept an old copy of the page and has unreliable microphone access.
2. Ask one warm-up question (type it or press a sample question). The first answer after a restart is slow because the model loads into memory.
3. Tap the round microphone and allow microphone access. Speak; the red level bars move while the microphone hears you.
4. Press **Stop and send**. The page refuses recordings shorter than 0.8 seconds or silent ones, and says why.
5. "Your recording" plays back exactly what was sent. If the bars did not move, choose another device in the **Microphone** picker.
6. "Streamed reply" sends the audio over the WebSocket and speaks each sentence as soon as it is ready; "Single reply" uses `/ask/audio` and plays the full answer at the end. Typed questions and sample questions always go over the WebSocket, so they are spoken in both modes.
7. Open "Retrieved passages and their scores" to see the dense, BM25, RRF and re-ranker score of each passage.

The page reports every click and microphone result to `POST /client-log`, so the server log shows the whole turn (`client_event mic_opened`, `rec_stop seconds=... peakRms=...`, `transcript '...'`, `ws_voice done found=...`). The page is served with `Cache-Control: no-store`, so a browser never shows an outdated copy. Interactive API docs are at <http://127.0.0.1:8000/docs>.

API examples:

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{\"question\": \"How much is an international wire at Kova?\"}'
curl.exe -X POST http://127.0.0.1:8000/ask/audio -F "file=@question.wav"
curl.exe http://127.0.0.1:8000/health
```

`/ask` returns `answer` (with `[source, page]` citations), `spoken` (what is read aloud), `found`, `citations`, the ranked `passages` with their dense, BM25, RRF and re-ranker scores, and `timings`. `/ask/audio` adds `transcript`, `audio_wav_base64` and `audio_seconds`. The WebSocket accepts a binary audio message or `{"question": "..."}` and replies with `start`, `transcript`, `query`, then a `sentence` JSON message followed by a binary WAV frame for every sentence, and finally `result` with citations and timings.

Docker (CPU): `docker compose up --build` starts Ollama, pulls the model once and serves the API on port 8000.

## Configuration

All settings live in [configs/default.yaml](../configs/default.yaml) and can be overridden with environment variables or a `.env` file (template: [.env.example](../.env.example)).

| Setting | Default | Effect |
| --- | --- | --- |
| `VRA_LLM__PROVIDER` | `ollama` | `openai` switches to any OpenAI-compatible hosted API (`VRA_LLM__OPENAI_BASE_URL`, `VRA_LLM__OPENAI_MODEL`, `VRA_LLM__API_KEY` or `OPENAI_API_KEY`) |
| `VRA_LLM__MODEL` | `llama3.2:3b` | `llama3.1:8b` for better answers at a higher latency |
| `PINECONE_API_KEY` | unset | When set (and `VRA_VECTOR_STORE__BACKEND=auto`), vectors go to and come from Pinecone; FAISS otherwise |
| `VRA_RETRIEVAL__NOT_FOUND_THRESHOLD` | `1.5` | Re-ranker logit below which the assistant answers "not found" without calling the LLM |
| `VRA_RETRIEVAL__CONTEXT_K` | `3` | Passages placed in the prompt (retrieval still returns `top_k` = 5) |
| `VRA_REWRITE__MODE` | `auto` | `auto` rewrites only conversational questions; `always` or `never` |
| `VRA_STT__MODEL_SIZE` | `small` | Any faster-whisper size; int8 on CPU |
| `VRA_STT__INITIAL_PROMPT` | `Acme Mutual, Kova Bank, ...` | Domain vocabulary given to Whisper; without it the small model transcribed "Kova" as "COVID" in testing |
| `VRA_TTS__VOICE` | `en_US-lessac-medium` | Any Piper voice downloaded into `models/piper` |
| `VRA_API__REQUEST_TIMEOUT_S` | `120` | Per-request timeout (504 on expiry) |

## Evaluation

The golden set [eval/questions.jsonl](../eval/questions.jsonl) has 35 questions: 30 answerable ones (some phrased like casual speech, for example "uh what's the grace period if I'm late paying my Acme premium?") and 5 that the documents cannot answer. Each answerable question is labelled with its source file, its page for PDFs, and a key phrase of the passage that answers it; a retrieved chunk counts as relevant when all three match, so the labels survive changes to chunk sizes.

- `uv run python -m eval.retrieval_eval` measures Recall@5 and MRR@10 for dense only, BM25 only, hybrid (RRF) and hybrid plus re-ranker, and the accuracy of the not-found gate on the re-ranker score. No LLM is involved, so CI runs it.
- `uv run python -m eval.answer_eval` answers every question with the local LLM and scores the answers with a local LLM judge using the RAGAS metric definitions: faithfulness (share of answer statements the judge finds supported by the passages the model saw) and answer relevancy (mean cosine similarity between the question and three questions the judge generates back from the answer, embedded with bge-small). It also reports how often answerable questions were answered, how often unanswerable ones were declined, and whether the cited passage is the labelled one. `--no-few-shot`, `--temperature` and `--tag` produce comparison runs.
- `uv run python -m scripts.demo_e2e` (with the API running) records per-stage timings for a text question, an audio question created with Piper, and a WebSocket turn, plus the word error rate of the transcript.
- `uv run python -m eval.report` writes the tables below from the files in `eval/results/`.

The RAGAS library itself resolves and installs into this environment (about 40 extra packages, including the LangChain stack), but it is not a project dependency: its metrics issue several structured-output calls per sample that a 3B local judge often fails to parse, and each call costs tens of seconds on this CPU. The judge prompts in `eval/answer_eval.py` implement the same two metric definitions with one short call each.

## Testing

The tests are in three levels. Markers in `pyproject.toml` decide which level runs; the default run excludes the two slow levels, so it stays fast.

| Level | Folder | Marker | What it uses | Command |
| --- | --- | --- | --- | --- |
| Fast unit suite | `tests/unit/` | none | Fakes only: hashed bag-of-words encoder, word-overlap re-ranker, scripted LLM, fake STT and TTS | `.venv\Scripts\python -m pytest` |
| Fast unit suite with coverage | `tests/unit/` | none | As above, plus a line coverage report of the project packages | `.venv\Scripts\python -m pytest --cov` |
| Functional, real models | `tests/functional/` (and `tests/unit/test_voice_models.py`) | `models` | Committed index, bge-small, MiniLM re-ranker, faster-whisper small, Piper; a scripted LLM where an answer is needed | `.venv\Scripts\python -m pytest -m models` |
| End to end, real LLM | `tests/functional/test_llm_e2e.py` | `llm` | Everything real, including llama3.2:3b in a running Ollama | `.venv\Scripts\python -m pytest -m llm` |

Set `KMP_DUPLICATE_LIB_OK=TRUE` and `HF_HUB_OFFLINE=1` before the two slow levels (PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:HF_HUB_OFFLINE="1"`). They need the models in `models/` (`python -m scripts.download_models`) and the index in `data/index`. With uv, replace `.venv\Scripts\python -m` with `uv run`.

What the functional tests (`-m models`) check:

- `test_retrieval_real.py`: seven golden questions from `eval/questions.jsonl` find their labelled source, page and key phrase in the top 5 of the full pipeline (dense, BM25, RRF and re-rank); four unanswerable questions, such as "Does Kova offer car loans?", score below the not-found threshold; each retrieval mode (dense, bm25, hybrid, full) returns hits with its own score.
- `test_ingest_real.py`: a copy of `data/docs` is indexed with the real encoder into pytest's temporary folder; the manifest (36 chunks, 384 dimensions, five sources) and the chunk records match the committed index, and the new index ranks the same passage first. The committed `data/index` is never written.
- `test_voice_real.py`: Piper speaks three golden questions, faster-whisper transcribes them with a word error rate of at most 0.3 (Piper varies its speech slightly on each run), and the transcript still retrieves the right document; Piper output is 16-bit mono WAV; silence gives an empty transcript.
- `test_api_real.py`: the API with the real retriever, STT and TTS and a scripted LLM (a fixed answer with a `[1]` citation). `POST /ask` returns passages with dense, BM25, RRF and re-ranker scores; a not-found question never calls the LLM; `POST /ask/audio` with a Piper recording returns the right transcript and a spoken answer; a WebSocket turn sends one WAV frame after each sentence message.

The end-to-end tests (`-m llm`) are skipped with a clear reason when Ollama is not reachable at `http://localhost:11434` or `llama3.2:3b` is not pulled. Start Ollama first (`ollama serve`, or the Ollama tray app). They check that an answerable question is answered with a citation of the right source through `POST /ask`, that an unanswerable question that passes the re-ranker gate is declined by the model with the not-found sentence, and that an unanswerable question that fails the gate never reaches the model. Each answer takes about 15 to 60 seconds on this laptop.

Heavy models are loaded once per test session (session-scoped fixtures in `tests/functional/conftest.py`), so the functional level takes a few minutes in total, most of it model loading.

Coverage settings are in `pyproject.toml` (`[tool.coverage.run]` and `[tool.coverage.report]`): the measured packages are `api`, `common`, `configs`, `ingest`, `retrieval`, `rag`, `voice` and `eval`; tests and scripts are left out; the report lists missing lines and hides fully covered files. No minimum percentage is enforced yet. `pytest --cov -m models` adds the functional tests to the measurement.

## CI

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs ruff and the fast unit tests with a coverage report (`uv run pytest --cov --cov-report=term-missing`) on every push and pull request, then builds the index from `data/docs`, runs the retrieval evaluation and fails the build when Recall@5 of the full pipeline drops below [eval/baseline.json](../eval/baseline.json). The LLM-based evaluation is not part of CI because the free runners cannot host the model; run it locally before raising the baseline.

## Known limits

- Latency is dominated by the local LLM on this low-power CPU: prompt evaluation runs at roughly 40 tokens per second, so the median first answer token in the evaluation arrives about 14 seconds after the question even with only three passages in the prompt, and a spoken question takes 13 to 18 seconds end to end. The hosted-LLM setting or a machine with a GPU removes most of that wait; speech-to-text takes about 4.5 seconds for a 4-second question, retrieval under a second and text-to-speech about a second.
- Latency numbers are sensitive to other load on the laptop: a first end-to-end run made while another heavy process was using most of the CPU was two to three times slower, so the tables use the runs made on an otherwise idle machine.
- The query rewrite is a second LLM call. It runs only when the question looks conversational, and it adds several seconds when it does.
- The not-found threshold was chosen from the re-ranker scores of this golden set (see the retrieval results). It is a coarse gate: unanswerable questions that share vocabulary with a passage (for example "Can I open a Kova account in euros?") score high and reach the LLM, which then has to decline on its own, and an answerable question whose passage is mostly about something else can fall below it.
- The golden set is small and was written by the same author as the documents, so the retrieval numbers are optimistic compared with real customer questions.
- The LLM judge is the same 3B model that writes the answers; it is lenient, so faithfulness and relevancy are a smoke signal rather than a benchmark.
- The word error rate is measured on speech synthesized by Piper, which is far cleaner than a real microphone recording.
- The Pinecone backend is covered by unit tests with a mocked client only; it has not been run against a live Pinecone index.
- The Docker image and compose file were written and the compose file validated, but no image was built on the development laptop (Docker Desktop was not running), and nothing has been deployed to a cloud.
- Interruptions (barge-in), phone calls and other languages from the plan's optional extras are not built.

## Conventions

- Every source file starts with a docstring containing a `Flow:` section tracing who calls the module and what it calls; functions carry short docstrings.
- Settings go in `configs/settings.py` and `configs/default.yaml`; secrets only through the environment or `.env`.
- Ruff with a line length of 100: `uv run ruff format . ; uv run ruff check .`
- Tests: `uv run pytest` (fast, no models); `uv run pytest -m models` runs the functional tests with the real models; `uv run pytest -m llm` runs the end-to-end tests with Ollama. See [Testing](#testing).

## How to read the results

The measured tables are in the README.


- The re-ranker is what lifts retrieval to Recall@5 1.000 on this set; the one question dense search ranks below fifth ("um so how long will Acme pay for my hotel...") is recovered by BM25 and the cross-encoder.
- The two answerable questions not answered by the few-shot run are q06 ("How can I get coverage for sewer backup?"), whose passage is mostly a list of exclusions and scores below the not-found threshold, and q04, where the model declined after the rewrite turned "hotel" into "temporary accommodation".
- All four statements the judge marked as unsupported in the few-shot run (q05, q08, q09, q25) were checked by hand against the documents and are correct, so faithfulness is understated by the 3B judge; the per-statement verdicts are kept in `eval/results/answers.json`.
- Few-shot is the default: zero-shot answered one more question but declined only 4 of the 5 unanswerable ones (on u03 it replied in its own words instead of the not-found sentence) and gave one answer with an invented condition.
- The text question's first token arrives much sooner than the audio question's because the audio path adds speech-to-text and the rewrite call, and because Ollama reuses the cached prompt prefix (system prompt and few-shot examples) when the previous call used the same prompt.
- Only one temperature (0.1) was evaluated; `--temperature` makes the comparison a single command.
