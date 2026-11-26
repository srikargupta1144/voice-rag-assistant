# Voice RAG Assistant

Ask a question out loud about a set of documents. The assistant finds the answer in the documents, speaks it back, and shows where it came from. If the documents do not have the answer, it says so instead of guessing.

Everything runs on an ordinary laptop with no GPU and no paid services.

## What it does

- **Listens:** you tap the microphone and ask, for example, "How much does Kova charge for an outgoing international wire?"
- **Finds:** it searches the documents and picks the passages that answer the question.
- **Answers:** it replies only from those passages, for example "Kova charges 45 dollars for an outgoing international wire", and names the source document and page.
- **Speaks:** it reads the answer aloud, starting with the first sentence while the rest is still being written.
- **Admits gaps:** for "Does Kova offer car loans?" it says "I could not find that in the documents."

The sample documents describe two made-up companies, Acme Mutual Insurance and Kova Bank, written for this project ([details](docs/DATA_SOURCES.md)).

## How it works

```text
 You speak  ->  1. Speech to text  ->  2. Clean up the question  ->  3. Search the documents
                                                                            |
 You hear   <-  6. Text to speech  <-  5. Write the answer       <-  4. Enough evidence?
                                         (with sources)                 no -> "not found"
```

1. **Speech to text:** faster-whisper turns your voice into text.
2. **Clean up:** filler words such as "um" and "like" are removed so the search gets a clear question.
3. **Search:** two searches run together. One matches meaning (embeddings in a FAISS index), the other matches exact words (BM25). Their results are merged, and a second model re-reads the best 20 and puts the most relevant first.
4. **Evidence check:** if even the best passage is a weak match, the assistant answers "not found" and stops.
5. **Answer:** a local language model (Llama 3.2 3B through Ollama) writes a short answer using only the top three passages and cites each fact.
6. **Text to speech:** Piper speaks the answer, one sentence at a time.

Before any of this, the documents are split into small sections (one per heading), turned into numbers that capture their meaning, and stored in the index. The parsed result is committed in [data/index](data/index), so you can read exactly what the search looks through.

Diagrams of every step: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Results in plain words

Measured on 35 test questions: 30 the documents answer and 5 they do not.

- **Search** puts the right passage in the top five for **all 30** answerable questions. Meaning-based search alone managed 28; adding exact-word search and the re-ranker closed the gap.
- **Honesty:** it declined **all 5** questions the documents cannot answer.
- **Citations** pointed to the correct source every time an answer was given.
- **Speed:** a spoken question takes about 13 to 18 seconds on a laptop CPU, mostly the language model. A hosted model or a GPU makes it much faster.

The full measured tables, written by the evaluation scripts:

<!-- RESULTS:START -->
### Retrieval

Retrieval over 30 answerable questions, 36 chunks (no LLM involved):

| Retrieval | Recall@5 | MRR@10 | Median latency |
| --- | --- | --- | --- |
| Dense only (FAISS, bge-small) | 0.933 | 0.919 | 51 ms |
| BM25 only | 0.967 | 0.904 | <1 ms |
| Hybrid (dense + BM25, RRF) | 0.967 | 0.913 | 48 ms |
| Hybrid + cross-encoder re-rank | 1.000 | 0.978 | 714 ms |

Not-found gate on the re-ranker score (threshold 1.5): 0.914 of all 35 questions classified correctly before any LLM call (lowest answerable score -5.424, highest unanswerable score 5.306).

### Answers

| Variant | Faithfulness | Answer relevancy | Answered (answerable) | Declined (unanswerable) | Citation correct | Median first token | Median total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.2:3b, few-shot, temp 0.1 | 0.857 | 0.857 | 0.933 | 1.000 | 1.000 | 14.0 s | 15.9 s |
| llama3.2:3b, zero-shot, temp 0.1 | 0.828 | 0.866 | 0.967 | 0.800 | 0.966 | 15.4 s | 17.9 s |

Judge: llama3.2:3b through Ollama, over 35 questions. Faithfulness and relevancy are averaged over answered, answerable questions.

### Latency (one run on the development laptop, CPU only)

| Stage | Audio question (/ask/audio) | Text question (/ask) |
| --- | --- | --- |
| Speech-to-text (faster-whisper small, int8) | 4.5 s | - |
| Query rewrite (LLM, only for casual speech) | 4.1 s | <1 ms |
| Retrieval (hybrid + re-rank) | 867 ms | 920 ms |
| LLM first token (since request start) | 15.0 s | 3.9 s |
| LLM call (prompt evaluation + generation) | 7.1 s | 7.7 s |
| Text-to-speech (Piper, all sentences) | 1.1 s | - |
| Total server time | 17.6 s | 8.6 s |

Streaming (/ws/voice, same recording): first reply audio after 13.6 s measured at the client, full reply after 13.6 s; 1 sentence(s) streamed. Word error rate of the transcript on the Piper-generated question: 0.000.
<!-- RESULTS:END -->

## Run it on Windows

You need Windows 10 or 11, Python 3.11 or 3.12, about 8 GB of free disk, and [Ollama](https://ollama.com). No GPU.

**One-time setup:**

```powershell
cd voice-rag-assistant
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e . pytest ruff fpdf2
winget install --id Ollama.Ollama -e
ollama pull llama3.2:3b
.venv\Scripts\python -m scripts.download_models
.venv\Scripts\vra ingest
```

**Every time:**

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

When the server is ready, an Edge window opens by itself. Tap the round microphone, ask your question, then press **Stop and send**. You can also type a question or press one of the sample questions.

Step-by-step guide, how to read the logs, and fixes for common problems: [SETUP_WINDOWS.md](SETUP_WINDOWS.md). Install with uv, the API, settings and the evaluation method: [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md).

## Built with

| Part | Tool | Why this one |
| --- | --- | --- |
| Speech to text | faster-whisper (small) | Accurate and quick on a CPU |
| Splitting documents | LlamaIndex | Splits by heading so each piece is one topic |
| Meaning search | bge-small embeddings + FAISS | Small, accurate, runs locally |
| Exact-word search | BM25 | Catches names, amounts and codes that meaning search blurs |
| Re-ranking | MiniLM cross-encoder | Reads question and passage together to pick the best |
| Answer | Llama 3.2 3B via Ollama | Free and local; one setting switches to a hosted API |
| Text to speech | Piper | Natural voice, fast on a CPU |
| Server and page | FastAPI, WebSocket, plain HTML | Streams the spoken answer sentence by sentence |
| Quality checks | 35-question test set, 74 unit tests, GitHub Actions | Search quality is checked on every push |

## Folders

| Folder | What is inside |
| --- | --- |
| `data/docs` | The sample documents |
| `data/index` | The parsed documents and the search index |
| `ingest` | Reading and splitting documents |
| `retrieval` | Meaning search, exact-word search, merging and re-ranking |
| `rag` | Prompts, the language model, citations and the "not found" check |
| `voice` | Speech to text and text to speech |
| `api`, `web` | The server and the web page |
| `eval` | Test questions, evaluation scripts and results |
| `tests` | Unit tests |
| `docs` | Architecture diagrams, technical reference, data sources |

## Limits

- The test set is small and was written by the same author as the documents, so real-world numbers would be lower.
- The answer checker is the same small model that writes the answers, so its scores are a rough signal.
- One real question is wrongly declined: "How can I get coverage for sewer backup?" The right page is found, but it is mostly a list of exclusions, so it scores as a weak match.
- It is slow on a CPU because of the local language model.
- English only. The Pinecone option is tested only with a mock, and the Docker and cloud setup is written but not yet deployed.

## License

MIT, see [LICENSE](LICENSE).
