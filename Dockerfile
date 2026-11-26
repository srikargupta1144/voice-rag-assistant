# CPU-only image for the Voice RAG Assistant API.
# Build:  docker build -t voice-rag-assistant .
# The image carries the code, the sample documents, a prebuilt FAISS index and all local
# models (embedder, re-ranker, faster-whisper small, one Piper voice). The LLM is NOT in
# the image: point VRA_LLM__OLLAMA_URL at an Ollama server (see docker-compose.yml) or set
# VRA_LLM__PROVIDER=openai with a key for the hosted backend.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_HUB_DISABLE_TELEMETRY=1

# libgomp: OpenMP runtime used by faiss-cpu and ctranslate2.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so code edits do not invalidate this layer. torch comes from the CPU
# index (pinned in pyproject), so no CUDA libraries end up in the image.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY configs ./configs
COPY common ./common
COPY ingest ./ingest
COPY retrieval ./retrieval
COPY rag ./rag
COPY voice ./voice
COPY api ./api
COPY eval ./eval
COPY scripts ./scripts
COPY web ./web
COPY data/docs ./data/docs
RUN uv sync --frozen --no-dev

# Models and the index are baked in so a container starts answering without downloads.
RUN uv run --no-sync python -m scripts.download_models \
    && uv run --no-sync vra ingest

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
