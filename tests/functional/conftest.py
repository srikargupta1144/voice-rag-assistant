"""
File: tests/functional/conftest.py

Description:
    Shared fixtures for the functional and end-to-end tests that run the real models. Every
    heavy component (encoder, re-ranker, Whisper, Piper) is loaded once per test session so
    the whole functional suite takes a few minutes, not one model load per test.

    The committed index in data/index is only read, never written. Tests that build an index
    write it under pytest's temporary directory.

Flow:
    tests/functional/test_*.py -> fixtures here
        real_settings     -> Settings with the repository paths and the FAISS store
        real_retriever    -> retrieval.retriever.load_retriever over data/index
        real_stt          -> voice.stt.get_stt (faster-whisper small)
        real_tts          -> voice.tts.get_tts (Piper en_US-lessac-medium)
        golden            -> eval/questions.jsonl, keyed by question id
        ollama_settings   -> real_settings, or a skip when Ollama is not reachable

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    json - standard library; reads eval/questions.jsonl
    os - standard library; sets KMP_DUPLICATE_LIB_OK and HF_HUB_OFFLINE
    httpx - third party; checks whether Ollama is reachable
    pytest - third party; fixture definitions and skips
    configs - project; REPO path and the Settings class

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from configs import REPO, Settings

# torch and ctranslate2 each ship an OpenMP runtime on Windows; the models are local.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
INDEX_DIR = REPO / "data" / "index"
DOCS_DIR = REPO / "data" / "docs"


def make_settings(**overrides) -> Settings:
    """
    Function name: make_settings
    Purpose: Build settings that ignore any .env file and any Pinecone key, so the tests
        always use the local FAISS index and the local Ollama.
    Parameters:
        overrides (dict): Extra setting groups, for example paths={...}.
    Returns:
        Settings: The test settings.
    """
    values = {
        "vector_store": {"backend": "faiss"},
        "llm": {"provider": "ollama", "model": OLLAMA_MODEL, "ollama_url": OLLAMA_URL},
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def ollama_skip_reason() -> str | None:
    """
    Function name: ollama_skip_reason
    Purpose: Check that Ollama answers at OLLAMA_URL and has the llama3.2:3b model pulled.
    Parameters:
        None
    Returns:
        str | None: The reason to skip the LLM tests, or None when Ollama is ready.
    """
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"Ollama is not reachable at {OLLAMA_URL} ({type(e).__name__}); start: ollama serve"
    names = {m.get("name") for m in r.json().get("models", [])}
    if OLLAMA_MODEL not in names:
        return (
            f"Ollama is running but {OLLAMA_MODEL} is not pulled; run: ollama pull {OLLAMA_MODEL}"
        )
    return None


@pytest.fixture(scope="session")
def real_settings() -> Settings:
    """
    Function name: real_settings
    Purpose: Fixture with the repository settings: data/index, data/docs, models/ and the
        FAISS vector store.
    Parameters:
        None
    Returns:
        Settings: The settings shared by the whole session.
    """
    return make_settings()


@pytest.fixture(scope="session")
def real_retriever(real_settings):
    """
    Function name: real_retriever
    Purpose: Fixture that loads the committed index with the real bge-small encoder and the
        MiniLM cross-encoder re-ranker, once per session.
    Parameters:
        real_settings (Settings): Session settings fixture.
    Returns:
        HybridRetriever: The retriever over data/index.
    """
    from retrieval.retriever import load_retriever

    return load_retriever(real_settings)


@pytest.fixture(scope="session")
def real_stt(real_settings):
    """
    Function name: real_stt
    Purpose: Fixture that loads faster-whisper small from models/, once per session.
    Parameters:
        real_settings (Settings): Session settings fixture.
    Returns:
        WhisperSTT: The speech-to-text model.
    """
    from voice.stt import get_stt

    return get_stt(real_settings)


@pytest.fixture(scope="session")
def real_tts(real_settings):
    """
    Function name: real_tts
    Purpose: Fixture that loads the Piper voice from models/piper, once per session.
    Parameters:
        real_settings (Settings): Session settings fixture.
    Returns:
        PiperTTS: The text-to-speech model.
    """
    from voice.tts import get_tts

    return get_tts(real_settings)


@pytest.fixture(scope="session")
def golden() -> dict[str, dict]:
    """
    Function name: golden
    Purpose: Fixture with the golden questions of eval/questions.jsonl, keyed by id.
    Parameters:
        None
    Returns:
        dict[str, dict]: Each question record by its id, for example golden["q01"].
    """
    path = REPO / "eval" / "questions.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["id"]: row for row in rows}


@pytest.fixture(scope="session")
def ollama_settings(real_settings) -> Settings:
    """
    Function name: ollama_settings
    Purpose: Fixture that skips the test with a clear reason when Ollama is not reachable
        or the model is missing; otherwise it returns the session settings.
    Parameters:
        real_settings (Settings): Session settings fixture.
    Returns:
        Settings: The session settings, with the Ollama provider.
    """
    reason = ollama_skip_reason()
    if reason:
        pytest.skip(reason)
    return real_settings
