"""LLM clients (HTTP mocked), backend selection, settings precedence, WAV helpers, timer."""

import json
from contextlib import contextmanager

import pytest

import rag.llm as llm_mod
from common.timing import StageTimer
from configs import Settings
from rag.llm import LLMError, OllamaClient, OpenAICompatClient, build_llm
from voice.audio import concat_wav, pcm_to_wav, wav_duration_s


class _Resp:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
        self.text = "error body"

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b""


def _fake_stream(lines, status=200, captured=None):
    @contextmanager
    def stream(method, url, **kwargs):
        if captured is not None:
            captured.update(url=url, **kwargs)
        yield _Resp(lines, status)

    return stream


def test_ollama_streams_ndjson(monkeypatch):
    lines = [json.dumps({"message": {"content": p}, "done": False}) for p in ("Hel", "lo")]
    lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    cap = {}
    monkeypatch.setattr(llm_mod.httpx, "stream", _fake_stream(lines, captured=cap))
    client = OllamaClient("http://ollama:11434/", "llama3.2:3b", max_tokens=50)
    assert client.complete([{"role": "user", "content": "hi"}]) == "Hello"
    assert cap["url"] == "http://ollama:11434/api/chat"
    assert cap["json"]["options"]["num_predict"] == 50 and cap["json"]["stream"] is True


def test_ollama_http_error(monkeypatch):
    monkeypatch.setattr(llm_mod.httpx, "stream", _fake_stream([], status=404))
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").complete([])


def test_openai_streams_sse(monkeypatch):
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "A"}}]}),
        "",
        "data: " + json.dumps({"choices": [{"delta": {"content": "B"}}]}),
        "data: [DONE]",
    ]
    cap = {}
    monkeypatch.setattr(llm_mod.httpx, "stream", _fake_stream(lines, captured=cap))
    client = OpenAICompatClient("https://api.example/v1", "gpt-x", "sk-test")
    assert client.complete([{"role": "user", "content": "q"}]) == "AB"
    assert cap["url"] == "https://api.example/v1/chat/completions"
    assert cap["headers"]["Authorization"] == "Bearer sk-test"


def test_openai_requires_key():
    with pytest.raises(LLMError):
        OpenAICompatClient("https://api.example/v1", "m", "").complete([])


def test_one_setting_switches_backend(monkeypatch):
    monkeypatch.setenv("VRA_LLM__PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    s = Settings(_env_file=None)
    client = build_llm(s.llm)
    assert isinstance(client, OpenAICompatClient) and client.api_key == "sk-env"
    monkeypatch.delenv("VRA_LLM__PROVIDER")
    assert isinstance(build_llm(Settings(_env_file=None).llm), OllamaClient)


def test_defaults_from_yaml():
    s = Settings(_env_file=None)
    assert s.llm.model == "llama3.2:3b"
    assert s.embedding.model == "BAAI/bge-small-en-v1.5"
    assert s.retrieval.top_k == 5


def test_wav_helpers():
    a = pcm_to_wav(b"\x00\x00" * 1600, 16000)
    assert wav_duration_s(a) == pytest.approx(0.1)
    joined = concat_wav([a, a], silence_ms=100)
    assert wav_duration_s(joined) == pytest.approx(0.3)
    assert concat_wav([]) == b""


def test_stage_timer():
    t = StageTimer()
    with t.stage("x"):
        pass
    with t.stage("x"):
        pass
    t.mark("first")
    t.mark("first")  # once
    d = t.as_dict()
    assert {"x_ms", "first_ms", "total_ms"} <= set(d)
    assert "x_ms=" in t.log_line()
