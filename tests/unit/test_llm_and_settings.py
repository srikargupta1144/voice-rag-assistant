"""
File: tests/unit/test_llm_and_settings.py

Description:
    Unit tests for the LLM clients (HTTP mocked), backend selection, settings precedence,
    WAV helpers and the stage timer.

Libraries used:
    json - standard library; builds the fake streamed response lines
    contextlib - standard library; contextmanager for the fake httpx.stream
    pytest - third party; monkeypatch, error checks and approximate comparison
    rag.llm - project; the LLM clients, LLMError and build_llm under test
    common.timing - project; the StageTimer under test
    configs - project; the Settings class under test
    voice.audio - project; WAV helpers under test

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

import json
from contextlib import contextmanager

import pytest

import rag.llm as llm_mod
from common.timing import StageTimer
from configs import Settings
from rag.llm import LLMError, OllamaClient, OpenAICompatClient, build_llm
from voice.audio import concat_wav, pcm_to_wav, wav_duration_s


class _Resp:
    """
    Class name: _Resp
    Purpose: Fake httpx streaming response with a status code, an error body and scripted
        lines.
    """

    def __init__(self, lines, status=200):
        """
        Function name: _Resp.__init__
        Purpose: Store the lines to stream and the HTTP status code.
        Parameters:
            lines (list[str]): Response lines returned by iter_lines().
            status (int): HTTP status code.
        Returns:
            None: Initialises the instance.
        """
        self._lines = lines
        self.status_code = status
        self.text = "error body"

    def iter_lines(self):
        """
        Function name: _Resp.iter_lines
        Purpose: Yield the scripted response lines in order.
        Parameters:
            None
        Returns:
            Iterator[str]: yields each response line.
        """
        yield from self._lines

    def read(self):
        """
        Function name: _Resp.read
        Purpose: Return an empty body, as the clients read the body before reporting an error.
        Parameters:
            None
        Returns:
            bytes: An empty byte string.
        """
        return b""


def _fake_stream(lines, status=200, captured=None):
    """
    Function name: _fake_stream
    Purpose: Build a replacement for httpx.stream that records the request URL and keyword
        arguments and yields a fake response.
    Parameters:
        lines (list[str]): Response lines for the fake response.
        status (int): HTTP status code for the fake response.
        captured (dict | None): Dictionary that receives the URL and request arguments;
            None records nothing.
    Returns:
        Callable: A context manager function with the httpx.stream signature.
    """

    @contextmanager
    def stream(method, url, **kwargs):
        """
        Function name: stream
        Purpose: Record the request when a capture dictionary was given and yield the fake
            response.
        Parameters:
            method (str): HTTP method, ignored.
            url (str): Request URL.
            kwargs (dict): Request keyword arguments such as json and headers.
        Returns:
            Iterator[_Resp]: yields the fake response.
        """
        if captured is not None:
            captured.update(url=url, **kwargs)
        yield _Resp(lines, status)

    return stream


def test_ollama_streams_ndjson(monkeypatch):
    """
    Function name: test_ollama_streams_ndjson
    Purpose: Check that the Ollama client joins streamed NDJSON pieces, calls /api/chat
        without a double slash, and sends num_predict and stream=True.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Replaces httpx.stream with the fake.
    Returns:
        None: The test passes or fails through assertions.
    """
    lines = [json.dumps({"message": {"content": p}, "done": False}) for p in ("Hel", "lo")]
    lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    cap = {}
    monkeypatch.setattr(llm_mod.httpx, "stream", _fake_stream(lines, captured=cap))
    client = OllamaClient("http://ollama:11434/", "llama3.2:3b", max_tokens=50)
    assert client.complete([{"role": "user", "content": "hi"}]) == "Hello"
    assert cap["url"] == "http://ollama:11434/api/chat"
    assert cap["json"]["options"]["num_predict"] == 50 and cap["json"]["stream"] is True


def test_ollama_http_error(monkeypatch):
    """
    Function name: test_ollama_http_error
    Purpose: Check that an HTTP error status from Ollama raises LLMError.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Replaces httpx.stream with the fake.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setattr(llm_mod.httpx, "stream", _fake_stream([], status=404))
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").complete([])


def test_openai_streams_sse(monkeypatch):
    """
    Function name: test_openai_streams_sse
    Purpose: Check that the OpenAI-compatible client joins SSE delta pieces, skips blank
        lines, stops at [DONE], calls /chat/completions and sends the bearer token.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Replaces httpx.stream with the fake.
    Returns:
        None: The test passes or fails through assertions.
    """
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
    """
    Function name: test_openai_requires_key
    Purpose: Check that the OpenAI-compatible client raises LLMError when no API key is set.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    with pytest.raises(LLMError):
        OpenAICompatClient("https://api.example/v1", "m", "").complete([])


def test_one_setting_switches_backend(monkeypatch):
    """
    Function name: test_one_setting_switches_backend
    Purpose: Check that VRA_LLM__PROVIDER=openai selects the OpenAI-compatible client with
        the key from OPENAI_API_KEY, and that removing it returns to the Ollama client.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Sets and removes environment variables.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setenv("VRA_LLM__PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    s = Settings(_env_file=None)
    client = build_llm(s.llm)
    assert isinstance(client, OpenAICompatClient) and client.api_key == "sk-env"
    monkeypatch.delenv("VRA_LLM__PROVIDER")
    assert isinstance(build_llm(Settings(_env_file=None).llm), OllamaClient)


def test_defaults_from_yaml():
    """
    Function name: test_defaults_from_yaml
    Purpose: Check that the default settings carry the expected LLM model, embedding model
        and retrieval top_k.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    s = Settings(_env_file=None)
    assert s.llm.model == "llama3.2:3b"
    assert s.embedding.model == "BAAI/bge-small-en-v1.5"
    assert s.retrieval.top_k == 5


def test_wav_helpers():
    """
    Function name: test_wav_helpers
    Purpose: Check the WAV duration, that joining two clips adds the silence gap, and that
        joining nothing gives empty bytes.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    a = pcm_to_wav(b"\x00\x00" * 1600, 16000)
    assert wav_duration_s(a) == pytest.approx(0.1)
    joined = concat_wav([a, a], silence_ms=100)
    assert wav_duration_s(joined) == pytest.approx(0.3)
    assert concat_wav([]) == b""


def test_stage_timer():
    """
    Function name: test_stage_timer
    Purpose: Check that the stage timer records repeated stages and marks under the
        expected keys with a total, and that its log line includes the stage.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
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
