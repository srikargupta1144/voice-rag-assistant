"""
File: tests/unit/test_llm_clients.py

Description:
    Unit tests for the Ollama and OpenAI-compatible LLM clients. The HTTP layer is replaced
    by httpx.MockTransport, so the real httpx streaming code runs but no network is used.
    Covers streaming, complete(), request payloads, defaults and overrides, error handling
    (HTTP status, error lines, timeouts, connection refused), health checks and build_llm.

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    contextlib - standard library; turns the fake stream function into a context manager
    json - standard library; builds and reads request and response bodies
    httpx - third party; MockTransport and the httpx error types
    pytest - third party; fixtures, parametrisation and error checks
    configs - project; the Settings class used to select a provider
    rag.llm - project; the LLM clients, LLMError and build_llm under test

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

import rag.llm as llm_mod
from configs import Settings
from rag.llm import LLMError, OllamaClient, OpenAICompatClient, build_llm

MSGS = [{"role": "user", "content": "hi"}]


class Server:
    """
    Class name: Server
    Purpose: A fake HTTP server behind httpx.MockTransport. It records every request and
        answers with a scripted status and body, or raises a scripted httpx error.
    """

    def __init__(self, status=200, body="", error=None):
        """
        Function name: Server.__init__
        Purpose: Store the scripted response and start an empty request log.
        Parameters:
            status (int): HTTP status of the response.
            body (str): Response body text.
            error (Exception | None): Error raised instead of a response when given.
        Returns:
            None: Initialises the instance.
        """
        self.status = status
        self.body = body
        self.error = error
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """
        Function name: Server.handler
        Purpose: Record the request, then raise the scripted error or return the response.
        Parameters:
            request (httpx.Request): The request sent by the client under test.
        Returns:
            httpx.Response: The scripted response.
        """
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx.Response(self.status, text=self.body)

    @property
    def payload(self) -> dict:
        """
        Function name: Server.payload
        Purpose: Return the JSON body of the last request.
        Parameters:
            None
        Returns:
            dict: The decoded request body.
        """
        return json.loads(self.requests[-1].content)


@pytest.fixture
def serve(monkeypatch):
    """
    Function name: serve
    Purpose: Fixture that returns a factory. The factory creates a Server and routes
        httpx.stream and httpx.get of the rag.llm module through its mock transport.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Replaces the httpx functions used by rag.llm.
    Returns:
        Callable[..., Server]: Factory taking status, body and error.
    """

    def factory(status=200, body="", error=None) -> Server:
        """
        Function name: serve.factory
        Purpose: Create a Server and install it as the transport for streaming and get calls.
        Parameters:
            status (int): HTTP status of the response.
            body (str): Response body text.
            error (Exception | None): Error raised instead of a response when given.
        Returns:
            Server: The fake server.
        """
        server = Server(status, body, error)
        transport = httpx.MockTransport(server.handler)

        @contextlib.contextmanager
        def fake_stream(method, url, **kwargs):
            """
            Function name: fake_stream
            Purpose: Replace httpx.stream by a streaming call on a mock-transport client.
            Parameters:
                method (str): HTTP method.
                url (str): Request URL.
                kwargs (dict): Request keyword arguments such as json, headers and timeout.
            Returns:
                Iterator[httpx.Response]: yields the streamed response.
            """
            with (
                httpx.Client(transport=transport) as client,
                client.stream(method, url, **kwargs) as resp,
            ):
                yield resp

        def fake_get(url, **kwargs):
            """
            Function name: fake_get
            Purpose: Replace httpx.get by a get call on a mock-transport client.
            Parameters:
                url (str): Request URL.
                kwargs (dict): Request keyword arguments such as timeout.
            Returns:
                httpx.Response: The response.
            """
            with httpx.Client(transport=transport) as client:
                return client.get(url, **kwargs)

        monkeypatch.setattr(llm_mod.httpx, "stream", fake_stream)
        monkeypatch.setattr(llm_mod.httpx, "get", fake_get)
        return server

    return factory


def _ndjson(*pieces: str, done: bool = True) -> str:
    """
    Function name: _ndjson
    Purpose: Build an Ollama NDJSON body from content pieces, with an optional final done line.
    Parameters:
        pieces (str): Content pieces, one line each.
        done (bool): Whether to append the line with done=true.
    Returns:
        str: The NDJSON text.
    """
    lines = [json.dumps({"message": {"content": p}, "done": False}) for p in pieces]
    if done:
        lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    return "\n".join(lines) + "\n"


def _sse(*pieces: str, done: bool = True) -> str:
    """
    Function name: _sse
    Purpose: Build a server-sent-event body from delta content pieces, with a final [DONE].
    Parameters:
        pieces (str): Delta content pieces, one event each.
        done (bool): Whether to append the data: [DONE] event.
    Returns:
        str: The event stream text.
    """
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n" for p in pieces]
    if done:
        lines.append("data: [DONE]\n")
    return "\n".join(lines)


def test_ollama_stream_yields_non_empty_pieces_in_order(serve):
    """
    Function name: test_ollama_stream_yields_non_empty_pieces_in_order
    Purpose: Check that the Ollama stream yields the content pieces in order and skips empty
        pieces and blank lines.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    body = _ndjson("A", "", "B", "C").replace("\n", "\n\n")
    serve(body=body)
    client = OllamaClient("http://ollama:11434", "m")
    assert list(client.stream(MSGS)) == ["A", "B", "C"]


def test_ollama_stops_at_done_line(serve):
    """
    Function name: test_ollama_stops_at_done_line
    Purpose: Check that content after the done line is ignored.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    body = _ndjson("A") + json.dumps({"message": {"content": "late"}, "done": False}) + "\n"
    serve(body=body)
    assert OllamaClient("http://o", "m").complete(MSGS) == "A"


def test_ollama_payload_uses_defaults(serve):
    """
    Function name: test_ollama_payload_uses_defaults
    Purpose: Check the request URL and payload when no overrides are passed: model,
        messages, stream flag, keep_alive, temperature and num_predict.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve(body=_ndjson("x"))
    client = OllamaClient("http://o:1/", "llama", temperature=0.3, max_tokens=77, keep_alive="5m")
    client.complete(MSGS)
    assert str(server.requests[0].url) == "http://o:1/api/chat"
    assert server.requests[0].method == "POST"
    assert server.payload == {
        "model": "llama",
        "messages": MSGS,
        "stream": True,
        "keep_alive": "5m",
        "options": {"temperature": 0.3, "num_predict": 77},
    }


def test_ollama_overrides_replace_defaults_and_zero_temperature_is_kept(serve):
    """
    Function name: test_ollama_overrides_replace_defaults_and_zero_temperature_is_kept
    Purpose: Check that explicit temperature and max_tokens override the defaults, and that
        a temperature of 0.0 is kept rather than replaced by the default.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve(body=_ndjson("x"))
    client = OllamaClient("http://o", "m", temperature=0.9, max_tokens=100)
    client.complete(MSGS, temperature=0.0, max_tokens=5)
    assert server.payload["options"] == {"temperature": 0.0, "num_predict": 5}


def test_ollama_zero_max_tokens_uses_default(serve):
    """
    Function name: test_ollama_zero_max_tokens_uses_default
    Purpose: Check that max_tokens=0 falls back to the client default.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve(body=_ndjson("x"))
    OllamaClient("http://o", "m", max_tokens=42).complete(MSGS, max_tokens=0)
    assert server.payload["options"]["num_predict"] == 42


def test_ollama_http_error_raises_llm_error_with_status_and_body(serve):
    """
    Function name: test_ollama_http_error_raises_llm_error_with_status_and_body
    Purpose: Check that a non-200 status raises LLMError containing the status and the start
        of the response body.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(status=500, body="model crashed")
    with pytest.raises(LLMError, match="ollama HTTP 500: model crashed"):
        list(OllamaClient("http://o", "m").stream(MSGS))


def test_ollama_error_body_is_truncated_to_200_characters(serve):
    """
    Function name: test_ollama_error_body_is_truncated_to_200_characters
    Purpose: Check that only the first 200 characters of an error body appear in the message.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(status=503, body="e" * 500)
    with pytest.raises(LLMError) as info:
        list(OllamaClient("http://o", "m").stream(MSGS))
    assert str(info.value) == "ollama HTTP 503: " + "e" * 200


def test_ollama_error_line_raises_llm_error(serve):
    """
    Function name: test_ollama_error_line_raises_llm_error
    Purpose: Check that an NDJSON line with an error key raises LLMError after earlier pieces
        were already yielded.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(body=_ndjson("A", done=False) + json.dumps({"error": "model not found"}) + "\n")
    gen = OllamaClient("http://o", "m").stream(MSGS)
    assert next(gen) == "A"
    with pytest.raises(LLMError, match="ollama: model not found"):
        next(gen)


def test_ollama_malformed_line_raises_llm_error(serve):
    """
    Function name: test_ollama_malformed_line_raises_llm_error
    Purpose: Check that a stream line that is not valid JSON raises LLMError, not a raw
        JSONDecodeError, so callers that catch LLMError handle it.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(body=_ndjson("A", done=False) + "{not json\n")
    gen = OllamaClient("http://o", "m").stream(MSGS)
    assert next(gen) == "A"
    with pytest.raises(LLMError, match="malformed stream line"):
        next(gen)


def test_openai_malformed_data_line_raises_llm_error(serve):
    """
    Function name: test_openai_malformed_data_line_raises_llm_error
    Purpose: Check that a server-sent data line that is not valid JSON raises LLMError.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(body="data: {not json\n\n")
    client = OpenAICompatClient("http://a/v1", "gpt", api_key="k")
    with pytest.raises(LLMError, match="malformed stream line"):
        list(client.stream(MSGS))


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("connect timed out"),
    ],
    ids=["connect-refused", "read-timeout", "connect-timeout"],
)
def test_ollama_network_failures_become_llm_error(serve, error):
    """
    Function name: test_ollama_network_failures_become_llm_error
    Purpose: Check that connection and timeout errors are wrapped in LLMError that names the
        server URL and keeps the original error as the cause.
    Parameters:
        serve (Callable): Fake server factory fixture.
        error (Exception): The httpx error raised by the fake server.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(error=error)
    with pytest.raises(LLMError, match="ollama unreachable at http://o:11434") as info:
        OllamaClient("http://o:11434/", "m").complete(MSGS)
    assert info.value.__cause__ is error


def test_ollama_empty_stream_gives_empty_reply(serve):
    """
    Function name: test_ollama_empty_stream_gives_empty_reply
    Purpose: Check that an empty response body gives an empty stream and an empty reply.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(body="")
    client = OllamaClient("http://o", "m")
    assert list(client.stream(MSGS)) == []
    assert client.complete(MSGS) == ""


def test_ollama_constructor_state():
    """
    Function name: test_ollama_constructor_state
    Purpose: Check the stored settings: trailing slash removal, model, name and the timeout
        with a 5 second connect limit.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    c = OllamaClient("http://o:1///", "m", timeout_s=12.0)
    assert c.base_url == "http://o:1"
    assert c.name == "ollama" and c.model == "m"
    assert c.timeout.read == 12.0 and c.timeout.connect == 5.0


def test_ollama_health_true_for_exact_and_latest_names(serve):
    """
    Function name: test_ollama_health_true_for_exact_and_latest_names
    Purpose: Check that health is True when the model is listed by its exact name or with
        the :latest suffix.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(body=json.dumps({"models": [{"name": "llama3.2:3b"}]}))
    assert OllamaClient("http://o", "llama3.2:3b").health() is True
    serve(body=json.dumps({"models": [{"name": "phi:latest"}]}))
    assert OllamaClient("http://o", "phi").health() is True


def test_ollama_health_false_for_missing_model_bad_json_and_errors(serve):
    """
    Function name: test_ollama_health_false_for_missing_model_bad_json_and_errors
    Purpose: Check that health is False when the model is not pulled, when the reply is not
        JSON and when the server is unreachable.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    client = OllamaClient("http://o", "llama")
    serve(body=json.dumps({"models": [{"name": "other"}]}))
    assert client.health() is False
    serve(body=json.dumps({}))
    assert client.health() is False
    serve(body="not json")
    assert client.health() is False
    serve(error=httpx.ConnectError("refused"))
    assert client.health() is False


def test_openai_stream_yields_delta_pieces(serve):
    """
    Function name: test_openai_stream_yields_delta_pieces
    Purpose: Check that the OpenAI-compatible stream yields delta content pieces in order
        and ignores events without content.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    role_event = f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}}]})}\n"
    null_event = f"data: {json.dumps({'choices': [{'delta': {'content': None}}]})}\n"
    empty_choices = f"data: {json.dumps({'choices': []})}\n"
    serve(body="\n".join([role_event, null_event, empty_choices, _sse("Hel", "lo")]))
    client = OpenAICompatClient("http://api/v1", "gpt", "key")
    assert list(client.stream(MSGS)) == ["Hel", "lo"]


def test_openai_ignores_non_data_lines_and_stops_at_done(serve):
    """
    Function name: test_openai_ignores_non_data_lines_and_stops_at_done
    Purpose: Check that comment lines are skipped and that events after [DONE] are ignored.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    body = ": keep-alive\n\n" + _sse("A") + "\n" + _sse("late", done=False)
    serve(body=body)
    assert OpenAICompatClient("http://api", "m", "k").complete(MSGS) == "A"


def test_openai_payload_url_and_headers(serve):
    """
    Function name: test_openai_payload_url_and_headers
    Purpose: Check the request URL, bearer header and payload: model, messages, stream flag,
        temperature and max_tokens.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve(body=_sse("x"))
    client = OpenAICompatClient(
        "https://api.example.com/v1/", "gpt-x", "sk-secret", temperature=0.2, max_tokens=64
    )
    client.complete(MSGS)
    req = server.requests[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-secret"
    assert server.payload == {
        "model": "gpt-x",
        "messages": MSGS,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 64,
    }


def test_openai_overrides_and_zero_temperature(serve):
    """
    Function name: test_openai_overrides_and_zero_temperature
    Purpose: Check that explicit overrides replace the defaults, that temperature 0.0 is
        kept and that max_tokens=0 falls back to the default.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve(body=_sse("x"))
    client = OpenAICompatClient("http://api", "m", "k", temperature=0.8, max_tokens=90)
    client.complete(MSGS, temperature=0.0, max_tokens=7)
    assert server.payload["temperature"] == 0.0 and server.payload["max_tokens"] == 7
    client.complete(MSGS, max_tokens=0)
    assert server.payload["temperature"] == 0.8 and server.payload["max_tokens"] == 90


def test_openai_missing_key_raises_before_any_request(serve):
    """
    Function name: test_openai_missing_key_raises_before_any_request
    Purpose: Check that an empty API key raises LLMError and that no request is sent.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve(body=_sse("x"))
    with pytest.raises(LLMError, match="no API key"):
        list(OpenAICompatClient("http://api", "m", "").stream(MSGS))
    assert server.requests == []


def test_openai_http_error_raises_llm_error(serve):
    """
    Function name: test_openai_http_error_raises_llm_error
    Purpose: Check that a 401 status raises LLMError with the status and the body start, and
        that a long body is cut to 200 characters.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(status=401, body="bad key")
    with pytest.raises(LLMError, match="openai HTTP 401: bad key"):
        list(OpenAICompatClient("http://api", "m", "k").stream(MSGS))
    serve(status=429, body="r" * 400)
    with pytest.raises(LLMError) as info:
        list(OpenAICompatClient("http://api", "m", "k").stream(MSGS))
    assert str(info.value) == "openai HTTP 429: " + "r" * 200


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
    ids=["connect-refused", "read-timeout"],
)
def test_openai_network_failures_become_llm_error(serve, error):
    """
    Function name: test_openai_network_failures_become_llm_error
    Purpose: Check that connection and timeout errors are wrapped in LLMError with the
        original error as the cause.
    Parameters:
        serve (Callable): Fake server factory fixture.
        error (Exception): The httpx error raised by the fake server.
    Returns:
        None: The test passes or fails through assertions.
    """
    serve(error=error)
    with pytest.raises(LLMError, match="unreachable") as info:
        OpenAICompatClient("http://api", "m", "k").complete(MSGS)
    assert info.value.__cause__ is error


def test_openai_health_depends_only_on_key(serve):
    """
    Function name: test_openai_health_depends_only_on_key
    Purpose: Check that health is True with a key and False without one, with no request.
    Parameters:
        serve (Callable): Fake server factory fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    server = serve()
    assert OpenAICompatClient("http://api", "m", "k").health() is True
    assert OpenAICompatClient("http://api", "m", "").health() is False
    assert server.requests == []


def test_openai_constructor_state():
    """
    Function name: test_openai_constructor_state
    Purpose: Check the stored settings: trailing slash removal, name, key and timeouts.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    c = OpenAICompatClient("http://api/v1/", "m", "k", timeout_s=7.0)
    assert c.base_url == "http://api/v1"
    assert c.name == "openai" and c.api_key == "k"
    assert c.timeout.read == 7.0 and c.timeout.connect == 5.0


def test_base_client_stream_is_abstract():
    """
    Function name: test_base_client_stream_is_abstract
    Purpose: Check that the shared base client does not implement stream().
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    base = llm_mod._BaseClient("m", 0.1, 10, 5.0)
    with pytest.raises(NotImplementedError):
        list(base.stream(MSGS))
    with pytest.raises(NotImplementedError):
        base.complete(MSGS)


def test_build_llm_defaults_to_ollama_with_settings(monkeypatch):
    """
    Function name: test_build_llm_defaults_to_ollama_with_settings
    Purpose: Check that the default provider builds an Ollama client that carries the URL,
        model, generation defaults, timeout and keep-alive from the settings.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Removes provider variables from the environment.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.delenv("VRA_LLM__PROVIDER", raising=False)
    s = Settings(
        _env_file=None,
        llm={
            "ollama_url": "http://box:11434/",
            "model": "tiny",
            "temperature": 0.4,
            "max_tokens": 11,
            "timeout_s": 33.0,
            "keep_alive": "1m",
        },
    )
    c = build_llm(s.llm)
    assert isinstance(c, OllamaClient)
    assert (c.base_url, c.model, c.temperature, c.max_tokens, c.keep_alive) == (
        "http://box:11434",
        "tiny",
        0.4,
        11,
        "1m",
    )
    assert c.timeout.read == 33.0


def test_build_llm_openai_uses_openai_fields_and_env_key(monkeypatch):
    """
    Function name: test_build_llm_openai_uses_openai_fields_and_env_key
    Purpose: Check that provider "openai" builds the OpenAI-compatible client with the
        OpenAI URL and model, taking the key from OPENAI_API_KEY when none is configured.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Sets the key variable in the environment.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.delenv("VRA_LLM__API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    s = Settings(
        _env_file=None,
        llm={
            "provider": "openai",
            "openai_base_url": "http://gw/v1/",
            "openai_model": "gpt-mini",
            "max_tokens": 21,
        },
    )
    c = build_llm(s.llm)
    assert isinstance(c, OpenAICompatClient)
    assert (c.base_url, c.model, c.api_key, c.max_tokens) == (
        "http://gw/v1",
        "gpt-mini",
        "env-key",
        21,
    )


def test_build_llm_openai_without_key_builds_client_that_fails_on_use(monkeypatch):
    """
    Function name: test_build_llm_openai_without_key_builds_client_that_fails_on_use
    Purpose: Check that building the OpenAI-compatible client without any key succeeds, and
        that using it raises LLMError.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Clears the key variables from the environment.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.delenv("VRA_LLM__API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = build_llm(Settings(_env_file=None, llm={"provider": "openai"}).llm)
    assert c.api_key == "" and c.health() is False
    with pytest.raises(LLMError):
        c.complete(MSGS)
