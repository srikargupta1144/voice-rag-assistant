"""
File: rag/llm.py

Description:
    LLM client interface with two streaming backends: Ollama (local) and OpenAI-compatible
    (hosted).

    The cloud deployment changes one setting (VRA_LLM__PROVIDER=openai plus a key and
    model); OpenAICompatClient works with any server that implements POST /chat/completions
    with SSE streaming (OpenAI, Azure OpenAI's v1 endpoint, vLLM, LM Studio, Ollama's /v1).

Flow:
    rag.pipeline.build_pipeline(settings)
        -> build_llm(settings.llm)                 # provider switch: "ollama" | "openai"
    rag.pipeline.AnswerPipeline / rag.rewrite / eval.answer_eval
        -> client.stream(messages)                 # yields text pieces as they arrive
        -> client.complete(messages)               # joined stream, for short calls

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    json - standard library; parses the streamed NDJSON and SSE lines
    logging - standard library; module logger
    collections.abc - standard library; Iterator type for streamed pieces
    typing - standard library; Protocol for the client interface and TYPE_CHECKING guard
    httpx - third party; HTTP streaming requests to the LLM servers
    configs.settings - project; the LlmSettings type (type checking only)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol

import httpx

if TYPE_CHECKING:
    from configs.settings import LlmSettings

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """
    Class name: LLMError
    Purpose: Raised when the LLM backend failed or is unreachable.
    """


class LLMClient(Protocol):
    """
    Class name: LLMClient
    Purpose: Interface for chat-style text generation with streaming.
    """

    name: str
    model: str

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        Function name: LLMClient.stream
        Purpose: Generate a reply to the chat messages and yield it in pieces as they arrive.
        Parameters:
            messages (list[Message]): The chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature; None uses the client default.
            max_tokens (int | None): Reply token limit; None uses the client default.
        Returns:
            Iterator[str]: yields text pieces of the reply.
        """
        ...

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Function name: LLMClient.complete
        Purpose: Generate a reply to the chat messages and return it in one piece.
        Parameters:
            messages (list[Message]): The chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature; None uses the client default.
            max_tokens (int | None): Reply token limit; None uses the client default.
        Returns:
            str: The full reply text.
        """
        ...


class _BaseClient:
    """
    Class name: _BaseClient
    Purpose: Shared complete() and defaults for the concrete LLM clients.
    """

    name = "base"

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout_s: float):
        """
        Function name: _BaseClient.__init__
        Purpose: Store the model and generation defaults and build the HTTP timeout, which
                 connects fast but allows a slow first token from a cold local model.
        Parameters:
            model (str): The model name on the server.
            temperature (float): The default sampling temperature.
            max_tokens (int): The default reply token limit.
            timeout_s (float): The read timeout in seconds; the connect timeout is 5 seconds.
        Returns:
            None
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Connect fast, but allow a slow first token from a cold local model.
        self.timeout = httpx.Timeout(timeout_s, connect=5.0)

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        """
        Function name: _BaseClient.stream
        Purpose: Placeholder that each concrete client overrides with its streaming call.
        Parameters:
            messages (list[Message]): The chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature; None uses the client default.
            max_tokens (int | None): Reply token limit; None uses the client default.
        Returns:
            Iterator[str]: never returns; always raises NotImplementedError.
        """
        raise NotImplementedError

    def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        """
        Function name: _BaseClient.complete
        Purpose: Non-streaming convenience that returns the concatenated stream.
        Parameters:
            messages (list[Message]): The chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature; None uses the client default.
            max_tokens (int | None): Reply token limit; None uses the client default.
        Returns:
            str: The full reply text.
        """
        return "".join(self.stream(messages, temperature=temperature, max_tokens=max_tokens))


class OllamaClient(_BaseClient):
    """
    Class name: OllamaClient
    Purpose: Client for Ollama /api/chat with NDJSON streaming.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 300,
        timeout_s: float = 90.0,
        keep_alive: str = "30m",
    ):
        """
        Function name: OllamaClient.__init__
        Purpose: Configure the client for one Ollama server and model.
        Parameters:
            base_url (str): The Ollama server URL; a trailing slash is removed.
            model (str): The model name, for example "llama3.2:3b".
            temperature (float): The default sampling temperature.
            max_tokens (int): The default reply token limit.
            timeout_s (float): The read timeout in seconds.
            keep_alive (str): How long Ollama keeps the model loaded after a call.
        Returns:
            None
        """
        super().__init__(model, temperature, max_tokens, timeout_s)
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        """
        Function name: OllamaClient.stream
        Purpose: Call /api/chat with streaming and yield content pieces until Ollama reports
                 done. Raises LLMError on an HTTP error, an error line or a network failure.
        Parameters:
            messages (list[Message]): The chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature; None uses the client default.
            max_tokens (int | None): Reply token limit; None or 0 uses the client default.
        Returns:
            Iterator[str]: yields the non-empty content pieces of the reply.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise LLMError(f"ollama HTTP {resp.status_code}: {resp.text[:200]}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise LLMError(f"ollama: {data['error']}")
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except httpx.HTTPError as e:
            raise LLMError(f"ollama unreachable at {self.base_url}: {e}") from e

    def health(self) -> bool:
        """
        Function name: OllamaClient.health
        Purpose: Check that the Ollama server answers and the configured model is pulled.
        Parameters:
            None
        Returns:
            bool: True when the server answers and the model is listed; False otherwise.
        """
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            names = {m.get("name") for m in r.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, ValueError):
            return False


class OpenAICompatClient(_BaseClient):
    """
    Class name: OpenAICompatClient
    Purpose: Client for OpenAI-compatible /chat/completions with server-sent-event streaming.
    """

    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 300,
        timeout_s: float = 60.0,
    ):
        """
        Function name: OpenAICompatClient.__init__
        Purpose: Configure the client for one OpenAI-compatible endpoint, model and key.
        Parameters:
            base_url (str): The API base URL; a trailing slash is removed.
            model (str): The model name on the server.
            api_key (str): The bearer token sent with each request.
            temperature (float): The default sampling temperature.
            max_tokens (int): The default reply token limit.
            timeout_s (float): The read timeout in seconds.
        Returns:
            None
        """
        super().__init__(model, temperature, max_tokens, timeout_s)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        """
        Function name: OpenAICompatClient.stream
        Purpose: Call /chat/completions with streaming and yield delta content pieces until
                 the [DONE] event. Raises LLMError when no key is set, on an HTTP error or on
                 a network failure.
        Parameters:
            messages (list[Message]): The chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature; None uses the client default.
            max_tokens (int | None): Reply token limit; None or 0 uses the client default.
        Returns:
            Iterator[str]: yields the non-empty delta content pieces of the reply.
        """
        if not self.api_key:
            raise LLMError("openai provider selected but no API key (VRA_LLM__API_KEY)")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise LLMError(f"openai HTTP {resp.status_code}: {resp.text[:200]}")
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or [{}]
                    piece = (choices[0].get("delta") or {}).get("content") or ""
                    if piece:
                        yield piece
        except httpx.HTTPError as e:
            raise LLMError(f"openai-compatible endpoint unreachable: {e}") from e

    def health(self) -> bool:
        """
        Function name: OpenAICompatClient.health
        Purpose: Report health without a network call; hosted APIs are assumed reachable
                 when a key is configured.
        Parameters:
            None
        Returns:
            bool: True when an API key is set.
        """
        return bool(self.api_key)


def build_llm(cfg: LlmSettings) -> OllamaClient | OpenAICompatClient:
    """
    Function name: build_llm
    Purpose: Instantiate the configured backend: the OpenAI-compatible client when the
             provider is "openai", otherwise the Ollama client.
    Parameters:
        cfg (LlmSettings): The LLM section of the settings.
    Returns:
        OllamaClient | OpenAICompatClient: The configured client.
    """
    if cfg.provider == "openai":
        return OpenAICompatClient(
            cfg.openai_base_url,
            cfg.openai_model,
            cfg.effective_key(),
            cfg.temperature,
            cfg.max_tokens,
            cfg.timeout_s,
        )
    return OllamaClient(
        cfg.ollama_url, cfg.model, cfg.temperature, cfg.max_tokens, cfg.timeout_s, cfg.keep_alive
    )
