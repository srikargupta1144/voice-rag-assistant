"""LLM client interface with two streaming backends: Ollama (local) and OpenAI-compatible (hosted).

Flow:
    rag.pipeline.build_pipeline(settings)
        -> build_llm(settings.llm)                 # provider switch: "ollama" | "openai"
    rag.pipeline.AnswerPipeline / rag.rewrite / eval.answer_eval
        -> client.stream(messages)                 # yields text pieces as they arrive
        -> client.complete(messages)               # joined stream, for short calls

The cloud deployment changes one setting (VRA_LLM__PROVIDER=openai plus a key and model);
OpenAICompatClient works with any server that implements POST /chat/completions with SSE
streaming (OpenAI, Azure OpenAI's v1 endpoint, vLLM, LM Studio, Ollama's /v1).
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
    """The LLM backend failed or is unreachable."""


class LLMClient(Protocol):
    """Chat-style text generation with streaming."""

    name: str
    model: str

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


class _BaseClient:
    """Shared complete() and defaults."""

    name = "base"

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout_s: float):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Connect fast, but allow a slow first token from a cold local model.
        self.timeout = httpx.Timeout(timeout_s, connect=5.0)

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        """Non-streaming convenience: the concatenated stream."""
        return "".join(self.stream(messages, temperature=temperature, max_tokens=max_tokens))


class OllamaClient(_BaseClient):
    """Ollama /api/chat with NDJSON streaming."""

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
        super().__init__(model, temperature, max_tokens, timeout_s)
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        """Yield content pieces until Ollama reports done."""
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
        """True when the server answers and the model is pulled."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            names = {m.get("name") for m in r.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, ValueError):
            return False


class OpenAICompatClient(_BaseClient):
    """OpenAI-compatible /chat/completions with server-sent-event streaming."""

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
        super().__init__(model, temperature, max_tokens, timeout_s)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def stream(self, messages, *, temperature=None, max_tokens=None) -> Iterator[str]:
        """Yield delta content pieces until the [DONE] event."""
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
        """Hosted APIs are assumed reachable when a key is configured."""
        return bool(self.api_key)


def build_llm(cfg: LlmSettings) -> OllamaClient | OpenAICompatClient:
    """Instantiate the configured backend."""
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
