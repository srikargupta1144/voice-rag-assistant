"""Application settings: one typed tree for ingestion, retrieval, answering, voice and API.

Flow:
    get_settings()                                # called once per process by entrypoints
        -> Settings()                             # pydantic-settings BaseSettings
            -> settings_customise_sources()       # fixes the source precedence
                -> init kwargs > env (VRA_*, __ nesting) > .env > configs/default.yaml

Secrets (API keys) are SecretStr and come only from the environment or .env. Two well-known
variables are honoured without the prefix so deployments can use the usual names:
PINECONE_API_KEY (switches the vector store to Pinecone when backend is "auto") and
OPENAI_API_KEY (fallback key for the OpenAI-compatible LLM backend).
Tests construct `Settings(_env_file=None, ...)` directly with overrides.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO = Path(__file__).resolve().parents[1]
_DEFAULT_YAML = Path(__file__).parent / "default.yaml"


class PathSettings(BaseModel):
    """Filesystem locations; relative paths resolve against the repository root."""

    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    models_dir: Path = Path("models")

    def resolve(self, p: Path) -> Path:
        """Return p made absolute against the repository root."""
        return p if p.is_absolute() else REPO / p


class ChunkSettings(BaseModel):
    """Heading-aware chunking, consumed by ingest.chunking."""

    chunk_tokens: int = 500
    overlap_tokens: int = 50


class EmbeddingSettings(BaseModel):
    """Dense encoder, consumed by retrieval.embeddings."""

    model: str = "BAAI/bge-small-en-v1.5"
    # bge v1.5 retrieval works best with this instruction on the query side only.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32


class RetrievalSettings(BaseModel):
    """Hybrid retrieval knobs, consumed by retrieval.retriever.HybridRetriever."""

    candidates: int = 20  # per retriever, before fusion
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20  # fused hits passed to the cross-encoder
    top_k: int = 5  # passages returned by retrieval (Recall@5 is measured on these)
    # Passages placed in the LLM prompt. Prompt evaluation runs at roughly 40 tokens/s on a
    # low-power laptop CPU, so each extra ~130-token passage adds about 3 s to the first token.
    context_k: int = 3
    # Cross-encoder logit below which the top passage counts as weak evidence and the
    # answer short-circuits to the not-found reply (calibrated on eval/questions.jsonl).
    not_found_threshold: float = 1.5


class VectorStoreSettings(BaseModel):
    """Vector store selection, consumed by retrieval.vectorstore.build_vector_store."""

    backend: Literal["auto", "faiss", "pinecone"] = "auto"  # auto = pinecone iff key set
    pinecone_index: str = "voice-rag-assistant"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_api_key: SecretStr = SecretStr("")

    def effective_key(self) -> str:
        """Key from VRA_VECTOR_STORE__PINECONE_API_KEY, else PINECONE_API_KEY, else empty."""
        return self.pinecone_api_key.get_secret_value() or os.environ.get("PINECONE_API_KEY", "")

    def effective_backend(self) -> str:
        """Resolve "auto" to pinecone when a key is present, otherwise faiss."""
        if self.backend == "auto":
            return "pinecone" if self.effective_key() else "faiss"
        return self.backend


class LlmSettings(BaseModel):
    """LLM backend selection, consumed by rag.llm.build_llm."""

    provider: Literal["ollama", "openai"] = "ollama"  # the one switch for cloud deploys
    model: str = "llama3.2:3b"  # Ollama tag; llama3.1:8b is the slower, stronger option
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"  # any OpenAI-compatible endpoint
    openai_model: str = "gpt-4o-mini"
    api_key: SecretStr = SecretStr("")
    temperature: float = 0.1
    max_tokens: int = 300
    timeout_s: float = 90.0
    few_shot: bool = True
    keep_alive: str = "30m"  # keeps the Ollama model loaded between questions

    def effective_key(self) -> str:
        """Key from VRA_LLM__API_KEY, else OPENAI_API_KEY, else empty."""
        return self.api_key.get_secret_value() or os.environ.get("OPENAI_API_KEY", "")


class RewriteSettings(BaseModel):
    """Spoken-question clean-up, consumed by rag.rewrite."""

    # auto = call the LLM only when the question looks conversational (fillers, chit-chat).
    mode: Literal["auto", "always", "never"] = "auto"
    max_tokens: int = 40


class SttSettings(BaseModel):
    """Speech-to-text, consumed by voice.stt."""

    model_size: str = "small"
    model_dir: Path = Path("models/faster-whisper-small")
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1  # greedy decoding: about 2x faster than beam 5 on CPU for short questions
    vad: bool = True
    cpu_threads: int = 0  # 0 = ctranslate2 default
    # Domain vocabulary shown to Whisper as preceding text, so brand names are spelled right
    # (without it the small model hears "Kova" as "COVID").
    initial_prompt: str = "Acme Mutual, Kova Bank, Kova Rewards card, Everyday Checking."


class TtsSettings(BaseModel):
    """Text-to-speech, consumed by voice.tts."""

    voice: str = "en_US-lessac-medium"
    voice_dir: Path = Path("models/piper")
    length_scale: float = 1.0  # >1 speaks slower


class ApiSettings(BaseModel):
    """HTTP layer, consumed by api.main and api.routes."""

    request_timeout_s: float = 120.0
    warm_up: bool = True  # load models at startup so the first question is not a cold start
    max_upload_mb: int = 10
    cors_origins: list[str] = ["*"]


class Settings(BaseSettings):
    """Root settings; nested groups override via VRA_<GROUP>__<FIELD>."""

    model_config = SettingsConfigDict(
        env_prefix="VRA_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        yaml_file=_DEFAULT_YAML,
    )

    log_level: str = "INFO"
    paths: PathSettings = PathSettings()
    chunking: ChunkSettings = ChunkSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    llm: LlmSettings = LlmSettings()
    rewrite: RewriteSettings = RewriteSettings()
    stt: SttSettings = SttSettings()
    tts: TtsSettings = TtsSettings()
    api: ApiSettings = ApiSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton for entrypoints (CLI, API, eval)."""
    return Settings()
