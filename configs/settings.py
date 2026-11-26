"""
File: configs/settings.py

Description:
    Application settings: one typed tree for ingestion, retrieval, answering, voice and API.

    Secrets (API keys) are SecretStr and come only from the environment or .env. Two
    well-known variables are honoured without the prefix so deployments can use the usual
    names: PINECONE_API_KEY (switches the vector store to Pinecone when backend is "auto")
    and OPENAI_API_KEY (fallback key for the OpenAI-compatible LLM backend).
    Tests construct `Settings(_env_file=None, ...)` directly with overrides.

Flow:
    get_settings()                                # called once per process by entrypoints
        -> Settings()                             # pydantic-settings BaseSettings
            -> settings_customise_sources()       # fixes the source precedence
                -> init kwargs > env (VRA_*, __ nesting) > .env > configs/default.yaml

Libraries used:
    os - standard library; reads PINECONE_API_KEY and OPENAI_API_KEY fallbacks
    functools - standard library; lru_cache makes get_settings a process-wide singleton
    pathlib - standard library; Path for repository and file locations
    typing - standard library; Literal types for fixed choices
    pydantic - third party; BaseModel for setting groups and SecretStr for API keys
    pydantic_settings - third party; BaseSettings, setting sources and the YAML source

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Class name: PathSettings
    Purpose: Filesystem locations; relative paths resolve against the repository root.
    """

    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    models_dir: Path = Path("models")

    def resolve(self, p: Path) -> Path:
        """
        Function name: PathSettings.resolve
        Purpose: Return the path made absolute against the repository root; absolute paths
                 are returned unchanged.
        Parameters:
            p (Path): A relative or absolute path.
        Returns:
            Path: The absolute path.
        """
        return p if p.is_absolute() else REPO / p


class ChunkSettings(BaseModel):
    """
    Class name: ChunkSettings
    Purpose: Heading-aware chunking, consumed by ingest.chunking.
    """

    chunk_tokens: int = 500
    overlap_tokens: int = 50


class EmbeddingSettings(BaseModel):
    """
    Class name: EmbeddingSettings
    Purpose: Dense encoder, consumed by retrieval.embeddings.
    """

    model: str = "BAAI/bge-small-en-v1.5"
    # bge v1.5 retrieval works best with this instruction on the query side only.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32


class RetrievalSettings(BaseModel):
    """
    Class name: RetrievalSettings
    Purpose: Hybrid retrieval knobs, consumed by retrieval.retriever.HybridRetriever.
    """

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
    """
    Class name: VectorStoreSettings
    Purpose: Vector store selection, consumed by retrieval.vectorstore.build_vector_store.
    """

    backend: Literal["auto", "faiss", "pinecone"] = "auto"  # auto = pinecone iff key set
    pinecone_index: str = "voice-rag-assistant"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_api_key: SecretStr = SecretStr("")

    def effective_key(self) -> str:
        """
        Function name: VectorStoreSettings.effective_key
        Purpose: Return the Pinecone key from VRA_VECTOR_STORE__PINECONE_API_KEY, else from
                 PINECONE_API_KEY, else an empty string.
        Parameters:
            None
        Returns:
            str: The Pinecone API key, or "" when none is set.
        """
        return self.pinecone_api_key.get_secret_value() or os.environ.get("PINECONE_API_KEY", "")

    def effective_backend(self) -> str:
        """
        Function name: VectorStoreSettings.effective_backend
        Purpose: Resolve "auto" to "pinecone" when a key is present, otherwise "faiss"; any
                 other backend value is returned unchanged.
        Parameters:
            None
        Returns:
            str: The vector store backend to use ("faiss" or "pinecone").
        """
        if self.backend == "auto":
            return "pinecone" if self.effective_key() else "faiss"
        return self.backend


class LlmSettings(BaseModel):
    """
    Class name: LlmSettings
    Purpose: LLM backend selection, consumed by rag.llm.build_llm.
    """

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
        """
        Function name: LlmSettings.effective_key
        Purpose: Return the LLM key from VRA_LLM__API_KEY, else from OPENAI_API_KEY, else an
                 empty string.
        Parameters:
            None
        Returns:
            str: The LLM API key, or "" when none is set.
        """
        return self.api_key.get_secret_value() or os.environ.get("OPENAI_API_KEY", "")


class RewriteSettings(BaseModel):
    """
    Class name: RewriteSettings
    Purpose: Spoken-question clean-up, consumed by rag.rewrite.
    """

    # auto = call the LLM only when the question looks conversational (fillers, chit-chat).
    mode: Literal["auto", "always", "never"] = "auto"
    max_tokens: int = 40


class SttSettings(BaseModel):
    """
    Class name: SttSettings
    Purpose: Speech-to-text, consumed by voice.stt.
    """

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
    """
    Class name: TtsSettings
    Purpose: Text-to-speech, consumed by voice.tts.
    """

    voice: str = "en_US-lessac-medium"
    voice_dir: Path = Path("models/piper")
    length_scale: float = 1.0  # >1 speaks slower


class ApiSettings(BaseModel):
    """
    Class name: ApiSettings
    Purpose: HTTP layer, consumed by api.main and api.routes.
    """

    request_timeout_s: float = 120.0
    warm_up: bool = True  # load models at startup so the first question is not a cold start
    max_upload_mb: int = 10
    cors_origins: list[str] = ["*"]


class Settings(BaseSettings):
    """
    Class name: Settings
    Purpose: Root settings; nested groups override via VRA_<GROUP>__<FIELD>.
    """

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
        """
        Function name: Settings.settings_customise_sources
        Purpose: Fix the source precedence: init keyword arguments, then environment
                 variables, then .env, then configs/default.yaml. File secrets are not used.
        Parameters:
            settings_cls (type[BaseSettings]): The settings class being built.
            init_settings (PydanticBaseSettingsSource): Values passed to the constructor.
            env_settings (PydanticBaseSettingsSource): Values from environment variables.
            dotenv_settings (PydanticBaseSettingsSource): Values from the .env file.
            file_secret_settings (PydanticBaseSettingsSource): Secret files source (unused).
        Returns:
            tuple[PydanticBaseSettingsSource, ...]: The sources in priority order.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Function name: get_settings
    Purpose: Return the process-wide settings singleton for entry points (CLI, API, eval).
             The result is cached, so settings are read only once.
    Parameters:
        None
    Returns:
        Settings: The application settings.
    """
    return Settings()
