"""Request and response models for the HTTP API (also drive the OpenAPI docs at /docs).

Flow:
    api.routes handlers -> AskRequest in, AskResponse / AudioAskResponse out
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["dense", "bm25", "hybrid", "full"] = "full"
    rewrite: Literal["auto", "always", "never"] | None = None  # None = configured default


class CitationOut(BaseModel):
    source: str
    page: int | None
    section: str
    chunk_id: str
    label: str


class PassageOut(BaseModel):
    rank: int
    chunk_id: str
    source: str
    page: int | None
    section: str
    label: str
    text: str
    scores: dict[str, float]


class AskResponse(BaseModel):
    request_id: str
    question: str
    query: str
    answer: str
    spoken: str
    found: bool
    citations: list[CitationOut]
    passages: list[PassageOut]
    top_score: float | None
    timings: dict[str, float]


class AudioAskResponse(AskResponse):
    transcript: str
    audio_wav_base64: str  # the spoken answer, 16-bit mono WAV
    audio_seconds: float


class IngestResponse(BaseModel):
    request_id: str
    manifest: dict
