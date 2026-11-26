"""
File: api/schemas.py

Description:
    Request and response models for the HTTP API. They also drive the OpenAPI docs at /docs.

Flow:
    api.routes handlers -> AskRequest in, AskResponse / AudioAskResponse out

Libraries used:
    typing - standard library; Literal types for the allowed mode values
    pydantic - third party; BaseModel and Field for validated request and response models

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Class name: AskRequest
    Purpose: Body of POST /ask: the question, the retrieval mode and the rewrite option.
    """

    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["dense", "bm25", "hybrid", "full"] = "full"
    rewrite: Literal["auto", "always", "never"] | None = None  # None = configured default


class CitationOut(BaseModel):
    """
    Class name: CitationOut
    Purpose: One citation in an answer: source file, page, section, chunk id and label.
    """

    source: str
    page: int | None
    section: str
    chunk_id: str
    label: str


class PassageOut(BaseModel):
    """
    Class name: PassageOut
    Purpose: One retrieved passage with its rank, provenance, text and per-stage scores.
    """

    rank: int
    chunk_id: str
    source: str
    page: int | None
    section: str
    label: str
    text: str
    scores: dict[str, float]


class AskResponse(BaseModel):
    """
    Class name: AskResponse
    Purpose: Answer to a text question, with citations, passages, top score and timings.
    """

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
    """
    Class name: AudioAskResponse
    Purpose: Answer to a recorded question: the AskResponse fields plus the transcript and
             the spoken answer as base64 WAV with its length in seconds.
    """

    transcript: str
    audio_wav_base64: str  # the spoken answer, 16-bit mono WAV
    audio_seconds: float


class IngestResponse(BaseModel):
    """
    Class name: IngestResponse
    Purpose: Result of POST /ingest: the request id and the index build manifest.
    """

    request_id: str
    manifest: dict
