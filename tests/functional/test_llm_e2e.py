"""
File: tests/functional/test_llm_e2e.py

Description:
    End-to-end tests with every real component: the committed index, the real encoder and
    re-ranker, and llama3.2:3b served by a local Ollama. Each test is skipped with a clear
    reason when Ollama is not reachable at http://localhost:11434 or the model is missing.
    Answers use temperature 0 and no query rewrite so they are as repeatable as possible.

    Run with: .venv\\Scripts\\python -m pytest -m llm

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    pytest - third party; the llm marker and fixtures
    fastapi.testclient - third party; in-process HTTP client
    api.main - project; create_app
    api.services - project; Services, given the real pipeline
    rag.pipeline - project; build_pipeline with the configured Ollama client
    rag.prompts - project; the NOT_FOUND reply

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import Services
from rag.pipeline import build_pipeline
from rag.prompts import NOT_FOUND

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def ollama_pipeline(ollama_settings, real_retriever):
    """
    Function name: ollama_pipeline
    Purpose: Fixture with the production answer pipeline: the session retriever and the
        Ollama client built from the settings.
    Parameters:
        ollama_settings (Settings): Session settings; skips when Ollama is not ready.
        real_retriever (HybridRetriever): Retriever over the committed index.
    Returns:
        AnswerPipeline: The pipeline with the real LLM.
    """
    return build_pipeline(ollama_settings, retriever=real_retriever)


def test_ask_answerable_question_cites_the_right_source(ollama_settings, ollama_pipeline):
    """
    Function name: test_ask_answerable_question_cites_the_right_source
    Purpose: Check through POST /ask that the real LLM answers a golden question with the
        fee from the documents and a citation of the Kova account FAQ.
    Parameters:
        ollama_settings (Settings): Session settings with the Ollama provider.
        ollama_pipeline (AnswerPipeline): Pipeline with the real LLM.
    Returns:
        None: The test passes or fails through assertions.
    """
    svc = Services(ollama_settings)
    svc._pipeline = ollama_pipeline
    client = TestClient(create_app(settings=ollama_settings, services=svc))
    question = "How much is an outgoing international wire at Kova?"
    r = client.post("/ask", json={"question": question, "rewrite": "never"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found"] is True, body["answer"]
    assert "kova_bank_account_faq.md" in {c["source"] for c in body["citations"]}
    assert "45" in body["answer"]
    assert body["timings"]["llm_ms"] > 0


def test_unanswerable_question_declined_by_the_model(ollama_pipeline):
    """
    Function name: test_unanswerable_question_declined_by_the_model
    Purpose: Check that a question whose words match real passages (so it passes the
        re-ranker gate and reaches the LLM) but whose answer is not in the documents is
        declined by the model with the not-found sentence.
    Parameters:
        ollama_pipeline (AnswerPipeline): Pipeline with the real LLM.
    Returns:
        None: The test passes or fails through assertions.
    """
    result = ollama_pipeline.answer(
        "Can I open a Kova account in euros?", rewrite_mode="never", temperature=0.0
    )
    assert result.top_score >= ollama_pipeline.not_found_threshold  # the LLM was asked
    assert "llm_ms" in result.timings
    assert result.found is False, result.raw_answer
    assert result.answer == NOT_FOUND
    assert result.citations == []


def test_unanswerable_question_stopped_by_the_gate(ollama_pipeline):
    """
    Function name: test_unanswerable_question_stopped_by_the_gate
    Purpose: Check that a question with no supporting passage gets the not-found sentence
        from the re-ranker gate, before any LLM call, in the production pipeline.
    Parameters:
        ollama_pipeline (AnswerPipeline): Pipeline with the real LLM.
    Returns:
        None: The test passes or fails through assertions.
    """
    result = ollama_pipeline.answer("Does Kova offer car loans?", temperature=0.0)
    assert result.found is False
    assert result.answer == NOT_FOUND
    assert result.spoken == NOT_FOUND
    assert "llm_ms" not in result.timings
