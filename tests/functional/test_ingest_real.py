"""
File: tests/functional/test_ingest_real.py

Description:
    Functional test of the index build with the real bge-small encoder. A copy of data/docs
    is indexed into pytest's temporary directory (the committed data/index is never
    written), and the result is compared with the committed manifest and chunks. The new
    index is then loaded and searched.

    Run with: .venv\\Scripts\\python -m pytest -m models

Libraries used:
    json - standard library; reads the committed manifest and chunk records
    shutil - standard library; copies data/docs into the temporary directory
    pytest - third party; the models marker
    ingest.build - project; build_index, the function under test
    retrieval.retriever - project; load_retriever and read_chunks for the new index
    tests.functional.conftest - project; DOCS_DIR, INDEX_DIR and make_settings

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

import json
import shutil

import pytest

from ingest.build import build_index
from retrieval.retriever import load_retriever, read_chunks
from tests.functional.conftest import DOCS_DIR, INDEX_DIR, make_settings

pytestmark = pytest.mark.models

# Manifest fields that depend on the documents and models, not on the machine.
STABLE_FIELDS = [
    "documents",
    "sources",
    "chunks",
    "max_chunk_tokens",
    "embedding_model",
    "dim",
    "vector_store",
]


@pytest.fixture(scope="module")
def built_index(tmp_path_factory, real_retriever):
    """
    Function name: built_index
    Purpose: Fixture that copies data/docs into a temporary folder and builds a fresh index
        there with the real encoder (reused from the session retriever).
    Parameters:
        tmp_path_factory (TempPathFactory): pytest factory for module temporary folders.
        real_retriever (HybridRetriever): Session retriever; its encoder is reused.
    Returns:
        tuple[Settings, dict]: The settings that point at the new index and its manifest.
    """
    root = tmp_path_factory.mktemp("ingest")
    shutil.copytree(DOCS_DIR, root / "docs")
    settings = make_settings(
        paths={"docs_dir": str(root / "docs"), "index_dir": str(root / "index")}
    )
    manifest = build_index(settings, encoder=real_retriever.encoder)
    return settings, manifest


def test_build_matches_committed_manifest(built_index):
    """
    Function name: test_build_matches_committed_manifest
    Purpose: Check that the new manifest matches the committed one on every stable field
        (36 chunks, 384 dimensions, the same five sources) and that the index files exist.
    Parameters:
        built_index (tuple[Settings, dict]): Settings and manifest of the new index.
    Returns:
        None: The test passes or fails through assertions.
    """
    settings, manifest = built_index
    committed = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunks"] == 36
    assert manifest["dim"] == 384
    assert {k: manifest[k] for k in STABLE_FIELDS} == {k: committed[k] for k in STABLE_FIELDS}
    index_dir = settings.paths.resolve(settings.paths.index_dir)
    for name in ("chunks.jsonl", "manifest.json", "faiss.index", "faiss_ids.json"):
        assert (index_dir / name).is_file(), name
    saved = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_build_produces_same_chunks_as_committed(built_index):
    """
    Function name: test_build_produces_same_chunks_as_committed
    Purpose: Check that the new chunk records have the same ids, text, sources and pages as
        the committed chunks.jsonl, in the same order.
    Parameters:
        built_index (tuple[Settings, dict]): Settings and manifest of the new index.
    Returns:
        None: The test passes or fails through assertions.
    """
    settings, _ = built_index
    new = read_chunks(settings.paths.resolve(settings.paths.index_dir))
    old = read_chunks(INDEX_DIR)
    assert [c.to_dict() for c in new] == [c.to_dict() for c in old]


def test_built_index_answers_a_question(built_index, real_retriever):
    """
    Function name: test_built_index_answers_a_question
    Purpose: Check that the new index loads and that the full pipeline over it ranks the
        same passage first as the committed index for a golden question.
    Parameters:
        built_index (tuple[Settings, dict]): Settings and manifest of the new index.
        real_retriever (HybridRetriever): Retriever over the committed index.
    Returns:
        None: The test passes or fails through assertions.
    """
    settings, _ = built_index
    fresh = load_retriever(
        settings, encoder=real_retriever.encoder, reranker=real_retriever.reranker
    )
    question = "What is the late payment fee on the Kova Rewards card?"
    new_top = fresh.retrieve(question, mode="full", top_k=1)[0]
    old_top = real_retriever.retrieve(question, mode="full", top_k=1)[0]
    assert new_top.chunk.chunk_id == old_top.chunk.chunk_id
    assert new_top.chunk.source == "kova_rewards_card_terms.pdf"
    assert new_top.scores["rerank"] == pytest.approx(old_top.scores["rerank"], abs=1e-3)
