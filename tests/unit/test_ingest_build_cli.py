"""
File: tests/unit/test_ingest_build_cli.py

Description:
    Unit tests for the ingestion build step (ingest.build.build_index) and the Typer
    command-line app (ingest.cli). The build runs on a temporary documents folder with the
    fake encoder, so no model is downloaded and the real index folder is never touched. The
    command-line tests replace build_index, build_pipeline and uvicorn with fakes.

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    json - standard library; reads the files written by the build and the command output
    pytest - third party; fixtures and error checks
    typer.testing - third party; CliRunner that invokes the Typer app in process
    ingest.build - project; build_index under test
    ingest.cli - project; the Typer app under test
    retrieval.retriever - project; file names and read_chunks for checking the written index
    retrieval.vectorstore - project; FaissStore used to load the written index

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import ingest.build as build_mod
import ingest.cli as cli_mod
from ingest.build import build_index
from retrieval.retriever import CHUNKS_FILE, MANIFEST_FILE, read_chunks
from retrieval.vectorstore import FaissStore

FAQ = "# Fees\n\nEveryday Checking has a monthly fee of 8 dollars.\n\n# Wires\n\nWires cost 45.\n"
NOTES = "Plain notes about deductibles. The standard deductible is 1,000 dollars.\n"


@pytest.fixture
def docs(settings):
    """
    Function name: docs
    Purpose: Fixture that creates the documents folder from the settings with one Markdown
        file and one text file.
    Parameters:
        settings (Settings): Test settings fixture.
    Returns:
        Path: The documents folder.
    """
    d = settings.paths.resolve(settings.paths.docs_dir)
    d.mkdir(parents=True)
    (d / "faq.md").write_text(FAQ, encoding="utf-8")
    (d / "notes.txt").write_text(NOTES, encoding="utf-8")
    return d


def test_build_index_manifest_fields(settings, docs, encoder):
    """
    Function name: test_build_index_manifest_fields
    Purpose: Check the returned manifest: document count, sorted sources, chunk count,
        embedding model, dimension, vector store name and the largest chunk size.
    Parameters:
        settings (Settings): Test settings fixture.
        docs (Path): Documents folder fixture.
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    m = build_index(settings, encoder)
    assert m["documents"] == 2
    assert m["sources"] == ["faq.md", "notes.txt"]
    assert m["chunks"] >= 2
    assert m["embedding_model"] == settings.embedding.model
    assert m["dim"] == encoder.dim == 64
    assert m["vector_store"] == "faiss"
    assert 0 < m["max_chunk_tokens"] <= settings.chunking.chunk_tokens
    assert isinstance(m["seconds"], float) and m["seconds"] >= 0
    assert set(m) == {
        "documents",
        "sources",
        "chunks",
        "max_chunk_tokens",
        "embedding_model",
        "dim",
        "vector_store",
        "seconds",
    }


def test_build_index_writes_chunks_manifest_and_faiss_files(settings, docs, encoder):
    """
    Function name: test_build_index_writes_chunks_manifest_and_faiss_files
    Purpose: Check that the index folder holds chunks.jsonl with one line per chunk, a
        manifest equal to the returned one, and the FAISS index and id files, and that the
        index can be loaded again with the right number of vectors.
    Parameters:
        settings (Settings): Test settings fixture.
        docs (Path): Documents folder fixture.
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    manifest = build_index(settings, encoder)
    index_dir = settings.paths.resolve(settings.paths.index_dir)
    lines = (index_dir / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest["chunks"]
    records = [json.loads(ln) for ln in lines]
    assert {r["source"] for r in records} == {"faq.md", "notes.txt"}
    assert all(r["text"].strip() for r in records)
    assert json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8")) == manifest
    assert (index_dir / FaissStore.INDEX_FILE).is_file()
    ids = json.loads((index_dir / FaissStore.IDS_FILE).read_text(encoding="utf-8"))
    assert ids == [r["chunk_id"] for r in records]
    assert [c.chunk_id for c in read_chunks(index_dir)] == ids
    store = FaissStore(encoder.dim)
    store.load(index_dir)
    assert store._index.ntotal == manifest["chunks"]


def test_build_index_creates_missing_index_folder(settings, docs, encoder):
    """
    Function name: test_build_index_creates_missing_index_folder
    Purpose: Check that the index folder is created when it does not exist, and that a second
        build over the same folder overwrites the files and gives the same counts.
    Parameters:
        settings (Settings): Test settings fixture.
        docs (Path): Documents folder fixture.
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    index_dir = settings.paths.resolve(settings.paths.index_dir)
    assert not index_dir.exists()
    first = build_index(settings, encoder)
    second = build_index(settings, encoder)
    assert index_dir.is_dir()
    assert first["chunks"] == second["chunks"]
    assert len(read_chunks(index_dir)) == second["chunks"]


def test_build_index_without_documents_raises(settings, encoder):
    """
    Function name: test_build_index_without_documents_raises
    Purpose: Check that an empty documents folder raises FileNotFoundError naming the folder
        and writes no index.
    Parameters:
        settings (Settings): Test settings fixture.
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    docs_dir = settings.paths.resolve(settings.paths.docs_dir)
    docs_dir.mkdir(parents=True)
    (docs_dir / "image.png").write_bytes(b"not a document")
    with pytest.raises(FileNotFoundError, match="no .pdf/.md/.txt files"):
        build_index(settings, encoder)
    assert not settings.paths.resolve(settings.paths.index_dir).exists()


def test_build_index_loads_configured_encoder_when_none_given(settings, docs, encoder, monkeypatch):
    """
    Function name: test_build_index_loads_configured_encoder_when_none_given
    Purpose: Check that build_index asks get_encoder for the encoder when none is passed, and
        that it does not call get_encoder when an encoder is passed.
    Parameters:
        settings (Settings): Test settings fixture.
        docs (Path): Documents folder fixture.
        encoder (FakeEncoder): Fake encoder fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces get_encoder in the embeddings module.
    Returns:
        None: The test passes or fails through assertions.
    """
    import retrieval.embeddings as emb

    seen = []

    def fake_get_encoder(s):
        """
        Function name: fake_get_encoder
        Purpose: Record the settings object and return the fake encoder.
        Parameters:
            s (Settings): Settings passed by build_index.
        Returns:
            FakeEncoder: The fake encoder.
        """
        seen.append(s)
        return encoder

    monkeypatch.setattr(emb, "get_encoder", fake_get_encoder)
    build_index(settings)
    assert seen == [settings]
    build_index(settings, encoder)
    assert len(seen) == 1


def test_build_index_respects_chunk_size_setting(settings, docs, encoder):
    """
    Function name: test_build_index_respects_chunk_size_setting
    Purpose: Check that smaller chunk settings give more chunks, each within the limit.
    Parameters:
        settings (Settings): Test settings fixture.
        docs (Path): Documents folder fixture.
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    (docs / "long.txt").write_text(("Alpha beta gamma delta. " * 12), encoding="utf-8")
    big = build_index(settings, encoder)
    small_settings = settings.model_copy(
        update={
            "chunking": settings.chunking.model_copy(
                update={"chunk_tokens": 30, "overlap_tokens": 3}
            )
        }
    )
    small = build_index(small_settings, encoder)
    assert small["chunks"] > big["chunks"]
    assert small["max_chunk_tokens"] <= 30


class _Result:
    """
    Class name: _Result
    Purpose: Fake answer result whose to_dict returns a fixed dictionary.
    """

    def to_dict(self) -> dict:
        """
        Function name: _Result.to_dict
        Purpose: Return the fixed answer dictionary.
        Parameters:
            None
        Returns:
            dict: A small answer document.
        """
        return {"answer": "42", "found": True}


class _Pipeline:
    """
    Class name: _Pipeline
    Purpose: Fake answer pipeline that records the question and mode it was asked.
    """

    def __init__(self):
        """
        Function name: _Pipeline.__init__
        Purpose: Start with an empty call log.
        Parameters:
            None
        Returns:
            None: Initialises the instance.
        """
        self.calls: list[tuple[str, str]] = []

    def answer(self, question: str, mode: str = "full") -> _Result:
        """
        Function name: _Pipeline.answer
        Purpose: Record the call and return the fixed result.
        Parameters:
            question (str): The question text.
            mode (str): The retrieval mode.
        Returns:
            _Result: The fake result.
        """
        self.calls.append((question, mode))
        return _Result()


@pytest.fixture
def cli(settings, monkeypatch):
    """
    Function name: cli
    Purpose: Fixture that points the command-line app at the test settings and silences
        logging set-up, and returns a runner.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces get_settings and configure_logging.
    Returns:
        CliRunner: The Typer test runner.
    """
    monkeypatch.setattr(cli_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_mod, "configure_logging", lambda level: None)
    return CliRunner()


def test_cli_help_lists_all_commands(cli):
    """
    Function name: test_cli_help_lists_all_commands
    Purpose: Check that --help succeeds and shows the app help and the three commands.
    Parameters:
        cli (CliRunner): Runner fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    r = cli.invoke(cli_mod.app, ["--help"])
    assert r.exit_code == 0
    for word in ("Voice RAG Assistant", "ingest", "ask", "serve"):
        assert word in r.output


def test_cli_without_command_shows_usage_error(cli):
    """
    Function name: test_cli_without_command_shows_usage_error
    Purpose: Check that running the app with no command exits with a non-zero code, and that
        an unknown command is also rejected.
    Parameters:
        cli (CliRunner): Runner fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert cli.invoke(cli_mod.app, []).exit_code != 0
    assert cli.invoke(cli_mod.app, ["nope"]).exit_code == 2


def test_cli_ingest_prints_manifest_json(cli, settings, monkeypatch):
    """
    Function name: test_cli_ingest_prints_manifest_json
    Purpose: Check that ingest calls build_index with the settings and prints its manifest as
        indented JSON.
    Parameters:
        cli (CliRunner): Runner fixture.
        settings (Settings): Test settings fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces build_index.
    Returns:
        None: The test passes or fails through assertions.
    """
    seen = []

    def fake_build(s):
        """
        Function name: fake_build
        Purpose: Record the settings and return a fixed manifest.
        Parameters:
            s (Settings): Settings passed by the command.
        Returns:
            dict: A fixed manifest.
        """
        seen.append(s)
        return {"documents": 2, "chunks": 5}

    monkeypatch.setattr(build_mod, "build_index", fake_build)
    r = cli.invoke(cli_mod.app, ["ingest"])
    assert r.exit_code == 0
    assert json.loads(r.output) == {"documents": 2, "chunks": 5}
    assert r.output.startswith('{\n  "documents"')
    assert seen == [settings]


def test_cli_ingest_failure_gives_non_zero_exit(cli, monkeypatch):
    """
    Function name: test_cli_ingest_failure_gives_non_zero_exit
    Purpose: Check that an error from build_index ends the command with a non-zero exit code
        and keeps the exception.
    Parameters:
        cli (CliRunner): Runner fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces build_index.
    Returns:
        None: The test passes or fails through assertions.
    """

    def boom(s):
        """
        Function name: boom
        Purpose: Raise FileNotFoundError as an empty documents folder would.
        Parameters:
            s (Settings): Settings passed by the command.
        Returns:
            None: Always raises.
        """
        raise FileNotFoundError("no .pdf/.md/.txt files under x")

    monkeypatch.setattr(build_mod, "build_index", boom)
    r = cli.invoke(cli_mod.app, ["ingest"])
    assert r.exit_code == 1
    assert isinstance(r.exception, FileNotFoundError)


def test_cli_ask_prints_answer_json_with_default_mode(cli, settings, monkeypatch):
    """
    Function name: test_cli_ask_prints_answer_json_with_default_mode
    Purpose: Check that ask builds the pipeline from the settings, asks the question in full
        mode by default and prints the result as JSON.
    Parameters:
        cli (CliRunner): Runner fixture.
        settings (Settings): Test settings fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces build_pipeline.
    Returns:
        None: The test passes or fails through assertions.
    """
    import rag.pipeline as pipeline_mod

    pipe = _Pipeline()
    built = []
    monkeypatch.setattr(pipeline_mod, "build_pipeline", lambda s: built.append(s) or pipe)
    r = cli.invoke(cli_mod.app, ["ask", "What is the fee?"])
    assert r.exit_code == 0
    assert json.loads(r.output) == {"answer": "42", "found": True}
    assert pipe.calls == [("What is the fee?", "full")]
    assert built == [settings]


def test_cli_ask_passes_mode_option(cli, monkeypatch):
    """
    Function name: test_cli_ask_passes_mode_option
    Purpose: Check that --mode is passed on to the pipeline.
    Parameters:
        cli (CliRunner): Runner fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces build_pipeline.
    Returns:
        None: The test passes or fails through assertions.
    """
    import rag.pipeline as pipeline_mod

    pipe = _Pipeline()
    monkeypatch.setattr(pipeline_mod, "build_pipeline", lambda s: pipe)
    r = cli.invoke(cli_mod.app, ["ask", "q", "--mode", "bm25"])
    assert r.exit_code == 0
    assert pipe.calls == [("q", "bm25")]


def test_cli_ask_requires_a_question(cli):
    """
    Function name: test_cli_ask_requires_a_question
    Purpose: Check that ask without a question is a usage error.
    Parameters:
        cli (CliRunner): Runner fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert cli.invoke(cli_mod.app, ["ask"]).exit_code == 2


def test_cli_serve_runs_uvicorn_with_factory(cli, monkeypatch):
    """
    Function name: test_cli_serve_runs_uvicorn_with_factory
    Purpose: Check that serve starts uvicorn with the app factory and the default host and
        port, and that --host and --port override them.
    Parameters:
        cli (CliRunner): Runner fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces uvicorn.run.
    Returns:
        None: The test passes or fails through assertions.
    """
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    assert cli.invoke(cli_mod.app, ["serve"]).exit_code == 0
    assert calls[-1] == (
        ("api.main:create_app",),
        {"factory": True, "host": "127.0.0.1", "port": 8000},
    )
    assert cli.invoke(cli_mod.app, ["serve", "--host", "0.0.0.0", "--port", "9001"]).exit_code == 0
    assert calls[-1][1]["host"] == "0.0.0.0" and calls[-1][1]["port"] == 9001
