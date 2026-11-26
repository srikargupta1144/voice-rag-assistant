"""
File: ingest/cli.py

Description:
    Command line entry point (`uv run vra ...`).

Flow:
    vra ingest   -> ingest.build.build_index(settings)
    vra ask "q"  -> rag.pipeline.build_pipeline(settings).answer(q)   # text question, no API
    vra serve    -> uvicorn api.main:create_app

Libraries used:
    json - standard library; prints results as indented JSON
    os - standard library; sets KMP_DUPLICATE_LIB_OK for the OpenMP runtimes on Windows
    typer - third party; command line application and argument parsing
    uvicorn - third party; runs the API server for the serve command
    common.logging - project; configure_logging
    configs - project; get_settings
    ingest.build - project; build_index
    rag.pipeline - project; build_pipeline

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format;
                              short one-line help text passed to each command.
"""

from __future__ import annotations

import json
import os

import typer

from common.logging import configure_logging
from configs import get_settings

# torch and ctranslate2/onnxruntime each ship an OpenMP runtime on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = typer.Typer(add_completion=False, help="Voice RAG Assistant")


@app.command(help="Index every document under data/docs (FAISS or Pinecone + BM25).")
def ingest() -> None:
    """
    Function name: ingest
    Purpose: Index every document under data/docs (FAISS or Pinecone + BM25) and print the
             ingestion manifest as JSON.
    Parameters:
        None
    Returns:
        None
    """
    from ingest.build import build_index

    settings = get_settings()
    configure_logging(settings.log_level)
    typer.echo(json.dumps(build_index(settings), indent=2))


@app.command(help="Answer one text question from the indexed documents.")
def ask(question: str, mode: str = "full") -> None:
    """
    Function name: ask
    Purpose: Answer one text question from the indexed documents and print the result as
             JSON.
    Parameters:
        question (str): the question to answer
        mode (str): retrieval mode passed to the pipeline (dense, bm25, hybrid or full)
    Returns:
        None
    """
    from rag.pipeline import build_pipeline

    settings = get_settings()
    configure_logging(settings.log_level)
    result = build_pipeline(settings).answer(question, mode=mode)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command(help="Run the API and web page.")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Function name: serve
    Purpose: Run the API and web page with uvicorn, using the api.main.create_app factory.
    Parameters:
        host (str): network address to listen on
        port (int): port to listen on
    Returns:
        None
    """
    import uvicorn

    uvicorn.run("api.main:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
