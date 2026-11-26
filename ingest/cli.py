"""Command line entrypoint (`uv run vra ...`).

Flow:
    vra ingest   -> ingest.build.build_index(settings)
    vra ask "q"  -> rag.pipeline.build_pipeline(settings).answer(q)   # text question, no API
    vra serve    -> uvicorn api.main:create_app
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


@app.command()
def ingest() -> None:
    """Index every document under data/docs (FAISS or Pinecone + BM25)."""
    from ingest.build import build_index

    settings = get_settings()
    configure_logging(settings.log_level)
    typer.echo(json.dumps(build_index(settings), indent=2))


@app.command()
def ask(question: str, mode: str = "full") -> None:
    """Answer one text question from the indexed documents."""
    from rag.pipeline import build_pipeline

    settings = get_settings()
    configure_logging(settings.log_level)
    result = build_pipeline(settings).answer(question, mode=mode)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API and web page."""
    import uvicorn

    uvicorn.run("api.main:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
