"""Download every local model into models/ (one-time, about 700 MB in total).

Flow:
    uv run python -m scripts.download_models [--skip-voice] [--skip-search]
        -> download_whisper(settings)  # Systran/faster-whisper-<size> -> models/faster-whisper-...
        -> download_piper(settings)    # piper-voices <voice>.onnx + .onnx.json -> models/piper
        -> download_search(settings)   # bge-small + MiniLM cross-encoder -> models/hf

CI runs only the search part (--skip-voice): the retrieval gate needs no speech models.
The Ollama model is pulled separately with `ollama pull llama3.2:3b`.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download, snapshot_download

from configs import Settings, get_settings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PIPER_REPO = "rhasspy/piper-voices"


def piper_voice_path(voice: str) -> str:
    """Repository path of a Piper voice, e.g. en_US-lessac-medium -> en/en_US/lessac/medium/..."""
    lang, name, quality = voice.split("-", 2)
    return f"{lang.split('_')[0]}/{lang}/{name}/{quality}/{voice}"


def download_whisper(settings: Settings) -> None:
    """faster-whisper CTranslate2 weights for the configured size."""
    target = settings.paths.resolve(settings.stt.model_dir)
    snapshot_download(f"Systran/faster-whisper-{settings.stt.model_size}", local_dir=str(target))
    print(f"whisper -> {target}")


def download_piper(settings: Settings) -> None:
    """One Piper voice: the ONNX model and its JSON config."""
    target = settings.paths.resolve(settings.tts.voice_dir)
    base = piper_voice_path(settings.tts.voice)
    for suffix in (".onnx", ".onnx.json"):
        path = hf_hub_download(PIPER_REPO, base + suffix, local_dir=str(target))
        # hf_hub_download keeps the repo sub-folders; flatten to models/piper/<voice>.onnx
        flat = target / (settings.tts.voice + suffix)
        if os.path.abspath(path) != str(flat):
            os.replace(path, flat)
    print(f"piper  -> {target / (settings.tts.voice + '.onnx')}")


def download_search(settings: Settings) -> None:
    """Embedding model and re-ranker, loaded once so they land in models/hf."""
    from retrieval.embeddings import get_encoder
    from retrieval.rerank import get_reranker

    get_encoder(settings).embed_query("warm up")
    get_reranker(settings).score("warm up", ["warm up"])
    print(f"search -> {settings.paths.resolve(settings.paths.models_dir) / 'hf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-voice", action="store_true", help="skip whisper and piper")
    parser.add_argument("--skip-search", action="store_true", help="skip embedder and re-ranker")
    args = parser.parse_args()
    settings = get_settings()
    if not args.skip_voice:
        download_whisper(settings)
        download_piper(settings)
    if not args.skip_search:
        download_search(settings)


if __name__ == "__main__":
    main()
