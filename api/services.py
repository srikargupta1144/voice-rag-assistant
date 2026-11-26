"""Lazily built, process-wide components the routes depend on.

Flow:
    api.main.create_app(services=None)
        -> Services(settings)               # nothing heavy is loaded yet
        -> lifespan: Services.warm_up()     # optional: load models before the first request
    api.routes (every handler)
        -> services.pipeline()  -> rag.pipeline.build_pipeline()   (retriever + LLM)
        -> services.stt()       -> voice.stt.get_stt()
        -> services.tts()       -> voice.tts.get_tts()
        -> services.reload_index()  after POST /ingest

Tests pass a Services subclass (or set the private fields) with fakes, so no model, LLM
or audio library is touched.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from configs import Settings
    from rag.pipeline import AnswerPipeline
    from voice.stt import SpeechToText
    from voice.tts import TextToSpeech

log = logging.getLogger(__name__)


class Services:
    """Holds the pipeline, STT and TTS; each is created on first use."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: AnswerPipeline | None = None
        self._stt: SpeechToText | None = None
        self._tts: TextToSpeech | None = None
        self._lock = threading.Lock()

    def pipeline(self) -> AnswerPipeline:
        with self._lock:
            if self._pipeline is None:
                from rag.pipeline import build_pipeline

                self._pipeline = build_pipeline(self.settings)
            return self._pipeline

    def stt(self) -> SpeechToText:
        if self._stt is None:
            from voice.stt import get_stt

            self._stt = get_stt(self.settings)
        return self._stt

    def tts(self) -> TextToSpeech:
        if self._tts is None:
            from voice.tts import get_tts

            self._tts = get_tts(self.settings)
        return self._tts

    def ingest(self) -> dict[str, Any]:
        """Rebuild the index from data/docs and swap in a fresh retriever."""
        from ingest.build import build_index

        manifest = build_index(self.settings)
        self.reload_index()
        return manifest

    def reload_index(self) -> None:
        with self._lock:
            self._pipeline = None

    def status(self) -> dict[str, Any]:
        """Component readiness for GET /health (never loads anything heavy)."""
        out: dict[str, Any] = {
            "llm_provider": self.settings.llm.provider,
            "vector_store": self.settings.vector_store.effective_backend(),
            "pipeline_loaded": self._pipeline is not None,
            "stt_loaded": self._stt is not None,
            "tts_loaded": self._tts is not None,
        }
        if self._pipeline is not None:
            out["chunks"] = len(self._pipeline.retriever.chunks)
            llm = self._pipeline.llm
            out["llm_model"] = llm.model
            health = getattr(llm, "health", None)
            out["llm_reachable"] = bool(health()) if callable(health) else None
        return out

    def warm_up(self) -> None:
        """Load every component (and run one tiny inference each) to avoid a slow first call."""
        for name, fn in (("pipeline", self.pipeline), ("stt", self.stt), ("tts", self.tts)):
            try:
                fn()
            except Exception as e:  # a missing voice model must not stop text answers
                log.warning("warm_up component=%s failed: %s", name, e)
        try:
            self.pipeline().retriever.retrieve("warm up", mode="full", top_k=1)
            self.tts().synthesize("Ready.")
        except Exception as e:
            log.warning("warm_up inference failed: %s", e)
