"""
File: api/services.py

Description:
    Lazily built, process-wide components that the routes depend on.

    Tests pass a Services subclass (or set the private fields) with fakes, so no model, LLM
    or audio library is touched.

Flow:
    api.main.create_app(services=None)
        -> Services(settings)               # nothing heavy is loaded yet
        -> lifespan: Services.warm_up()     # optional: load models before the first request
    api.routes (every handler)
        -> services.pipeline()  -> rag.pipeline.build_pipeline()   (retriever + LLM)
        -> services.stt()       -> voice.stt.get_stt()
        -> services.tts()       -> voice.tts.get_tts()
        -> services.reload_index()  after POST /ingest

Libraries used:
    logging - standard library; module logger for warm-up failures
    threading - standard library; lock that guards building and resetting the pipeline
    typing - standard library; TYPE_CHECKING and Any for type hints
    configs - project; Settings type (type checking only)
    rag.pipeline - project; build_pipeline creates the answer pipeline (imported on first use)
    ingest.build - project; build_index rebuilds the search index (imported on first use)
    voice.stt - project; get_stt creates the speech-to-text model (imported on first use)
    voice.tts - project; get_tts creates the text-to-speech voice (imported on first use)

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
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
    """
    Class name: Services
    Purpose: Holds the answer pipeline, STT and TTS; each is created on first use.
    """

    def __init__(self, settings: Settings):
        """
        Function name: Services.__init__
        Purpose: Store the settings and prepare empty slots for the lazily built components.
        Parameters:
            settings (Settings): Application settings used to build each component.
        Returns:
            None
        """
        self.settings = settings
        self._pipeline: AnswerPipeline | None = None
        self._stt: SpeechToText | None = None
        self._tts: TextToSpeech | None = None
        self._lock = threading.Lock()

    def pipeline(self) -> AnswerPipeline:
        """
        Function name: Services.pipeline
        Purpose: Return the answer pipeline (retriever and LLM), building it under a lock on
                 first use or after the index was reloaded.
        Parameters:
            None
        Returns:
            AnswerPipeline: The shared answer pipeline.
        """
        with self._lock:
            if self._pipeline is None:
                from rag.pipeline import build_pipeline

                self._pipeline = build_pipeline(self.settings)
            return self._pipeline

    def stt(self) -> SpeechToText:
        """
        Function name: Services.stt
        Purpose: Return the speech-to-text component, creating it with get_stt on first use.
        Parameters:
            None
        Returns:
            SpeechToText: The shared speech-to-text component.
        """
        if self._stt is None:
            from voice.stt import get_stt

            self._stt = get_stt(self.settings)
        return self._stt

    def tts(self) -> TextToSpeech:
        """
        Function name: Services.tts
        Purpose: Return the text-to-speech component, creating it with get_tts on first use.
        Parameters:
            None
        Returns:
            TextToSpeech: The shared text-to-speech component.
        """
        if self._tts is None:
            from voice.tts import get_tts

            self._tts = get_tts(self.settings)
        return self._tts

    def ingest(self) -> dict[str, Any]:
        """
        Function name: Services.ingest
        Purpose: Rebuild the index from data/docs and drop the current pipeline so that a
                 fresh retriever is built on the next request.
        Parameters:
            None
        Returns:
            dict[str, Any]: The manifest returned by ingest.build.build_index.
        """
        from ingest.build import build_index

        manifest = build_index(self.settings)
        self.reload_index()
        return manifest

    def reload_index(self) -> None:
        """
        Function name: Services.reload_index
        Purpose: Discard the current pipeline under the lock so the next call to pipeline()
                 loads the new index.
        Parameters:
            None
        Returns:
            None
        """
        with self._lock:
            self._pipeline = None

    def status(self) -> dict[str, Any]:
        """
        Function name: Services.status
        Purpose: Report component readiness for GET /health without loading anything heavy.
                 When the pipeline is loaded it also reports the chunk count, the LLM model
                 and whether the LLM is reachable.
        Parameters:
            None
        Returns:
            dict[str, Any]: Readiness flags and component details.
        """
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
        """
        Function name: Services.warm_up
        Purpose: Load every component and run one small retrieval and one short synthesis
                 so the first real call is not slow. Failures are logged as warnings and do
                 not stop startup, so a missing voice model does not block text answers.
        Parameters:
            None
        Returns:
            None
        """
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
