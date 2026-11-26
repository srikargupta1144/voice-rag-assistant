"""Per-stage wall-clock timing for one request (stt, rewrite, retrieval, llm, tts).

Flow:
    rag.pipeline.AnswerPipeline / api.routes
        -> timings = StageTimer()
        -> with timings.stage("retrieval"): ...      # records milliseconds
        -> timings.mark("llm_first_token")           # time since the timer started
        -> timings.as_dict()                         # returned in API responses and logged
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """Collects stage durations and milestone offsets in milliseconds."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._values: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block; repeated stages with the same name accumulate."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000)

    def add(self, name: str, ms: float) -> None:
        """Add ms to stage name (accumulates)."""
        self._values[f"{name}_ms"] = round(self._values.get(f"{name}_ms", 0.0) + ms, 1)

    def mark(self, name: str, *, once: bool = True) -> None:
        """Record the offset since the timer started, e.g. llm_first_token."""
        key = f"{name}_ms"
        if once and key in self._values:
            return
        self._values[key] = round((time.perf_counter() - self._t0) * 1000, 1)

    def as_dict(self) -> dict[str, float]:
        """All recorded values plus total_ms so far."""
        out = dict(self._values)
        out["total_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        return out

    def log_line(self) -> str:
        """key=value rendering for log lines."""
        return " ".join(f"{k}={v}" for k, v in self.as_dict().items())
