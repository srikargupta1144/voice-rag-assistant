"""
File: common/timing.py

Description:
    Per-stage wall-clock timing for one request (stt, rewrite, retrieval, llm, tts).

Flow:
    rag.pipeline.AnswerPipeline / api.routes
        -> timings = StageTimer()
        -> with timings.stage("retrieval"): ...      # records milliseconds
        -> timings.mark("llm_first_token")           # time since the timer started
        -> timings.as_dict()                         # returned in API responses and logged

Libraries used:
    time - standard library; perf_counter for wall-clock measurement
    collections.abc - standard library; Iterator type for the stage context manager
    contextlib - standard library; contextmanager decorator for stage()

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """
    Class name: StageTimer
    Purpose: Collects stage durations and milestone offsets in milliseconds.
    """

    def __init__(self) -> None:
        """
        Function name: StageTimer.__init__
        Purpose: Start the timer and create an empty store of recorded values.
        Parameters:
            None
        Returns:
            None
        """
        self._t0 = time.perf_counter()
        self._values: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """
        Function name: StageTimer.stage
        Purpose: Time a block of code; repeated stages with the same name accumulate.
        Parameters:
            name (str): Stage name; the value is stored as "<name>_ms".
        Returns:
            Iterator[None]: yields once while the timed block runs.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000)

    def add(self, name: str, ms: float) -> None:
        """
        Function name: StageTimer.add
        Purpose: Add a duration to a stage; values for the same stage accumulate and are
                 rounded to one decimal place.
        Parameters:
            name (str): Stage name; the value is stored as "<name>_ms".
            ms (float): Duration in milliseconds to add.
        Returns:
            None
        """
        self._values[f"{name}_ms"] = round(self._values.get(f"{name}_ms", 0.0) + ms, 1)

    def mark(self, name: str, *, once: bool = True) -> None:
        """
        Function name: StageTimer.mark
        Purpose: Record the time since the timer started, e.g. llm_first_token.
        Parameters:
            name (str): Milestone name; the value is stored as "<name>_ms".
            once (bool): When True, keep the first recorded value and ignore later calls.
        Returns:
            None
        """
        key = f"{name}_ms"
        if once and key in self._values:
            return
        self._values[key] = round((time.perf_counter() - self._t0) * 1000, 1)

    def as_dict(self) -> dict[str, float]:
        """
        Function name: StageTimer.as_dict
        Purpose: Return all recorded values plus total_ms so far.
        Parameters:
            None
        Returns:
            dict[str, float]: Stage and milestone values in milliseconds, with total_ms.
        """
        out = dict(self._values)
        out["total_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        return out

    def log_line(self) -> str:
        """
        Function name: StageTimer.log_line
        Purpose: Render all values from as_dict() as key=value pairs for log lines.
        Parameters:
            None
        Returns:
            str: Space-separated key=value pairs.
        """
        return " ".join(f"{k}={v}" for k, v in self.as_dict().items())
