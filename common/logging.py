"""
File: common/logging.py

Description:
    Logging setup: one key=value line per event so that logs are easy to search and parse.

Flow:
    api.main.create_app(), ingest.cli, eval scripts
        -> configure_logging(level) once at startup
    api.middleware.RequestIdMiddleware
        -> request_id_var.set(...)        # every log line in that request carries the id

Libraries used:
    contextvars - standard library; request_id_var holds the current request id
    logging - standard library; root handler, formatter and the request id filter

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import contextvars
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """
    Class name: _RequestIdFilter
    Purpose: Injects the current request id into every log record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Function name: _RequestIdFilter.filter
        Purpose: Set record.request_id from request_id_var ("-" outside a request) and keep
                 the record.
        Parameters:
            record (logging.LogRecord): The log record being handled.
        Returns:
            bool: Always True, so no record is dropped.
        """
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """
    Function name: configure_logging
    Purpose: Configure root logging once per process entry point: a single stream handler
             with the request id filter and a key=value format, replacing any old handlers.
    Parameters:
        level (str): Root log level name, e.g. "INFO".
    Returns:
        None
    """
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s level=%(levelname)s logger=%(name)s request_id=%(request_id)s %(message)s"
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
