"""
File: api/middleware.py

Description:
    Request ids and access logging.

Flow:
    every HTTP request
        -> RequestIdMiddleware.dispatch()
            -> request id = incoming X-Request-ID header or a new 12-hex id
            -> common.logging.request_id_var.set(id)   # all log lines in the request carry it
            -> call_next() -> response header X-Request-ID + one access log line

Libraries used:
    logging - standard library; access logger "api.access"
    re - standard library; checks that an incoming request id is safe to reuse
    time - standard library; measures request duration
    uuid - standard library; generates new request ids
    starlette - third party; BaseHTTPMiddleware, Request and Response types
    common.logging - project; request_id_var context variable for log lines

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
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from common.logging import request_id_var

log = logging.getLogger("api.access")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id() -> str:
    """
    Function name: new_request_id
    Purpose: Create a new request id from the first 12 hex characters of a random UUID.
    Parameters:
        None
    Returns:
        str: A 12-character hexadecimal request id.
    """
    return uuid.uuid4().hex[:12]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Class name: RequestIdMiddleware
    Purpose: Gives every HTTP request an id, makes it available to all log lines, returns it
             in the X-Request-ID header and writes one access log line per request.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Function name: RequestIdMiddleware.dispatch
        Purpose: Reuse a safe incoming X-Request-ID header or create a new id, store it on
                 request.state and in request_id_var, call the next handler, then add the
                 X-Request-ID response header and log method, path, status and duration.
        Parameters:
            request (Request): The incoming HTTP request.
            call_next (Callable): Starlette callable that runs the rest of the application.
        Returns:
            Response: The response from the application with the X-Request-ID header added.
        """
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _SAFE_ID.match(incoming) else new_request_id()
        request.state.request_id = rid
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        log.info(
            "method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
            rid,
        )
        return response
