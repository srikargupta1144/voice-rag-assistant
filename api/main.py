"""
File: api/main.py

Description:
    FastAPI application factory.

Flow:
    uv run uvicorn api.main:create_app --factory --port 8000   (or: uv run vra serve)
        -> create_app(settings=None, services=None)
            -> common.logging.configure_logging()
            -> api.services.Services(settings)              # unless injected (tests)
            -> lifespan: Services.warm_up() when api.warm_up  # models load before traffic
            -> RequestIdMiddleware, CORS
            -> api.routes.router        # /health /ingest /ask /ask/audio /ws/voice
            -> NoCacheStaticFiles(web/) at /                # record-button page, never cached

Libraries used:
    logging - standard library; module logger for warm-up messages
    os - standard library; sets the KMP_DUPLICATE_LIB_OK environment variable
    contextlib - standard library; asynccontextmanager for the application lifespan
    fastapi - third party; FastAPI application, thread pool helper, CORS and static files
    api.middleware - project; RequestIdMiddleware for request ids and access logs
    api.routes - project; router with all HTTP and WebSocket routes
    api.services - project; Services holding the pipeline, STT and TTS
    common.logging - project; configure_logging for the root logger
    configs - project; REPO path, Settings model and get_settings

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  Page served with Cache-Control: no-store (NoCacheStaticFiles).
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.middleware import RequestIdMiddleware
from api.routes import router
from api.services import Services
from common.logging import configure_logging
from configs import REPO, Settings, get_settings

# torch and ctranslate2/onnxruntime each ship an OpenMP runtime on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
log = logging.getLogger(__name__)
WEB_DIR = REPO / "web"


class NoCacheStaticFiles(StaticFiles):
    """
    Class name: NoCacheStaticFiles
    Purpose: StaticFiles that tells the browser not to cache the page. Without this a browser
             kept showing an old copy of web/index.html after it was updated, and the old
             page's errors looked like bugs in the new one.
    """

    def file_response(self, *args, **kwargs):
        """
        Function name: NoCacheStaticFiles.file_response
        Purpose: Build the normal static file response and add the header
                 "Cache-Control: no-store" so the browser always fetches a fresh copy.
        Parameters:
            args (tuple): Positional arguments passed unchanged to StaticFiles.file_response.
            kwargs (dict): Keyword arguments passed unchanged to StaticFiles.file_response.
        Returns:
            Response: The file response with the Cache-Control header set.
        """
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """
    Function name: create_app
    Purpose: Build the FastAPI application with logging, services, middleware, routes and the
             static web page. Tests inject settings and a Services object holding fakes.
    Parameters:
        settings (Settings | None): Settings to use; None loads them with get_settings().
        services (Services | None): Prebuilt services (used by tests); None creates a new
                                    Services object and enables warm-up at startup.
    Returns:
        FastAPI: The configured application.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    svc = services or Services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Function name: lifespan
        Purpose: Application lifespan handler. Loads all models in the thread pool before
                 the first request when api.warm_up is on and services were not injected.
        Parameters:
            app (FastAPI): The application being started.
        Returns:
            AsyncIterator[None]: yields once while the application is running.
        """
        if settings.api.warm_up and services is None:
            log.info("warm_up start")
            await run_in_threadpool(svc.warm_up)
            log.info("warm_up done")
        yield

    app = FastAPI(
        title="Voice RAG Assistant",
        version="0.1.0",
        description="Ask questions about a document set by voice or text; answers cite sources.",
        lifespan=lifespan,
    )
    app.state.services = svc
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.include_router(router)
    if WEB_DIR.exists():
        app.mount("/", NoCacheStaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app
