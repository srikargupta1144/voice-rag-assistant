"""FastAPI application factory.

Flow:
    uv run uvicorn api.main:create_app --factory --port 8000   (or: uv run vra serve)
        -> create_app(settings=None, services=None)
            -> common.logging.configure_logging()
            -> api.services.Services(settings)              # unless injected (tests)
            -> lifespan: Services.warm_up() when api.warm_up  # models load before traffic
            -> RequestIdMiddleware, CORS
            -> api.routes.router        # /health /ingest /ask /ask/audio /ws/voice
            -> NoCacheStaticFiles(web/) at /                # record-button page, never cached
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
    """StaticFiles that tells the browser not to cache the page.

    Without this a browser kept showing an old copy of web/index.html after it was
    updated, and the old page's errors looked like bugs in the new one.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response

def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app; tests inject settings and a Services object holding fakes."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    svc = services or Services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
