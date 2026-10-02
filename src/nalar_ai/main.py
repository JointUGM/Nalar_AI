"""FastAPI application factory. Run with: uvicorn --factory nalar_ai.main:create_app"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.formparsers import MultiPartParser

from nalar_ai import __version__
from nalar_ai.container import Container, build_container
from nalar_ai.platform.embeddings.router import router as embeddings_router
from nalar_ai.platform.http.errors import register_error_handlers
from nalar_ai.platform.http.health import router as health_router
from nalar_ai.platform.http.middleware import (
    MULTIPART_SLACK_BYTES,
    BodySizeLimitMiddleware,
    RequestIdMiddleware,
)
from nalar_ai.settings import get_settings
from nalar_ai.subsystems.s1_knowledge_base.api.router import router as s1_router
from nalar_ai.subsystems.s2_mission_designer.api.router import router as s2_router
from nalar_ai.subsystems.s3_socratic_prober.api.router import router as s3_router
from nalar_ai.subsystems.s4_session_evaluator.api.router import router as s4_router
from nalar_ai.subsystems.s5_insight_synthesizer.api.router import router as s5_router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await app.state.container.aclose()


def create_app(container: Container | None = None) -> FastAPI:
    container = container or build_container(get_settings())
    app = FastAPI(
        title="NALAR AI Service",
        version=__version__,
        summary="Stateless AI service for NALAR. Called by nalar-backend only.",
        lifespan=_lifespan,
    )
    app.state.container = container
    max_body_bytes = container.settings.max_upload_bytes + MULTIPART_SLACK_BYTES
    # Teacher material stays in memory (design doc §15): never spool uploads to temp files.
    # Process-wide Starlette setting; the body limit below bounds the memory it can take.
    MultiPartParser.spool_max_size = max_body_bytes
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=max_body_bytes)
    app.add_middleware(RequestIdMiddleware)  # added last = outermost, so request_id is set first
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(embeddings_router)
    app.include_router(s1_router)
    app.include_router(s2_router)
    app.include_router(s3_router)
    app.include_router(s4_router)
    app.include_router(s5_router)
    return app
