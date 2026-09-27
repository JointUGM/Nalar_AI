"""FastAPI application factory. Run with: uvicorn --factory nalar_ai.main:create_app"""

from fastapi import FastAPI

from nalar_ai import __version__
from nalar_ai.container import Container, build_container
from nalar_ai.platform.http.errors import register_error_handlers
from nalar_ai.platform.http.health import router as health_router
from nalar_ai.platform.http.middleware import RequestIdMiddleware
from nalar_ai.settings import get_settings


def create_app(container: Container | None = None) -> FastAPI:
    container = container or build_container(get_settings())
    app = FastAPI(
        title="NALAR AI Service",
        version=__version__,
        summary="Stateless AI service for NALAR. Called by nalar-backend only.",
    )
    app.state.container = container
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)
    app.include_router(health_router)
    return app
