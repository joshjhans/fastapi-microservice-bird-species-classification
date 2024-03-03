from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.cors import CORSMiddleware

from src.events.lifespan import lifespan
from src.exceptions.http_exception_handler import http_exception_handler
from src.routers.base_http_router import base_http_router
from src.settings.config import get_app_settings


def get_application() -> FastAPI:
    """Get `FastAPI` application object

    Returns:
        FastAPI: `FastAPI` application object
    """
    settings = get_app_settings()

    settings.configure_logging()

    application = FastAPI(
        **settings.fastapi_kwargs,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    Instrumentator().instrument(
        app=application,
    ).expose(
        app=application,
        tags=["Metrics 👓"],
    )

    application.add_exception_handler(
        HTTPException,
        http_exception_handler,
    )

    application.include_router(base_http_router)

    return application


application = get_application()
