from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.settings.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.username is None:
        raise Exception(
            "`username` setting is required; this may be set as environment variable."
        )
    if settings.password is None:
        raise Exception(
            "`password` setting is required; this may be set as environment variable."
        )

    yield
