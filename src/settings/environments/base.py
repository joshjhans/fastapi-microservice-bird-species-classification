import logging
import sys
from typing import Any

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.settings import __version__
from src.settings.logging.intercept_handler import InterceptHandler
from src.settings.types.app_env import AppEnv


class BaseAppSettings(BaseSettings):
    app_env: AppEnv = "devl"
    debug: bool = False

    docs_url: str = "/docs"
    openapi_prefix: str = ""
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/redoc"
    title: str = "🐦 Bird Species Image Classification"
    version: str = __version__

    username: str | None = None
    password: str | None = None

    kaggle_username: str | None = None
    kaggle_key: str | None = None

    allowed_hosts: list[str] = ["*"]

    logging_level: int = logging.INFO
    loggers: tuple[str, str] = ("uvicorn.asgi", "uvicorn.access")

    @property
    def fastapi_kwargs(self) -> dict[str, Any]:
        return {
            "debug": self.debug,
            "docs_url": self.docs_url,
            "openapi_prefix": self.openapi_prefix,
            "openapi_url": self.openapi_url,
            "redoc_url": self.redoc_url,
            "title": self.title,
            "version": self.version,
        }

    def configure_logging(self) -> None:
        logging.getLogger().handlers = [InterceptHandler()]
        for logger_name in self.loggers:
            logging_logger = logging.getLogger(logger_name)
            logging_logger.handlers = [InterceptHandler(level=self.logging_level)]

        logger.configure(
            handlers=[
                {
                    "sink": sys.stderr,
                    "level": self.logging_level,
                }
            ]
        )

    model_config = SettingsConfigDict(
        validate_assignment=True,
    )
