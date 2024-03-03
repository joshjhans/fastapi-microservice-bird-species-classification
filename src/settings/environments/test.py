import logging

from pydantic_settings import SettingsConfigDict

from src.settings.environments.base import BaseAppSettings
from src.settings.types.app_env import AppEnv


class TestAppSettings(BaseAppSettings):
    app_env: AppEnv = "test"
    debug: bool = True

    logging_level: int = logging.DEBUG

    model_config = SettingsConfigDict(
        env_file="/src/.env",
    )
