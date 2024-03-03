import logging

from pydantic_settings import SettingsConfigDict

from src.settings.environments.base import BaseAppSettings
from src.settings.types.app_env import AppEnv


class ProdAppSettings(BaseAppSettings):
    app_env: AppEnv = "prod"
    debug: bool = False

    logging_level: int = logging.ERROR

    model_config = SettingsConfigDict(
        env_file="/src/.env",
    )
