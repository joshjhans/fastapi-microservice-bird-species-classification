from functools import lru_cache
from typing import Type

from src.settings.environments.base import BaseAppSettings
from src.settings.environments.devl import DevlAppSettings
from src.settings.environments.prod import ProdAppSettings
from src.settings.environments.stage import StageAppSettings
from src.settings.environments.test import TestAppSettings
from src.settings.types.app_env import AppEnv

environments: dict[AppEnv, Type[BaseAppSettings]] = {
    "devl": DevlAppSettings,
    "test": TestAppSettings,
    "stage": StageAppSettings,
    "prod": ProdAppSettings,
}


@lru_cache
def get_app_settings() -> BaseAppSettings:
    app_env = BaseAppSettings().app_env
    settings = environments[app_env]
    return settings()


settings = get_app_settings()
