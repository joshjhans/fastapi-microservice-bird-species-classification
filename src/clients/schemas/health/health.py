from typing import Literal

from src.clients.schemas.base import AppBaseModel


class HealthSchema(AppBaseModel):
    is_healthy: bool = True


class PingSchema(AppBaseModel):
    response: Literal["pong"] = "pong"
