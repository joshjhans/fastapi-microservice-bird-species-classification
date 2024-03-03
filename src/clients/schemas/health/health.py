from src.clients.schemas.base import AppBaseModel


class HealthSchema(AppBaseModel):
    is_healthy: bool = True


class PingSchema(AppBaseModel):
    response: str = "pong"
