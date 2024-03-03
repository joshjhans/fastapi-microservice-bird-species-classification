from fastapi import APIRouter
from src.clients.schemas.health.health import HealthSchema, PingSchema

health_router = APIRouter(
    prefix="/health",
    tags=["Health ❤️"],
)


@health_router.get(
    path="/check",
    name="health:check",
)
async def health_check() -> HealthSchema:
    return HealthSchema()


@health_router.get(
    path="/ping",
    name="health:ping",
)
async def health_ping() -> PingSchema:
    return PingSchema()
