from fastapi import APIRouter
from src.routers.health.health import health_router
from src.routers.models.models import model_router

base_http_router = APIRouter()

base_http_router.include_router(health_router)
base_http_router.include_router(model_router)
