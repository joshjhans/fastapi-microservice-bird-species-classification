from typing import Annotated

from fastapi import APIRouter, Depends, File
from src.dependencies.get_current_username.get_current_username import (
    get_current_username,
)
from src.dependencies.get_model_service.get_model_service import get_model_service
from src.services.model.model import ModelService, Prediction

model_router = APIRouter(
    prefix="/models",
    tags=["Models"],
    dependencies=[Depends(get_current_username)],
)


@model_router.post(
    path="/predict",
    name="models:predict",
)
async def models_predict(
    image: Annotated[bytes, File()],
    model_service: Annotated[ModelService, Depends(get_model_service)],
    top_k: int = 3,
) -> Prediction:
    prediction = model_service.predict(
        image=image,
        top_k=top_k,
    )
    return prediction
