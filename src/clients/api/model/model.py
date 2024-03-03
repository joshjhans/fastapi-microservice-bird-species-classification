from fastapi import status
from httpx import Client, Response
from httpx._types import RequestFiles

from src.clients.schemas.model.model import Prediction


class ModelClient:
    def __init__(
        self,
        client: Client,
    ) -> None:
        self.client = client

    def models_predict(
        self,
        image: bytes,
        top_k: int,
    ) -> tuple[Response, Prediction | None]:
        """Predict specie of bird by image

        Returns:
            tuple[Response, Prediction | None]: returns `Response` object and `Prediction` if
            request successful
        """
        url: str = "/models/predict"

        params: dict[str, int] = dict(
            top_k=top_k,
        )

        files: RequestFiles = dict(
            image=image,
        )

        response = self.client.post(
            url=url,
            params=params,
            files=files,
        )

        if response.status_code == status.HTTP_200_OK:
            return (response, Prediction(**response.json()))

        return (response, None)
