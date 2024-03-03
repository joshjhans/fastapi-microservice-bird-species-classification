from fastapi import status
from src.clients.api.model.model import ModelClient


def test_feature_model_check_pass_200(
    authorized_model_client: ModelClient,
    test_image: bytes,
):
    response, obj = authorized_model_client.models_predict(
        top_k=3,
        image=test_image,
    )

    assert response.status_code == status.HTTP_200_OK
    assert obj is not None
