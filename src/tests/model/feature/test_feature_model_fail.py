from fastapi import status
from src.clients.api.model.model import ModelClient


def test_feature_model_predict_fail_401(
    unauthorized_model_client: ModelClient,
    test_image: bytes,
):
    response, obj = unauthorized_model_client.models_predict(
        image=test_image,
        top_k=3,
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert obj is None
