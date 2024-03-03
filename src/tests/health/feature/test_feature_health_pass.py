from fastapi import status
from src.clients.api.health.health import HealthClient


def test_feature_health_check_pass_200(
    authorized_health_client: HealthClient,
):
    response, obj = authorized_health_client.health_check()

    assert response.status_code == status.HTTP_200_OK
    assert obj is not None
    assert obj.is_healthy is True


def test_feature_health_ping_pass_200(
    authorized_health_client: HealthClient,
):
    response, obj = authorized_health_client.health_check()

    assert response.status_code == status.HTTP_200_OK
    assert obj is not None
    assert obj.is_healthy is True
