from fastapi import status
from src.clients.api.health.health import HealthClient


def test_feature_health_check_fail_401(
    unauthorized_health_client: HealthClient,
):
    response, obj = unauthorized_health_client.health_check()

    assert response.status_code != status.HTTP_401_UNAUTHORIZED
    assert obj is not None
    assert obj.is_healthy is True


def test_feature_health_ping_fail_401(
    unauthorized_health_client: HealthClient,
):
    response, obj = unauthorized_health_client.health_check()

    assert response.status_code != status.HTTP_401_UNAUTHORIZED
    assert obj is not None
    assert obj.is_healthy is True
