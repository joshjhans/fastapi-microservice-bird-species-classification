import pytest
from fastapi.testclient import TestClient
from httpx import BasicAuth, Client

from src.clients.api.health.health import HealthClient
from src.clients.api.model.model import ModelClient
from src.main import application  # type: ignore
from src.settings.config import settings


@pytest.fixture
def test_client() -> TestClient:
    test_client = TestClient(app=application)
    return test_client


@pytest.fixture
def unauthorized_http_client(
    test_client: TestClient,
) -> Client:
    return test_client


@pytest.fixture
def authorized_http_client(
    test_client: TestClient,
) -> Client:
    assert settings.username is not None, "`settings.username` is `None`"
    assert settings.password is not None, "`settings.password` is `None`"

    client = test_client

    client.auth = BasicAuth(
        username=settings.username,
        password=settings.password,
    )

    return client


@pytest.fixture
def unauthorized_health_client(
    unauthorized_http_client: Client,
) -> HealthClient:
    unauthorized_health_client = HealthClient(
        client=unauthorized_http_client,
    )

    return unauthorized_health_client


@pytest.fixture
def authorized_health_client(
    authorized_http_client: Client,
) -> HealthClient:
    authorized_health_client = HealthClient(
        client=authorized_http_client,
    )

    return authorized_health_client


@pytest.fixture
def unauthorized_model_client(
    unauthorized_http_client: Client,
) -> ModelClient:
    unauthorized_model_client = ModelClient(
        client=unauthorized_http_client,
    )

    return unauthorized_model_client


@pytest.fixture
def authorized_model_client(
    authorized_http_client: Client,
) -> ModelClient:
    authorized_model_client = ModelClient(
        client=authorized_http_client,
    )

    return authorized_model_client


@pytest.fixture
def test_image() -> bytes:
    with open("tests/model/data/asian_green_bee_eater.jpg", "rb") as f:
        return f.read()
