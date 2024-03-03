from fastapi import status
from httpx import Client, Response

from src.clients.schemas.health.health import HealthSchema, PingSchema


class HealthClient:
    def __init__(
        self,
        client: Client,
    ) -> None:
        self.client = client

    def health_check(
        self,
    ) -> tuple[Response, HealthSchema | None]:
        """API health check

        Returns:
            tuple[Response, HealthSchema | None]: returns `Response` object and `HealthSchema` if
            request successful
        """
        url: str = "/health/check"

        response = self.client.get(
            url=url,
        )

        if response.status_code == status.HTTP_200_OK:
            return (response, HealthSchema(**response.json()))

        return (response, None)

    def health_ping(
        self,
    ) -> tuple[Response, PingSchema | None]:
        """API ping

        Returns:
            tuple[Response, HealthSchema | None]: returns `Response` object and `PingSchema` if
            request successful
        """
        url: str = "/health/ping"

        response = self.client.get(
            url=url,
        )

        if response.status_code == status.HTTP_200_OK:
            return (response, PingSchema(**response.json()))

        return (response, None)
