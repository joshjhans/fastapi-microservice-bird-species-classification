import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from src.settings.config import get_app_settings
from src.settings.environments.base import BaseAppSettings

security = HTTPBasic()


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    settings: Annotated[BaseAppSettings, Depends(get_app_settings)],
):
    assert settings.username is not None
    assert settings.password is not None
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = settings.username.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes,
        correct_username_bytes,
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = settings.password.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes,
        correct_password_bytes,
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
