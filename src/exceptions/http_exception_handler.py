from fastapi import HTTPException, status
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse


class HttpError(BaseModel):
    status_code: int
    description: str
    detail: str | None = None


messages: dict[
    int,
    HttpError,
] = {
    status.HTTP_400_BAD_REQUEST: HttpError(
        status_code=status.HTTP_400_BAD_REQUEST,
        description=(
            "The HyperText Transfer Protocol (HTTP) 400 Bad Request response "
            "status code indicates that the server cannot or will not process "
            "the request due to something that is perceived to be a client error "
            "(for example, malformed request syntax, invalid request message framing, "
            "or deceptive request routing)."
        ),
    ),
    status.HTTP_401_UNAUTHORIZED: HttpError(
        status_code=status.HTTP_401_UNAUTHORIZED,
        description=(
            "The HyperText Transfer Protocol (HTTP) 401 Unauthorized response status code "
            "indicates that the client request has not been completed because it lacks "
            "valid authentication credentials for the requested resource."
        ),
    ),
    status.HTTP_403_FORBIDDEN: HttpError(
        status_code=status.HTTP_403_FORBIDDEN,
        description=(
            "The HTTP 403 Forbidden response status code indicates that the server understands "
            "the request but refuses to authorize it. This status is similar to 401, but for the "
            "403 Forbidden status code, re-authenticating makes no difference. The access is tied "
            "to the application logic, such as insufficient rights to a resource."
        ),
    ),
    status.HTTP_404_NOT_FOUND: HttpError(
        status_code=status.HTTP_404_NOT_FOUND,
        description=(
            "The HTTP 404 Not Found response status code indicates that the server cannot find "
            "the requested resource. Links that lead to a 404 page are often called broken or "
            "dead links and can be subject to link rot. A 404 status code only indicates that the "
            "resource is missing: not whether the absence is temporary or permanent. If a "
            "resource is permanently removed, use the 410 (Gone) status instead."
        ),
    ),
    status.HTTP_500_INTERNAL_SERVER_ERROR: HttpError(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        description=(
            "The HyperText Transfer Protocol (HTTP) 500 Internal Server Error server error "
            "response code indicates that the server encountered an unexpected condition that "
            "prevented it from fulfilling the request. This error response is a generic "
            "'catch-all' response. Usually, this indicates the server cannot find a better 5xx "
            "error code to response. Sometimes, server administrators log error responses like "
            "the 500 status code with more details about the request to prevent the error from "
            "happening again in the future."
        ),
    ),
}


async def http_exception_handler(
    _: Request,
    exc: Exception,
) -> JSONResponse:
    if isinstance(exc, HTTPException):
        status_code: int = exc.status_code
        description: str = exc.detail
        detail: str = description

        response = HttpError(
            status_code=status_code,
            description=description,
            detail=detail,
        )

        if status_code in messages:
            response = messages[status_code]

        return JSONResponse(
            response.model_dump(),
            status_code=status_code,
        )
    return JSONResponse({""})
