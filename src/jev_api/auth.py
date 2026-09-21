"""Bearer API-key authentication for /v1/* routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from jev_api.config import Settings


def _settings_from_request(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is not None:
        return settings
    from jev_api.config import get_settings

    return get_settings()


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    expected = _settings_from_request(request).jev_api_key
    if not expected:
        # Misconfiguration: refuse rather than run open.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JEV_API_KEY is not configured on the server",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer <API_KEY>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
