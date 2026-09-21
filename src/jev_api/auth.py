"""Bearer API-key authentication — Jev-compatible 401/403 shapes."""

from __future__ import annotations

from fastapi import Header, Request

from jev_api.config import Settings
from jev_api.errors import raise_auth, raise_usage


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
        raise_usage(503, "JEV_API_KEY is not configured on the server", error_type="server_error")

    if authorization is None or not authorization.strip():
        # TypeSafe: missing key → 403
        raise_auth(403, "Must supply an API key! Check your request and try again.")

    if not authorization.startswith("Bearer "):
        raise_auth(401, "Cannot authenticate with the server. Please check your API key and try again.")

    token = authorization.removeprefix("Bearer ").strip()
    if not token or token != expected:
        raise_auth(401, "Cannot authenticate with the server. Please check your API key and try again.")
