"""Jev-compatible error helpers.

Observed TypeSafe / jev-agent shapes (jevaiguide.com, live probes):

  401 → {"detail": {"error_type": "authentication_error", "message": "..."}}
  403 → {"detail": {"error_type": "authentication_error", "message": "Must supply an API key!..."}}
  400 → {"detail": {"error_type": "api_usage_error"|"max_tokens_exceeded", "message": "..."}}
  422 → FastAPI-style list of {loc, msg, type}
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def auth_detail(message: str) -> dict[str, str]:
    return {"error_type": "authentication_error", "message": message}


def usage_detail(message: str, error_type: str = "api_usage_error") -> dict[str, str]:
    return {"error_type": error_type, "message": message}


def raise_auth(status_code: int, message: str) -> None:
    raise HTTPException(status_code=status_code, detail=auth_detail(message))


def raise_usage(status_code: int, message: str, error_type: str = "api_usage_error") -> None:
    raise HTTPException(status_code=status_code, detail=usage_detail(message, error_type))


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Preserve FastAPI/Pydantic list shape used by TypeSafe 422 responses.
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail: Any = exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
        headers=getattr(exc, "headers", None),
    )
