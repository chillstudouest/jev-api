"""Jev-compatible error helpers."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
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
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail: Any = exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": jsonable_encoder(detail)},
        headers=getattr(exc, "headers", None),
    )
