"""HTTP client for official AutoJev-27B (denis-pplx/autojev).

Official BF16 weights need ~49 GiB plus runtime overhead (typically an 80GB GPU).
This module never loads those weights. It forwards System One requests to a
running `autojev-serve` when AUTOJEV_BASE_URL is set.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from jev_api.config import Settings
from jev_api.schemas import ApiUsageError, SystemOneRequest

logger = logging.getLogger("jev_api.autojev")

AUTOJEV_HF_REPO = "denis-pplx/autojev-27b"
AUTOJEV_PARAMETERS = 27_000_000_000
NOT_CONFIGURED_MESSAGE = (
    "AutoJev is not configured. Official denis-pplx/autojev-27b needs ~49 GiB "
    "BF16 plus overhead (typically an 80GB GPU) and will not fit the Modal T4. "
    "Set AUTOJEV_BASE_URL to an autojev-serve instance to benchmark it."
)


def autojev_configured(settings: Settings) -> bool:
    return bool(settings.autojev_base_url.strip())


def evaluate_autojev(settings: Settings, request: SystemOneRequest) -> dict[str, Any]:
    base_url = settings.autojev_base_url.strip().rstrip("/")
    if not base_url:
        raise RuntimeError(NOT_CONFIGURED_MESSAGE)

    payload: dict[str, Any] = {
        "model": settings.autojev_upstream_model,
        "state": request.state,
        "questions": request.questions,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.autojev_api_key:
        headers["Authorization"] = f"Bearer {settings.autojev_api_key}"

    logger.info("Proxying System One to AutoJev at %s", base_url)
    try:
        response = httpx.post(
            f"{base_url}/v1/systemone",
            json=payload,
            headers=headers,
            timeout=settings.autojev_timeout,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"AutoJev timed out after {settings.autojev_timeout:.0f}s at {base_url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"AutoJev is unreachable at {base_url}: {exc}") from exc

    if response.status_code == 529:
        raise RuntimeError("AutoJev is busy. Retry shortly.")
    if response.status_code in {401, 403}:
        raise ApiUsageError("AutoJev rejected the upstream API key", "authentication_error")
    if response.status_code >= 400:
        message = _error_message(response)
        if response.status_code in {400, 422}:
            raise ApiUsageError(message)
        raise RuntimeError(message)

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("AutoJev returned a non-object result")
    return data


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"AutoJev HTTP {response.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            return detail["message"]
        if isinstance(detail, str):
            return detail
        if isinstance(body.get("message"), str):
            return body["message"]
    return f"AutoJev HTTP {response.status_code}"
