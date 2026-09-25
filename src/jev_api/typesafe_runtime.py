"""Official TypeSafe Jev cloud (`model=jev`) behind the local System One contract."""

from __future__ import annotations

from typing import Any

import httpx

from jev_api.config import Settings
from jev_api.schemas import Usage


def call_typesafe_jev(
    settings: Settings,
    state: object,
    questions: dict[str, Any],
) -> dict[str, Any]:
    key = (settings.typesafe_api_key or "").strip()
    if not key:
        raise RuntimeError(
            "Official Jev needs TYPESAFE_API_KEY on the server. "
            "JEV_API_KEY only authenticates this proxy."
        )
    base = settings.typesafe_base_url.rstrip("/")
    payload = {
        "state": state,
        "questions": questions,
        "model": settings.typesafe_model,
    }
    try:
        response = httpx.post(
            f"{base}/v1/systemone",
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
            timeout=settings.typesafe_timeout,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"TypeSafe Jev timed out after {settings.typesafe_timeout:.0f}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"TypeSafe Jev request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:400]
        raise RuntimeError(f"TypeSafe Jev HTTP {response.status_code}: {detail}")

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("TypeSafe Jev returned a non-object body")
    raw_answers = data.get("answers")
    if not isinstance(raw_answers, dict):
        raise RuntimeError("TypeSafe Jev response is missing answers")
    from jev_api.engine import answers_from_mapping

    answers = answers_from_mapping(raw_answers, source="Jev")
    usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage = Usage(
        input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
        output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
    )
    return {
        "model": str(data.get("model") or settings.typesafe_model),
        "answers": answers,
        "usage": usage,
        "upstream_duration_ms": float(data.get("duration_ms") or 0.0),
        "upstream_inference_ms": float(data.get("inference_ms") or 0.0),
        "gpu_duration_ms": data.get("gpu_duration_ms"),
    }
