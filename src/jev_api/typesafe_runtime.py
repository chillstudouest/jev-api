"""Official TypeSafe Jev API client behind the System One contract.

Proxies POST https://api.typesafe.ai/v1/systemone with TYPESAFE_API_KEY.
Local model alias `jev` maps to a remote TypeSafe model (default `jev-latest`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from jev_api.config import Settings

logger = logging.getLogger("jev_api.typesafe")


@dataclass
class TypesafeClient:
    base_url: str
    api_key: str
    remote_model: str
    timeout: float

    def evaluate(
        self, state: object, questions: dict[str, Any], *, request_model: str | None = None
    ) -> dict[str, Any]:
        return evaluate_typesafe(self, state, questions, request_model=request_model)


def _remote_model_for(client: TypesafeClient, request_model: str | None) -> str:
    """Map local aliases to a TypeSafe model id."""
    if not request_model or request_model in {"jev", "typesafe", "typesafe-jev", "jev-official"}:
        return client.remote_model
    # Allow passthrough of official names if the caller sent one via a future alias.
    return request_model


def load_typesafe_client(settings: Settings) -> tuple[TypesafeClient, int | None, str]:
    api_key = (settings.typesafe_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Create a key at console.typesafe.ai "
            "and export TYPESAFE_API_KEY to use model=jev."
        )
    base = (settings.typesafe_base_url or "https://api.typesafe.ai").strip().rstrip("/")
    remote = (settings.typesafe_remote_model or "jev-latest").strip() or "jev-latest"
    client = TypesafeClient(
        base_url=base,
        api_key=api_key,
        remote_model=remote,
        timeout=float(settings.typesafe_timeout),
    )
    _probe(client)
    backbone = f"{client.base_url} → {client.remote_model}"
    logger.info("TypeSafe Jev client ready: %s", backbone)
    return client, None, backbone


def _probe(client: TypesafeClient) -> None:
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=min(client.timeout, 30.0)) as http:
            response = http.get(f"{client.base_url}/v1/models", headers=headers)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot reach TypeSafe at {client.base_url}: {exc}") from exc
    if response.status_code == 401:
        raise RuntimeError("TypeSafe rejected TYPESAFE_API_KEY (HTTP 401)")
    if response.status_code >= 400:
        raise RuntimeError(
            f"TypeSafe /v1/models returned HTTP {response.status_code}: {response.text[:200]}"
        )


def _normalize_answer(qid: str, raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"TypeSafe answer for {qid} is not an object")
    qtype = str(raw.get("type", "")).strip().lower()
    if qtype == "noul":
        return {"type": "noul", "noul": float(raw["noul"])}
    if qtype == "choice":
        probs = {str(key): float(value) for key, value in dict(raw["probabilities"]).items()}
        return {
            "type": "choice",
            "choice": str(raw["choice"]),
            "probabilities": probs,
            "confidence": float(raw.get("confidence", 0.0)),
        }
    if qtype == "score":
        legend_raw = raw.get("legend") or {}
        legend = {
            str(key): (value if isinstance(value, str) else str(value))
            for key, value in dict(legend_raw).items()
        }
        probs = {str(key): float(value) for key, value in dict(raw["probabilities"]).items()}
        return {
            "type": "score",
            "score": float(raw["score"]),
            "legend": legend,
            "probabilities": probs,
            "confidence": float(raw.get("confidence", 0.0)),
        }
    raise RuntimeError(f"Unknown TypeSafe answer type for {qid}: {qtype!r}")


def evaluate_typesafe(
    client: TypesafeClient,
    state: object,
    questions: dict[str, Any],
    *,
    request_model: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": _remote_model_for(client, request_model),
        "state": state,
        "questions": questions,
    }
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=client.timeout) as http:
            response = http.post(
                f"{client.base_url}/v1/systemone", json=payload, headers=headers
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"TypeSafe request failed: {exc}") from exc

    if response.status_code == 401:
        raise RuntimeError("TypeSafe rejected TYPESAFE_API_KEY (HTTP 401)")
    if response.status_code >= 400:
        raise RuntimeError(
            f"TypeSafe returned HTTP {response.status_code}: {response.text[:400]}"
        )

    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("TypeSafe returned a non-object JSON body")
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, dict) or not raw_answers:
        raise RuntimeError("TypeSafe response is missing answers")
    answers = {str(qid): _normalize_answer(str(qid), ans) for qid, ans in raw_answers.items()}
    usage_raw = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {
        "answers": answers,
        "usage": {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        },
        "upstream_model": str(body.get("model") or payload["model"]),
    }
