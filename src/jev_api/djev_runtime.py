"""djev (Maisa / DiffusionGemma) behind the Jev System One contract.

This is the JevBench row **djev (Maisa, diffusion-gemma)** — Apache-2.0 stack
https://github.com/Davipar/djev-dev over Google DiffusionGemma-26B-A4B-it.

Inference path (djev-dev defaults):
  enable_thinking=false, diffusion_max_steps=1, read_only=true
  → one-step structured label readout via POST /v1/request

NOT the experimental "djev thinking" full-generation path (JevBench #21).

Local djev-dev (Apache-2.0) needs **no API key** by default. An API key is only
for Maisa's hosted https://api.djev.dev. On the 8 GiB CPU VPS, DiffusionGemma-26B
does not fit — point DJEV_BASE_URL at a self-hosted GPU instance, or use Maisa
with DJEV_API_KEY.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from jev_api.config import Settings

logger = logging.getLogger("jev_api.djev")

# Official remote model id for Maisa / djev-dev structured reads.
DJEV_REMOTE_MODELS = frozenset({"djev", "djev-latest", "djev-0.1"})


@dataclass
class DjevClient:
    """HTTP client for Davipar/djev-dev (or Maisa hosted) POST /v1/request."""

    base_url: str
    api_key: str | None
    remote_model: str
    timeout: float
    prefer_low_latency: bool

    def evaluate(self, state: object, questions: dict[str, Any]) -> dict[str, Any]:
        return evaluate_djev(self, state, questions)


def load_djev_client(settings: Settings) -> tuple[DjevClient, int | None, str]:
    base = (settings.djev_base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "DJEV_BASE_URL is empty. Point it at a self-hosted Davipar/djev-dev "
            "instance (no API key by default), or at https://api.djev.dev with DJEV_API_KEY."
        )
    remote_model = (settings.djev_remote_model or "djev").strip()
    if remote_model not in DJEV_REMOTE_MODELS:
        raise RuntimeError(
            f"DJEV_REMOTE_MODEL must be one of {sorted(DJEV_REMOTE_MODELS)} "
            f"(Maisa / djev-dev structured read). Got {remote_model!r}."
        )
    client = DjevClient(
        base_url=base,
        api_key=(settings.djev_api_key or None),
        remote_model=remote_model,
        timeout=float(settings.djev_timeout),
        prefer_low_latency=bool(settings.djev_prefer_low_latency),
    )
    _probe_ready(client)
    # Parameters stay None — weights live on the remote GPU / Maisa API.
    backbone = f"Davipar/djev-dev → {client.base_url} ({client.remote_model})"
    logger.info("djev (Maisa, diffusion-gemma) client ready: %s", backbone)
    return client, None, backbone


def _probe_ready(client: DjevClient) -> None:
    headers = _auth_headers(client)
    try:
        with httpx.Client(timeout=min(client.timeout, 30.0)) as http:
            response = http.get(f"{client.base_url}/ready", headers=headers)
            if response.status_code == 404:
                response = http.get(f"{client.base_url}/health", headers=headers)
            if response.status_code == 503:
                raise RuntimeError(
                    f"djev at {client.base_url} is not ready (HTTP 503): {response.text[:200]}"
                )
            if response.status_code == 401:
                raise RuntimeError(
                    "djev rejected the API key (HTTP 401). "
                    "Self-hosted djev-dev needs no key unless DJEV_API_KEY is set on that server. "
                    "Hosted api.djev.dev requires DJEV_API_KEY."
                )
            if response.status_code >= 400:
                logger.warning(
                    "djev readiness probe returned HTTP %s; continuing "
                    "(POST /v1/request will validate)",
                    response.status_code,
                )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot reach djev at {client.base_url}: {exc}") from exc


def _auth_headers(client: DjevClient) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
    return headers


def _normalize_answer(qid: str, raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"djev answer for {qid} is not an object")
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
    raise RuntimeError(f"Unknown djev answer type for {qid}: {qtype!r}")


def evaluate_djev(client: DjevClient, state: object, questions: dict[str, Any]) -> dict[str, Any]:
    """Forward System One questions to djev-dev POST /v1/request (structured read)."""
    payload: dict[str, Any] = {
        "model": client.remote_model,
        "state": state,
        "questions": questions,
        # Explicit one-step structured defaults (djev-dev contract; not thinking).
        "options": {"seed": 0, "samples": 1, "steps": 1},
    }
    headers = {
        **_auth_headers(client),
        "Content-Type": "application/json",
        "X-Djev-Operation-Id": str(uuid4()),
    }
    if client.prefer_low_latency:
        headers["Prefer"] = "low-latency"

    try:
        with httpx.Client(timeout=client.timeout) as http:
            response = http.post(f"{client.base_url}/v1/request", json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"djev request failed: {exc}") from exc

    if response.status_code == 401:
        raise RuntimeError(
            "djev rejected credentials (HTTP 401). "
            "Self-hosted Davipar/djev-dev: leave DJEV_API_KEY unset (local default has no auth). "
            "Maisa hosted api.djev.dev: set DJEV_API_KEY from an invitation."
        )
    if response.status_code == 402:
        raise RuntimeError("djev insufficient credits (HTTP 402)")
    if response.status_code >= 400:
        detail = response.text[:400]
        raise RuntimeError(f"djev returned HTTP {response.status_code}: {detail}")

    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("djev returned a non-object JSON body")
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, dict) or not raw_answers:
        raise RuntimeError("djev response is missing answers")
    answers = {str(qid): _normalize_answer(str(qid), ans) for qid, ans in raw_answers.items()}
    usage_raw = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {
        "answers": answers,
        "usage": {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        },
    }
