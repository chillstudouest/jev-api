from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from jev_api.autojev_runtime import NOT_CONFIGURED_MESSAGE, evaluate_autojev
from jev_api.config import Settings
from jev_api.engine import VerdictEngine
from jev_api.main import create_app
from jev_api.schemas import ApiUsageError, SystemOneRequest


def _choice_request() -> SystemOneRequest:
    return SystemOneRequest(
        state="charged twice",
        model="autojev",
        questions={
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "refunds", "technical": "bugs"},
            }
        },
    )


def test_evaluate_autojev_posts_systemone(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> httpx.Response:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return httpx.Response(
            200,
            json={
                "model": "autojev-qwen3.8-27b",
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.75, "technical": 0.25},
                        "confidence": 0.5,
                    }
                },
                "usage": {"input_tokens": 40, "output_tokens": 0},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("jev_api.autojev_runtime.httpx.post", fake_post)
    settings = Settings(
        autojev_base_url="http://gpu.example:8000/",
        autojev_api_key="upstream-secret",
        autojev_timeout=33.0,
    )
    body = evaluate_autojev(settings, _choice_request())
    assert captured["url"] == "http://gpu.example:8000/v1/systemone"
    assert captured["json"]["model"] == "autojev"
    assert captured["json"]["state"] == "charged twice"
    assert captured["headers"]["Authorization"] == "Bearer upstream-secret"
    assert captured["timeout"] == 33.0
    assert body["answers"]["route"]["choice"] == "billing"


def test_evaluate_autojev_maps_busy(monkeypatch) -> None:
    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(529, json={"detail": "busy"}, request=httpx.Request("POST", url))

    monkeypatch.setattr("jev_api.autojev_runtime.httpx.post", fake_post)
    settings = Settings(autojev_base_url="http://gpu.example:8000")
    with pytest.raises(RuntimeError, match="busy"):
        evaluate_autojev(settings, _choice_request())


def test_evaluate_autojev_maps_usage_error(monkeypatch) -> None:
    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": {"message": "Unknown model. Use autojev-qwen3.8-27b or jev-latest."}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("jev_api.autojev_runtime.httpx.post", fake_post)
    settings = Settings(autojev_base_url="http://gpu.example:8000")
    with pytest.raises(ApiUsageError, match="Unknown model"):
        evaluate_autojev(settings, _choice_request())


def test_http_autojev_unconfigured_503(tmp_path) -> None:
    settings = Settings(
        jev_api_key="test-secret-key",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
        autojev_base_url="",
    )
    engine = VerdictEngine(settings)
    engine.von_status.ready = True
    app = create_app(settings=settings, engine=engine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer test-secret-key"},
            json={
                "state": "x",
                "model": "autojev",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which?",
                        "criteria": {"a": "A", "b": "B"},
                    }
                },
            },
        )
    assert response.status_code == 503
    assert "AUTOJEV_BASE_URL" in response.json()["detail"]["message"]
    assert NOT_CONFIGURED_MESSAGE in response.json()["detail"]["message"]
