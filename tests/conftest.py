"""Shared pytest fixtures with a fake engine (no model weights required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_api.config import Settings
from jev_api.engine import EngineStatus, VerdictEngine
from jev_api.main import create_app
from jev_api.schemas import DecideRequest, DecideResponse

FIXTURES = Path(__file__).parent / "fixtures" / "decisions.json"


class FakeEngine(VerdictEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.status = EngineStatus(
            ready=True,
            loading=False,
            parameters=149_601_234,
            backbone="vendor/modernbert",
            device="cpu",
            checkpoint_dir="/tmp/fake",
        )

    def start_background_load(self) -> None:
        return

    def decide(self, request: DecideRequest) -> DecideResponse:
        keys = [o.id for o in request.options] or ["false", "true"]
        n = len(keys)
        # Deterministic decreasing scores that sum to 1 — mirrors renormalized softmax.
        raw = [float(n - i) for i in range(n)]
        total = sum(raw)
        scores = {k: raw[i] / total for i, k in enumerate(keys)}
        choice = keys[0]
        qtype = "noul" if request.type == "noul" else ("score" if request.type == "score" else "choice")
        return DecideResponse(
            choice=choice,
            scores=scores,
            confidence=0.91,
            latency_ms=1.23,
            qtype=qtype,  # type: ignore[arg-type]
            expected_level=sum(i * scores[k] for i, k in enumerate(keys)),
            label_index=0,
            option_keys=keys,
            model=self.settings.model_name,
        )


@pytest.fixture
def fixtures() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        jev_api_key="test-secret-key",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
        backbone_dir=Path(__file__).resolve().parents[1] / "vendor" / "modernbert",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    engine = FakeEngine(settings)
    app = create_app(settings=settings, engine=engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.jev_api_key}"}
