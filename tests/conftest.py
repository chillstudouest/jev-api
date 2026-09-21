"""Shared pytest fixtures — FakeEngine returns Jev-shaped answers without weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_api.adapter import to_jev_answer
from jev_api.config import Settings
from jev_api.engine import EngineStatus, VerdictEngine
from jev_api.main import create_app
from jev_api.schemas import (
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    parse_questions,
    resolve_request_model,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "jev"


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

    def systemone(self, request: SystemOneRequest) -> SystemOneResponse:
        model_name = resolve_request_model(request.model)
        questions = parse_questions(request.questions)
        answers = {}
        input_tokens = 0
        for qid, question in questions.items():
            # Deterministic fake OpenJev record → Jev answer via the real adapter.
            if question.type == "choice":
                keys = list(question.criteria.keys())
            elif question.type == "score":
                keys = [str(i) for i in range(len(question.criteria))]
            else:
                keys = ["false", "true"]
            n = len(keys)
            raw = [float(n - i) for i in range(n)]
            total = sum(raw)
            probs = [r / total for r in raw]
            # For noul fixtures that expect high "true", bias true when instructions contain "urgent"
            if question.type == "noul" and "urgent" in question.instructions.lower():
                probs = [0.07, 0.93]
            infer = {
                "choice": keys[int(max(range(n), key=lambda i: probs[i]))],
                "option_keys": keys,
                "probs": probs,
                "confidence": 0.91,
                "expected_level": sum(i * probs[i] for i in range(n)),
                "label_index": int(max(range(n), key=lambda i: probs[i])),
            }
            answers[qid] = to_jev_answer(question, infer)
            input_tokens += 40 + n * 10
        return SystemOneResponse(
            model=model_name,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )


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


@pytest.fixture
def jev_fixtures() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in FIXTURES_DIR.glob("*.json"):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out
