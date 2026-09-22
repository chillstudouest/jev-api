"""Shared pytest fixtures — FakeEngine returns Jev-shaped answers without Von weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_api.config import Settings
from jev_api.engine import EngineStatus, VerdictEngine
from jev_api.main import create_app
from jev_api.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
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
        self.von_status = EngineStatus(
            ready=True,
            loading=False,
            parameters=395_000_000,
            backbone="wfzyx/von-1.0",
            device="cpu",
            backend="von",
            name="von",
        )
        self.laya_status = EngineStatus(
            ready=True,
            loading=False,
            parameters=421_000_000,
            backbone="convaiinnovations/laya",
            device="cpu",
            backend="laya",
            name="laya",
        )
        self.agentjev_status = EngineStatus(
            ready=True,
            loading=False,
            parameters=598_000_000,
            backbone="aimeigaoshou/agent-jev",
            device="cpu",
            backend="agent-jev",
            name="agent-jev",
        )
        self.semif_status = EngineStatus(
            ready=True,
            loading=False,
            parameters=4_000_000_000,
            backbone="Qwen/Qwen3.5-4B",
            device="cpu",
            backend="llamacpp",
            name="semif",
        )
        self.djev_status = EngineStatus(
            ready=True,
            loading=False,
            parameters=None,
            backbone="https://api.djev.dev",
            device="remote",
            backend="djev-http",
            name="djev",
        )

    def start_background_load(self) -> None:
        return

    def systemone(self, request: SystemOneRequest) -> SystemOneResponse:
        import time

        started = time.perf_counter()
        model_name = resolve_request_model(request.model)
        questions = parse_questions(request.questions)
        answers = {}
        input_tokens = 0
        for qid, question in questions.items():
            if question.type == "choice":
                keys = list(question.criteria.keys())
                n = len(keys)
                raw = [float(n - i) for i in range(n)]
                total = sum(raw)
                probs = {k: raw[i] / total for i, k in enumerate(keys)}
                choice = max(probs, key=probs.get)  # type: ignore[arg-type]
                sorted_p = sorted(probs.values(), reverse=True)
                conf = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
                answers[qid] = ChoiceAnswer(choice=choice, probabilities=probs, confidence=round(conf, 3))
            elif question.type == "score":
                n = len(question.criteria)
                raw = [float(n - i) for i in range(n)]
                total = sum(raw)
                probs = {str(i): raw[i] / total for i in range(n)}
                score = sum(i * probs[str(i)] for i in range(n))
                legend = {str(i): label for i, label in enumerate(question.criteria)}
                sorted_p = sorted(probs.values(), reverse=True)
                conf = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
                answers[qid] = ScoreAnswer(
                    score=round(score, 2),
                    legend=legend,
                    probabilities=probs,
                    confidence=round(conf, 3),
                )
            else:
                noul = 0.93 if "urgent" in question.instructions.lower() else 0.35
                answers[qid] = NoulAnswer(noul=noul)
            input_tokens += 50
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return SystemOneResponse(
            model=model_name,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=len(answers)),
            duration_ms=duration_ms,
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        jev_api_key="test-secret-key",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
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
