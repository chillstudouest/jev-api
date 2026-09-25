from __future__ import annotations

import pytest

from jev_api.engine import _time_inference, answers_from_mapping
from jev_api.schemas import ApiUsageError, NoulAnswer, resolve_engine, resolve_request_model


def test_default_model_is_von() -> None:
    assert resolve_request_model(None) == "von"
    assert resolve_engine(None) == "von"


@pytest.mark.parametrize("name", ["von", "von-latest", "jev-latest", "jev-1.13.0"])
def test_von_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "von"


@pytest.mark.parametrize("name", ["jev", "jev-official", "typesafe", "typesafe-jev"])
def test_jev_official_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "jev"


@pytest.mark.parametrize("name", ["laya", "laya-latest", "laya-1.0"])
def test_laya_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "laya"




@pytest.mark.parametrize("name", ["semif", "semif-latest", "semif-phase1", "openjev"])
def test_semif_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "semif"




@pytest.mark.parametrize("name", ["glinner", "glinner-latest", "gliclass", "gliner-class"])
def test_glinner_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "glinner"


def test_time_inference_reports_wall_and_optional_gpu() -> None:
    result, inference_ms, gpu_ms = _time_inference(lambda: 42)
    assert result == 42
    assert inference_ms >= 0
    try:
        import torch
    except ImportError:
        assert gpu_ms is None
        return
    if torch.cuda.is_available():
        assert gpu_ms is not None and gpu_ms >= 0
    else:
        assert gpu_ms is None


def test_unknown_model() -> None:
    with pytest.raises(ApiUsageError, match="Unknown model"):
        resolve_request_model("jev-9.9.9")


def test_answers_from_mapping_strips_noul_confidence() -> None:
    answers = answers_from_mapping(
        {
            "urgency": {
                "type": "noul",
                "noul": 0.81,
                "confidence": 0.81,
                "action": {"act_probability": 1.0},
            }
        },
        source="Laya",
    )
    dumped = answers["urgency"].model_dump()
    assert dumped == {"type": "noul", "noul": 0.81}


def test_engine_dispatches_laya_without_von_weights(tmp_path) -> None:
    from jev_api.config import Settings
    from jev_api.engine import VerdictEngine
    from jev_api.schemas import SystemOneRequest

    settings = Settings(
        jev_api_key="test",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
    )
    engine = VerdictEngine(settings)
    engine.von_status.ready = True
    engine.laya_status.ready = True

    class StubLaya:
        def system_one(self, state: object, questions: dict[str, object]) -> dict[str, object]:
            assert state == "charged twice"
            return {
                "answers": {
                    "urgent": {"type": "noul", "noul": 0.42, "confidence": 0.58},
                },
                "usage": {"input_tokens": 12, "output_tokens": 0},
            }

    engine._laya = StubLaya()
    response = engine.systemone(
        SystemOneRequest(
            state="charged twice",
            model="laya",
            questions={"urgent": {"type": "noul", "instructions": "Urgent?"}},
        )
    )
    assert response.model == "laya"
    urgent = response.answers["urgent"]
    assert isinstance(urgent, NoulAnswer)
    assert urgent.noul == 0.42
    assert response.usage.input_tokens == 12




def test_engine_dispatches_semif_without_weights(tmp_path) -> None:
    from jev_api.config import Settings
    from jev_api.engine import VerdictEngine
    from jev_api.schemas import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest
    from jev_api.semif_runtime import questions_to_semif, semif_results_to_systemone

    settings = Settings(
        jev_api_key="test",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
    )
    engine = VerdictEngine(settings)
    engine.von_status.ready = True
    engine.semif_status.ready = True

    class StubSemIf:
        def evaluate(self, state: object, questions: dict[str, object]) -> dict[str, object]:
            assert state == "charged twice"
            assert "route" in questions
            return {
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.8, "technical": 0.2},
                        "confidence": 0.6,
                    }
                },
                "usage": {"input_tokens": 64, "output_tokens": 0},
            }

    engine._semif = StubSemIf()
    response = engine.systemone(
        SystemOneRequest(
            state="charged twice",
            model="semif",
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "refunds", "technical": "bugs"},
                }
            },
        )
    )
    assert response.model == "semif"
    route = response.answers["route"]
    assert isinstance(route, ChoiceAnswer)
    assert route.choice == "billing"
    assert response.usage.input_tokens == 64

    rows = questions_to_semif(
        "state",
        {
            "urgent": {
                "type": "noul",
                "instructions": "Urgent?",
                "criteria": {"true": "now", "false": "later"},
            },
            "severity": {
                "type": "score",
                "instructions": "How bad?",
                "criteria": ["Low", "High"],
            },
        },
    )
    assert rows[0]["options"] == [
        {"id": "true", "description": "now"},
        {"id": "false", "description": "later"},
    ]
    assert rows[1]["options"][1] == {"id": "1", "description": "High"}
    converted = semif_results_to_systemone(
        rows,
        [
            {
                "id": "urgent",
                "option_ids": ["true", "false"],
                "probabilities": [0.81, 0.19],
            },
            {
                "id": "severity",
                "option_ids": ["0", "1"],
                "probabilities": [0.25, 0.75],
            },
        ],
    )
    assert converted["urgent"] == {"type": "noul", "noul": 0.81}
    assert converted["severity"]["type"] == "score"
    assert converted["severity"]["score"] == 0.75
    assert isinstance(ScoreAnswer.model_validate(converted["severity"]), ScoreAnswer)
    assert isinstance(NoulAnswer.model_validate(converted["urgent"]), NoulAnswer)




def test_engine_dispatches_glinner_without_weights(tmp_path) -> None:
    from jev_api.config import Settings
    from jev_api.engine import VerdictEngine
    from jev_api.schemas import ChoiceAnswer, SystemOneRequest

    settings = Settings(
        jev_api_key="test",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
    )
    engine = VerdictEngine(settings)
    engine.von_status.ready = True
    engine.glinner_status.ready = True

    class StubGliner:
        def evaluate(self, state: object, questions: dict[str, object]) -> dict[str, object]:
            assert state == "charged twice"
            return {
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.7, "technical": 0.3},
                        "confidence": 0.4,
                    }
                },
                "usage": {"input_tokens": 40, "output_tokens": 0},
            }

    engine._glinner = StubGliner()
    response = engine.systemone(
        SystemOneRequest(
            state="charged twice",
            model="glinner",
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "refunds", "technical": "bugs"},
                }
            },
        )
    )
    assert response.model == "glinner"
    route = response.answers["route"]
    assert isinstance(route, ChoiceAnswer)
    assert route.choice == "billing"
    assert response.usage.input_tokens == 40
    assert response.inference_ms >= 0


def test_engine_dispatches_jev_to_typesafe(tmp_path, monkeypatch) -> None:
    from jev_api.config import Settings
    from jev_api.engine import VerdictEngine
    from jev_api.schemas import ChoiceAnswer, NoulAnswer, SystemOneRequest, Usage

    settings = Settings(
        jev_api_key="test",
        typesafe_api_key="ts-key",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
    )
    engine = VerdictEngine(settings)

    def fake_call(_settings: Settings, state: object, questions: dict[str, object]) -> dict[str, object]:
        assert state == "charged twice"
        assert "route" in questions
        return {
            "model": "jev-1.13.0",
            "answers": {
                "route": ChoiceAnswer(
                    choice="billing",
                    probabilities={"billing": 0.8, "technical": 0.2},
                    confidence=0.6,
                )
            },
            "usage": Usage(input_tokens=12, output_tokens=0),
            "upstream_duration_ms": 90.0,
            "upstream_inference_ms": 80.0,
            "gpu_duration_ms": None,
        }

    monkeypatch.setattr("jev_api.typesafe_runtime.call_typesafe_jev", fake_call)
    response = engine.systemone(
        SystemOneRequest(
            state="charged twice",
            model="jev",
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "refunds", "technical": "bugs"},
                }
            },
        )
    )
    assert response.model == "jev"
    assert response.timings is not None
    assert response.timings.engine == "jev"
    route = response.answers["route"]
    assert isinstance(route, ChoiceAnswer)
    assert route.choice == "billing"


def test_answers_from_mapping_choice_and_score() -> None:
    answers = answers_from_mapping(
        {
            "route": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.7, "technical": 0.3},
                "confidence": 0.4,
            },
            "severity": {
                "type": "score",
                "score": 1.2,
                "legend": {"0": "Low", "1": "High"},
                "probabilities": {"0": 0.8, "1": 0.2},
                "confidence": 0.6,
            },
        },
        source="Laya",
    )
    assert answers["route"].choice == "billing"
    assert answers["severity"].score == 1.2
