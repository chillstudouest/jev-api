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


@pytest.mark.parametrize("name", ["glinner", "glinner-latest", "decide", "gliner2-decide"])
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
        source="Gliner",
    )
    dumped = answers["urgency"].model_dump()
    assert dumped == {"type": "noul", "noul": 0.81}


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
        source="Gliner",
    )
    assert answers["route"].choice == "billing"
    assert answers["severity"].score == 1.2


def test_gliner_torch_scorer_maps_heads_to_system_one() -> None:
    from jev_api.glinner_runtime import GlinerRuntime, _torch_scorer

    class StubExtractor:
        def classify_text(
            self, text: str, tasks: dict[str, dict[str, object]], include_confidence: bool
        ) -> dict[str, list[dict[str, object]]]:
            assert text == "charged twice"
            assert include_confidence
            assert list(tasks) == ["Which team?", "Urgent?", "Which team? #again"]
            return {
                "Which team?": [
                    {"label": "billing", "confidence": 0.8},
                    {"label": "technical", "confidence": 0.2},
                ],
                "Urgent?": [
                    {"label": "true", "confidence": 0.3},
                    {"label": "false", "confidence": 0.7},
                ],
                "Which team? #again": [
                    {"label": "0", "confidence": 0.25},
                    {"label": "1", "confidence": 0.75},
                ],
            }

    runtime = GlinerRuntime(
        scorer=_torch_scorer(StubExtractor()), backbone="fastino/GLiNER2.5-Decide", device="cpu"
    )
    raw = runtime.evaluate(
        "charged twice",
        {
            "route": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "refunds", "technical": "bugs"},
            },
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "again": {"type": "score", "instructions": "Which team?", "criteria": ["Low", "High"]},
        },
    )
    answers = raw["answers"]
    assert answers["route"]["choice"] == "billing"
    assert answers["urgent"]["noul"] == 0.3
    assert answers["again"]["score"] == 0.75
    assert answers["again"]["legend"] == {"0": "Low", "1": "High"}


def test_gliner_onnx_encode_places_label_markers() -> None:
    from jev_api.glinner_onnx import GlinerOnnx, OnnxTask

    class StubEncoding:
        def __init__(self, piece: str) -> None:
            self.ids = [len(piece)]

    class StubTokenizer:
        def encode(self, piece: str, add_special_tokens: bool) -> StubEncoding:
            assert not add_special_tokens
            return StubEncoding(piece)

    model = GlinerOnnx.__new__(GlinerOnnx)
    model.tokenizer = StubTokenizer()
    model._cache = {}
    ids, positions = model.encode(
        "Refund me",
        [OnnxTask("Which team?", {"billing": "refunds", "technical": None}), OnnxTask("Urgent?", {"true": None, "false": None})],
    )
    # ( [P] prompt ( [L] billing [L] technical ) ) [SEP_STRUCT] ( [P] prompt ( [L] true [L] false ) ) [SEP_TEXT] refund me .
    assert positions == [4, 6, 15, 17]
    assert len(ids) == 25
