from __future__ import annotations

import pytest

from jev_api.engine import answers_from_mapping
from jev_api.schemas import ApiUsageError, NoulAnswer, resolve_engine, resolve_request_model


def test_default_model_is_von() -> None:
    assert resolve_request_model(None) == "von"
    assert resolve_engine(None) == "von"


@pytest.mark.parametrize("name", ["von", "von-latest", "jev-latest", "jev-1.13.0"])
def test_von_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "von"


@pytest.mark.parametrize("name", ["laya", "laya-latest", "laya-1.0"])
def test_laya_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "laya"


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
