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


@pytest.mark.parametrize("name", ["agent-jev", "agentjev", "agent-jev-0.6b"])
def test_agentjev_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "agent-jev"


@pytest.mark.parametrize("name", ["semif", "semif-latest", "semif-phase1", "openjev"])
def test_semif_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "semif"


@pytest.mark.parametrize("name", ["djev", "djev-latest", "djev-0.1", "diffusion-gemma", "diffusiongemma"])
def test_djev_aliases(name: str) -> None:
    assert resolve_request_model(name) == name
    assert resolve_engine(name) == "djev"


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


def test_engine_dispatches_agentjev_without_weights(tmp_path) -> None:
    from jev_api.agentjev_runtime import agentjev_answers_to_systemone, questions_to_agentjev
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
    engine.agentjev_status.ready = True

    class StubAgentJev:
        def evaluate(self, payload: dict[str, object]) -> dict[str, object]:
            assert isinstance(payload["questions"], list)
            return {
                "results": [
                    {
                        "id": "0",
                        "answers": [
                            {
                                "id": "route",
                                "type": "choice",
                                "value": "billing",
                                "margin": 0.5,
                                "distribution": {"billing": 0.75, "technical": 0.25},
                            }
                        ],
                    }
                ],
                "usage": {"input_path_tokens": 88, "generated_tokens": 0},
            }

    engine._agentjev = StubAgentJev()
    response = engine.systemone(
        SystemOneRequest(
            state="charged twice",
            model="agent-jev",
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "refunds", "technical": "bugs"},
                }
            },
        )
    )
    assert response.model == "agent-jev"
    route = response.answers["route"]
    assert isinstance(route, ChoiceAnswer)
    assert route.choice == "billing"
    assert response.usage.input_tokens == 88
    mapped = questions_to_agentjev(
        {
            "urgent": {
                "type": "noul",
                "instructions": "Urgent?",
                "criteria": {"true": "now", "false": "later"},
            }
        }
    )
    assert mapped[0]["type"] == "boolean"
    converted = agentjev_answers_to_systemone(
        [{"id": "urgent", "type": "boolean", "probability": 0.81, "value": True}]
    )
    assert converted["urgent"] == {"type": "noul", "noul": 0.81}


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


def test_engine_dispatches_djev_without_remote(tmp_path) -> None:
    from jev_api.config import Settings
    from jev_api.engine import VerdictEngine
    from jev_api.schemas import ChoiceAnswer, SystemOneRequest

    settings = Settings(
        jev_api_key="test",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
        djev_base_url="https://api.djev.dev",
    )
    engine = VerdictEngine(settings)
    engine.von_status.ready = True
    engine.djev_status.ready = True

    class StubDjev:
        def evaluate(self, state: object, questions: dict[str, object]) -> dict[str, object]:
            assert state == "checkout down"
            assert "team" in questions
            return {
                "answers": {
                    "team": {
                        "type": "choice",
                        "choice": "engineering",
                        "probabilities": {"billing": 0.1, "engineering": 0.9},
                        "confidence": 0.53,
                    }
                },
                "usage": {"input_tokens": 240, "output_tokens": 8},
            }

    engine._djev = StubDjev()
    response = engine.systemone(
        SystemOneRequest(
            state="checkout down",
            model="djev",
            questions={
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "invoices", "engineering": "outages"},
                }
            },
        )
    )
    assert response.model == "djev"
    team = response.answers["team"]
    assert isinstance(team, ChoiceAnswer)
    assert team.choice == "engineering"
    assert response.usage.input_tokens == 240


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
