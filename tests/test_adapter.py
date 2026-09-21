"""Adapter unit tests: Jev answer shapes from OpenJev inference records."""

from __future__ import annotations

from jev_api.adapter import to_jev_answer
from jev_api.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion


def test_choice_adapter() -> None:
    q = ChoiceQuestion(
        type="choice",
        instructions="Which?",
        criteria={"a": "A", "b": "B", "c": "C"},
    )
    answer = to_jev_answer(
        q,
        {
            "choice": "b",
            "option_keys": ["a", "b", "c"],
            "probs": [0.03, 0.91, 0.06],
            "confidence": 0.94,
        },
    )
    assert answer.model_dump() == {
        "type": "choice",
        "choice": "b",
        "probabilities": {"a": 0.03, "b": 0.91, "c": 0.06},
        "confidence": 0.94,
    }


def test_score_adapter_weighted_mean() -> None:
    q = ScoreQuestion(
        type="score",
        instructions="Severity?",
        criteria=["Low", "Medium", "High", "Critical"],
    )
    answer = to_jev_answer(
        q,
        {
            "choice": "3",
            "option_keys": ["0", "1", "2", "3"],
            "probs": [0.0, 0.0, 0.32, 0.68],
            "confidence": 0.88,
        },
    )
    data = answer.model_dump()
    assert data["type"] == "score"
    assert abs(data["score"] - 2.68) < 1e-9
    assert data["legend"]["2"] == "High"
    assert "confidence" in data


def test_noul_adapter_no_confidence() -> None:
    q = NoulQuestion(
        type="noul",
        instructions="Urgent?",
        criteria={"true": "yes", "false": "no"},
    )
    answer = to_jev_answer(
        q,
        {
            "choice": "true",
            "option_keys": ["false", "true"],
            "probs": [0.07, 0.93],
            "confidence": 0.99,
        },
    )
    data = answer.model_dump()
    assert data == {"type": "noul", "noul": 0.93}
