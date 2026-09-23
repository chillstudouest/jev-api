"""Jev System One compatibility / golden shape tests."""

from __future__ import annotations

import pytest


def _post(client, auth_headers, body: dict):
    return client.post("/v1/systemone", json=body, headers=auth_headers)


def test_choice_two_options_shape(client, auth_headers, jev_fixtures) -> None:
    fx = jev_fixtures["choice_two_options"]
    response = _post(client, auth_headers, fx["request"])
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "jev-1.13.0"  # fixture still sends the Jev alias
    answer = body["answers"]["route"]
    assert answer["type"] == "choice"
    assert answer["choice"] in fx["constraints"]["route_choice_in"]
    assert set(answer["probabilities"]) == {"billing", "technical"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6
    assert 0.0 <= answer["confidence"] <= 1.0
    assert "usage" in body and body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] >= 0
    assert isinstance(body["duration_ms"], (int, float))
    assert body["duration_ms"] >= 0


def test_choice_three_options(client, auth_headers, jev_fixtures) -> None:
    fx = jev_fixtures["choice_three_options"]
    body = _post(client, auth_headers, fx["request"]).json()
    answer = body["answers"]["next"]
    assert answer["type"] == "choice"
    assert len(answer["probabilities"]) == 3
    assert answer["choice"] in answer["probabilities"]


def test_choice_many_options(client, auth_headers) -> None:
    criteria = {f"opt_{i}": f"Option {i}" for i in range(8)}
    payload = {
        "state": "Pick one.",
        "questions": {
            "pick": {"type": "choice", "instructions": "Which option?", "criteria": criteria}
        },
    }
    answer = _post(client, auth_headers, payload).json()["answers"]["pick"]
    assert len(answer["probabilities"]) == 8
    assert answer["choice"] in criteria


def test_choice_null_criteria_description(client, auth_headers) -> None:
    payload = {
        "state": "state",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": None, "b": "Beta"},
            }
        },
    }
    response = _post(client, auth_headers, payload)
    assert response.status_code == 200
    assert set(response.json()["answers"]["route"]["probabilities"]) == {"a", "b"}


def test_score_weighted_mean_and_legend(client, auth_headers, jev_fixtures) -> None:
    fx = jev_fixtures["score_four_levels"]
    body = _post(client, auth_headers, fx["request"]).json()
    assert body["model"] == "jev-latest"
    answer = body["answers"]["severity"]
    assert answer["type"] == "score"
    assert set(answer["legend"]) == {"0", "1", "2", "3"}
    assert answer["legend"]["0"] == "Low"
    assert set(answer["probabilities"]) == {"0", "1", "2", "3"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6
    # score is weighted mean of indices, not 0..1
    expected = sum(int(k) * v for k, v in answer["probabilities"].items())
    assert abs(answer["score"] - expected) < 1e-6
    assert 0.0 <= answer["score"] <= 3.0
    assert "confidence" in answer


def test_score_two_levels(client, auth_headers) -> None:
    payload = {
        "state": "mild issue",
        "questions": {
            "s": {
                "type": "score",
                "instructions": "Severity?",
                "criteria": ["ok", "bad"],
            }
        },
    }
    answer = _post(client, auth_headers, payload).json()["answers"]["s"]
    assert 0.0 <= answer["score"] <= 1.0
    assert answer["legend"] == {"0": "ok", "1": "bad"}


def test_noul_shape_no_confidence(client, auth_headers, jev_fixtures) -> None:
    fx = jev_fixtures["noul_urgency"]
    answer = _post(client, auth_headers, fx["request"]).json()["answers"]["urgency"]
    assert answer["type"] == "noul"
    assert "noul" in answer
    assert 0.0 <= answer["noul"] <= 1.0
    assert "confidence" not in answer
    assert set(answer.keys()) == {"type", "noul"}


def test_noul_default_criteria(client, auth_headers) -> None:
    payload = {
        "state": "maybe",
        "questions": {
            "q": {"type": "noul", "instructions": "Is this about billing?"}
        },
    }
    answer = _post(client, auth_headers, payload).json()["answers"]["q"]
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0


def test_multi_question_same_keys(client, auth_headers, jev_fixtures) -> None:
    fx = jev_fixtures["multi_choice_score_noul"]
    body = _post(client, auth_headers, fx["request"]).json()
    assert body["model"] == "jev-preview"
    assert set(body["answers"]) == {"route", "urgency", "severity"}
    assert body["answers"]["route"]["type"] == "choice"
    assert body["answers"]["urgency"]["type"] == "noul"
    assert body["answers"]["severity"]["type"] == "score"
    assert "confidence" not in body["answers"]["urgency"]


def test_api_v1_alias(client, auth_headers, jev_fixtures) -> None:
    fx = jev_fixtures["choice_two_options"]
    response = client.post("/api/v1/systemone", json=fx["request"], headers=auth_headers)
    assert response.status_code == 200
    assert "route" in response.json()["answers"]




def test_semif_model_switch(client, auth_headers) -> None:
    payload = {
        "state": "Charged twice for September and cancelling Friday unless refunded.",
        "model": "semif",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
            }
        },
    }
    body = _post(client, auth_headers, payload).json()
    assert body["model"] == "semif"
    assert body["answers"]["route"]["choice"] in {"billing", "technical"}


def test_autojev_model_switch(client, auth_headers) -> None:
    payload = {
        "state": "Charged twice for September and cancelling Friday unless refunded.",
        "model": "autojev",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
            }
        },
    }
    body = _post(client, auth_headers, payload).json()
    assert body["model"] == "autojev"
    assert body["answers"]["route"]["choice"] in {"billing", "technical"}


def test_laya_model_switch(client, auth_headers) -> None:
    payload = {
        "state": "Charged twice for September and cancelling Friday unless refunded.",
        "model": "laya",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
            }
        },
    }
    body = _post(client, auth_headers, payload).json()
    assert body["model"] == "laya"
    assert body["answers"]["route"]["type"] == "choice"
    assert body["answers"]["route"]["choice"] in {"billing", "technical"}


def test_default_model_is_von(client, auth_headers) -> None:
    payload = {
        "state": "x",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }
    body = _post(client, auth_headers, payload).json()
    assert body["model"] == "von"


def test_model_info(client, auth_headers) -> None:
    body = client.get("/v1/model", headers=auth_headers).json()
    assert body["protocol"] == "jev-systemone"
    assert body["engine"] == "von-option-marker-395m"
    assert "von" in body["accepted_models"]
    assert "laya" in body["accepted_models"]
    assert "semif" in body["accepted_models"]
    assert "autojev" in body["accepted_models"]
    assert body["extras"]["default_model"] == "von"
    assert "semif" in body["extras"]
    assert "autojev" in body["extras"]
    assert body["extras"]["autojev"]["configured"] is False


@pytest.mark.parametrize(
    "model",
    ["von", "laya", "semif", "autojev", "autojev-27b", "jev-latest", "jev-preview", "jev-1.13.0", "von-option-marker", None],
)
def test_accepted_model_aliases(client, auth_headers, model) -> None:
    payload = {
        "state": "x",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }
    if model is not None:
        payload["model"] = model
    body = _post(client, auth_headers, payload).json()
    assert body["model"] == (model or "von")
