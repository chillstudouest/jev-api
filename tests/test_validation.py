from __future__ import annotations


def test_unknown_question_type_400(client, auth_headers) -> None:
    payload = {
        "state": "x",
        "questions": {
            "bad": {"type": "boolean", "instructions": "?", "criteria": {"a": "A", "b": "B"}}
        },
    }
    response = client.post("/v1/systemone", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert response.json()["detail"]["error_type"] == "api_usage_error"


def test_unknown_model_400(client, auth_headers) -> None:
    payload = {
        "state": "x",
        "model": "jev-9.9.9",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }
    response = client.post("/v1/systemone", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]["message"]


def test_missing_state_422(client, auth_headers) -> None:
    payload = {
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "A", "b": "B"},
            }
        }
    }
    response = client.post("/v1/systemone", json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_empty_questions_422(client, auth_headers) -> None:
    response = client.post(
        "/v1/systemone",
        json={"state": "x", "questions": {}},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_choice_missing_criteria_422(client, auth_headers) -> None:
    payload = {
        "state": "x",
        "questions": {"route": {"type": "choice", "instructions": "Which?"}},
    }
    response = client.post("/v1/systemone", json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_empty_state_422(client, auth_headers) -> None:
    payload = {
        "state": "",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }
    response = client.post("/v1/systemone", json=payload, headers=auth_headers)
    assert response.status_code == 422
