from __future__ import annotations


def _payload() -> dict:
    return {
        "state": "hello",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }


def test_missing_key_returns_403(client) -> None:
    response = client.post("/v1/systemone", json=_payload())
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["error_type"] == "authentication_error"
    assert "API key" in detail["message"]


def test_bad_key_returns_401(client) -> None:
    response = client.post(
        "/v1/systemone",
        json=_payload(),
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["error_type"] == "authentication_error"


def test_health_and_ready_are_public(client) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_model_requires_auth(client) -> None:
    assert client.get("/v1/model").status_code == 403
