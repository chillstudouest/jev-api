from __future__ import annotations


def test_decide_requires_auth(client, fixtures) -> None:
    response = client.post("/v1/decide", json=fixtures["bathroom_quote"])
    assert response.status_code == 401


def test_decide_rejects_bad_key(client, fixtures) -> None:
    response = client.post(
        "/v1/decide",
        json=fixtures["bathroom_quote"],
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_model_requires_auth(client) -> None:
    assert client.get("/v1/model").status_code == 401


def test_health_and_ready_are_public(client) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
