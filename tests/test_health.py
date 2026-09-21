from __future__ import annotations


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_model_loaded(client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ready"
