from __future__ import annotations


def test_decide_workflow(client, auth_headers, fixtures) -> None:
    response = client.post("/v1/decide", json=fixtures["bathroom_quote"], headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["choice"] in body["scores"]
    assert abs(sum(body["scores"].values()) - 1.0) < 1e-6
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["qtype"] == "choice"
    assert set(body["scores"]) == {"create_client", "create_job", "create_quote"}


def test_decide_different_option_set(client, auth_headers, fixtures) -> None:
    response = client.post("/v1/decide", json=fixtures["support_triage"], headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert set(body["scores"]) == {"billing", "identity", "network"}
    assert body["choice"] == "billing"


def test_decide_score_and_noul(client, auth_headers, fixtures) -> None:
    score = client.post("/v1/decide", json=fixtures["score_severity"], headers=auth_headers)
    assert score.status_code == 200
    assert score.json()["qtype"] == "score"
    assert len(score.json()["scores"]) == 5

    noul = client.post("/v1/decide", json=fixtures["noul_default"], headers=auth_headers)
    assert noul.status_code == 200
    assert noul.json()["qtype"] == "noul"
    assert set(noul.json()["scores"]) == {"false", "true"}


def test_scores_are_coherent_probabilities(client, auth_headers, fixtures) -> None:
    body = client.post("/v1/decide", json=fixtures["bathroom_quote"], headers=auth_headers).json()
    values = list(body["scores"].values())
    assert all(v >= 0 for v in values)
    assert abs(sum(values) - 1.0) < 1e-6
    assert body["choice"] == max(body["scores"], key=body["scores"].get)


def test_model_info(client, auth_headers) -> None:
    response = client.get("/v1/model", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["runtime"] == "pytorch+transformers"
    assert body["parameters"] is not None
