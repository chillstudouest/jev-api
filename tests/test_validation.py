from __future__ import annotations


def test_decide_rejects_single_option(client, auth_headers) -> None:
    payload = {
        "type": "workflow",
        "question": "What next?",
        "state": "state",
        "options": [{"id": "only", "description": "one"}],
    }
    response = client.post("/v1/decide", json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_decide_rejects_duplicate_option_ids(client, auth_headers) -> None:
    payload = {
        "type": "choice",
        "question": "What next?",
        "state": "state",
        "options": [
            {"id": "a", "description": "A"},
            {"id": "a", "description": "A again"},
        ],
    }
    response = client.post("/v1/decide", json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_decide_rejects_empty_question(client, auth_headers, fixtures) -> None:
    payload = {**fixtures["bathroom_quote"], "question": ""}
    response = client.post("/v1/decide", json=payload, headers=auth_headers)
    assert response.status_code == 422
