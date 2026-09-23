"""Unit tests for the official TypeSafe Jev client (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jev_api.config import Settings
from jev_api.typesafe_runtime import TypesafeClient, evaluate_typesafe, load_typesafe_client


def test_load_typesafe_requires_api_key(tmp_path) -> None:
    settings = Settings(
        jev_api_key="x",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
        typesafe_api_key=None,
    )
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        load_typesafe_client(settings)


def test_evaluate_typesafe_normalizes_answers() -> None:
    client = TypesafeClient(
        base_url="https://api.typesafe.ai",
        api_key="apikey_test",
        remote_model="jev-latest",
        timeout=30.0,
    )
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "urgency": {"type": "noul", "noul": 0.88},
            "route": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.7, "technical": 0.3},
                "confidence": 0.4,
            },
        },
        "usage": {"input_tokens": 42, "output_tokens": 3},
    }

    with patch("jev_api.typesafe_runtime.httpx.Client") as mock_client_cls:
        mock_http = MagicMock()
        mock_http.__enter__.return_value = mock_http
        mock_http.__exit__.return_value = False
        mock_http.post.return_value = fake_response
        mock_client_cls.return_value = mock_http

        result = evaluate_typesafe(
            client,
            "charged twice",
            {"urgency": {"type": "noul", "instructions": "Urgent?"}},
            request_model="jev",
        )

    assert result["upstream_model"] == "jev-1.13.0"
    assert result["answers"]["urgency"] == {"type": "noul", "noul": 0.88}
    assert result["answers"]["route"]["choice"] == "billing"
    assert result["usage"]["input_tokens"] == 42
    mock_http.post.assert_called_once()
    call_kwargs = mock_http.post.call_args
    assert call_kwargs.args[0] == "https://api.typesafe.ai/v1/systemone"
    assert call_kwargs.kwargs["json"]["model"] == "jev-latest"
