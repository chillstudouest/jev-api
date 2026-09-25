"""Unit tests for the official TypeSafe Jev client (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jev_api.config import Settings
from jev_api.typesafe_runtime import call_typesafe_jev


def test_call_typesafe_requires_api_key(tmp_path) -> None:
    settings = Settings(
        jev_api_key="x",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
        typesafe_api_key="",
    )
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        call_typesafe_jev(settings, "state", {"q": {"type": "noul", "instructions": "Urgent?"}})


def test_call_typesafe_maps_answers(tmp_path) -> None:
    settings = Settings(
        jev_api_key="x",
        download_on_startup=False,
        model_cache_dir=tmp_path / "models",
        typesafe_api_key="apikey_test",
        typesafe_model="jev-1.13.0",
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

    with patch("jev_api.typesafe_runtime.httpx.post", return_value=fake_response) as mock_post:
        result = call_typesafe_jev(
            settings,
            "charged twice",
            {"urgency": {"type": "noul", "instructions": "Urgent?"}},
        )

    assert result["model"] == "jev-1.13.0"
    assert result["answers"]["urgency"].noul == 0.88
    assert result["answers"]["route"].choice == "billing"
    assert result["usage"].input_tokens == 42
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "https://api.typesafe.ai/v1/systemone"
    assert mock_post.call_args.kwargs["json"]["model"] == "jev-1.13.0"
