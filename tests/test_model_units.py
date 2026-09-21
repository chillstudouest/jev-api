"""Model unit tests that do not require the Hugging Face checkpoint."""

from __future__ import annotations

import torch

from jev_api.engine import EngineStatus, VerdictEngine
from jev_api.model import CorrectnessHead, apply_temperature, cardinality_bucket


def test_cardinality_bucket() -> None:
    assert cardinality_bucket(2) == 0
    assert cardinality_bucket(9) == 7
    assert cardinality_bucket(100) == 7


def test_apply_temperature_scales_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    qtype = torch.tensor([0])
    k = torch.tensor([3])
    temperature = torch.ones(3, 8) * 2.0
    scaled = apply_temperature(logits, qtype, k, temperature)
    assert torch.allclose(scaled, logits / 2.0)


def test_correctness_features_shape() -> None:
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    mask = torch.tensor([[True, True, True]])
    qtype = torch.tensor([0])
    feats = CorrectnessHead.features(probs, mask, qtype)
    assert feats.shape == (1, 7)


def test_engine_starts_not_ready(tmp_path) -> None:
    from jev_api.config import Settings

    settings = Settings(
        jev_api_key="x",
        download_on_startup=False,
        model_cache_dir=tmp_path,
        backbone_dir=tmp_path,
    )
    engine = VerdictEngine(settings)
    assert isinstance(engine.status, EngineStatus)
    assert engine.status.ready is False
