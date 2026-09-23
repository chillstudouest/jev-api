"""Official Laya Python runtime (CPU-capable) behind the System One contract.

laya-mlx is Apple Silicon / MLX only and is not used here. This wrapper loads
the official `laya` package (PyTorch + Transformers) on the configured device.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from jev_api.config import Settings

logger = logging.getLogger("jev_api.laya")


class LayaAgent(Protocol):
    def system_one(self, state: object, questions: dict[str, object]) -> dict[str, object]: ...


def load_laya_agent(settings: Settings) -> tuple[LayaAgent, int | None, str]:
    cache = str(settings.model_cache_dir)
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache
    os.environ.setdefault("USE_TF", "0")
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

    import laya

    repo = settings.laya_hf_repo
    subfolder = settings.laya_subfolder or None
    logger.info("Loading Laya repo=%s subfolder=%s device=%s cache=%s", repo, subfolder, settings.device, cache)
    agent = laya.load(repo, device=settings.device, token=settings.hf_token, subfolder=subfolder)

    params: int | None = None
    model = getattr(agent, "model", None)
    if model is not None and hasattr(model, "parameters"):
        params = sum(p.numel() for p in model.parameters())

    backbone = repo if not subfolder else f"{repo}/{subfolder}"
    if params is not None:
        logger.info("Laya ready: %.1fM parameters on %s", params / 1e6, settings.device)
    else:
        logger.info("Laya ready on %s", settings.device)
    return agent, params, backbone


def laya_system_one(agent: LayaAgent, state: object, questions: dict[str, object]) -> dict[str, object]:
    result = agent.system_one(state, questions)
    if not isinstance(result, dict):
        raise RuntimeError("Laya returned a non-object result")
    return result
