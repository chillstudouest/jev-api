"""AgentJev-0.6B runtime (CPU-capable) behind the Jev System One contract.

Maps noul→boolean, choice→choice, score→score. Uses the official
`jev_service.DecisionEngine` from https://github.com/malevrigns/agent-jev
and weights from https://huggingface.co/aimeigaoshou/agent-jev.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from jev_api.config import Settings

logger = logging.getLogger("jev_api.agentjev")

AGENTJEV_GIT = "https://github.com/malevrigns/agent-jev.git"


def _ensure_src_on_path(settings: Settings) -> Path:
    candidates: list[Path] = []
    if settings.agentjev_src:
        candidates.append(Path(settings.agentjev_src))
    candidates.append(Path("/opt/agent-jev"))
    cache_clone = Path(settings.model_cache_dir) / "src" / "agent-jev"
    candidates.append(cache_clone)

    for path in candidates:
        if (path / "jev_service" / "engine.py").is_file():
            resolved = str(path)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return path

    cache_clone.parent.mkdir(parents=True, exist_ok=True)
    if not (cache_clone / "jev_service" / "engine.py").is_file():
        logger.info("Cloning AgentJev source into %s", cache_clone)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", AGENTJEV_GIT, str(cache_clone)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    sys.path.insert(0, str(cache_clone))
    return cache_clone


def _prepare_checkpoint(settings: Settings) -> tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    cache = Path(settings.model_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    token = settings.hf_token

    backbone_dir = Path(
        snapshot_download(
            settings.agentjev_backbone,
            cache_dir=str(cache),
            token=token,
        )
    )
    weights = Path(
        hf_hub_download(
            settings.agentjev_hf_repo,
            "model.safetensors",
            cache_dir=str(cache),
            token=token,
        )
    )
    temperatures = Path(
        hf_hub_download(
            settings.agentjev_hf_repo,
            "temperatures.json",
            cache_dir=str(cache),
            token=token,
        )
    )
    checkpoint = cache / "agentjev_v1.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_mtime < weights.stat().st_mtime:
        import torch
        from safetensors.torch import load_file

        logger.info("Wrapping AgentJev safetensors into %s", checkpoint)
        torch.save({"state_dict": load_file(str(weights))}, checkpoint)
    return checkpoint, backbone_dir, temperatures


def load_agentjev_engine(settings: Settings) -> tuple[Any, int | None, str]:
    _ensure_src_on_path(settings)
    checkpoint, backbone_dir, temperatures = _prepare_checkpoint(settings)

    from jev_service.engine import DecisionEngine

    device = settings.device if settings.device != "cuda" else "cuda:0"
    logger.info(
        "Loading AgentJev checkpoint=%s backbone=%s device=%s",
        checkpoint,
        backbone_dir,
        device,
    )
    engine = DecisionEngine(
        checkpoint=str(checkpoint),
        model_path=str(backbone_dir),
        device=device,
        temperatures=str(temperatures),
    )
    params: int | None = None
    model = getattr(engine, "model", None)
    if model is not None and hasattr(model, "parameters"):
        params = sum(p.numel() for p in model.parameters())
    backbone = f"{settings.agentjev_hf_repo} + {settings.agentjev_backbone}"
    if params is not None:
        logger.info("AgentJev ready: %.1fM parameters on %s", params / 1e6, device)
    else:
        logger.info("AgentJev ready on %s", device)
    return engine, params, backbone


def questions_to_agentjev(questions: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, raw in questions.items():
        if not isinstance(raw, dict):
            raise ValueError(f"questions.{qid} must be an object")
        qtype = str(raw.get("type", "")).strip().lower()
        instructions = raw.get("instructions")
        if qtype == "noul":
            row: dict[str, Any] = {"id": qid, "type": "boolean", "question": instructions}
            criteria = raw.get("criteria")
            if isinstance(criteria, dict) and criteria:
                row["criteria"] = {
                    key: (value if isinstance(value, str) and value.strip() else key)
                    for key, value in criteria.items()
                }
            rows.append(row)
        elif qtype == "choice":
            criteria = raw.get("criteria")
            if not isinstance(criteria, dict):
                raise ValueError(f"questions.{qid} choice criteria must be an object")
            options = {
                key: (value if isinstance(value, str) and value.strip() else key)
                for key, value in criteria.items()
            }
            rows.append(
                {"id": qid, "type": "choice", "question": instructions, "options": options}
            )
        elif qtype == "score":
            levels = raw.get("criteria")
            if not isinstance(levels, list):
                raise ValueError(f"questions.{qid} score criteria must be a list")
            rows.append(
                {"id": qid, "type": "score", "question": instructions, "levels": list(levels)}
            )
        else:
            raise ValueError(f"Unsupported question type for AgentJev: {qtype!r}")
    return rows


def agentjev_answers_to_systemone(answers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for ans in answers:
        qid = str(ans["id"])
        kind = ans["type"]
        if kind == "boolean":
            mapped[qid] = {"type": "noul", "noul": float(ans["probability"])}
        elif kind == "choice":
            dist = {str(key): float(value) for key, value in dict(ans["distribution"]).items()}
            ranked = sorted(dist.values(), reverse=True)
            margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
            mapped[qid] = {
                "type": "choice",
                "choice": str(ans["value"]),
                "probabilities": dist,
                "confidence": float(ans.get("margin", margin)),
            }
        elif kind == "score":
            legend_raw = ans.get("legend") or []
            legend = {str(index): str(label) for index, label in enumerate(legend_raw)}
            dist = {str(key): float(value) for key, value in dict(ans["distribution"]).items()}
            ranked = sorted(dist.values(), reverse=True)
            conf = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
            mapped[qid] = {
                "type": "score",
                "score": float(ans["score"]),
                "legend": legend,
                "probabilities": dist,
                "confidence": float(conf),
            }
        else:
            raise ValueError(f"Unknown AgentJev answer type: {kind!r}")
    return mapped


def evaluate_agentjev(engine: Any, state: object, questions: dict[str, Any]) -> dict[str, Any]:
    payload = {"state": state, "questions": questions_to_agentjev(questions)}
    raw = engine.evaluate(payload)
    if not isinstance(raw, dict):
        raise RuntimeError("AgentJev returned a non-object result")
    results = raw.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("AgentJev response is missing results")
    first = results[0]
    answers = first.get("answers") if isinstance(first, dict) else None
    if not isinstance(answers, list):
        raise RuntimeError("AgentJev response is missing answers")
    usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return {
        "answers": agentjev_answers_to_systemone(answers),
        "usage": {
            "input_tokens": int(usage_raw.get("input_path_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("generated_tokens", 0) or 0),
        },
    }
