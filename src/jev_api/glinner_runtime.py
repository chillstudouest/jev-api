"""GLiNER2.5-Decide behind the Jev System One contract.

Named `glinner` in this API: Fastino GLiNER2.5-Decide. Every question becomes
a classification head of the same forward pass, whatever the question count.
choice / score / noul map onto single-label classification over the supplied
criteria.

Two backends, picked with `GLINNER_BACKEND`:
- `torch`: the `gliner2` library on PyTorch.
- `onnx`: onnxruntime on an ONNX export (`GLINNER_ONNX_VARIANT` fp32 or int8).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_api.config import Settings

logger = logging.getLogger("jev_api.glinner")

DEFAULT_GLINNER_REPO = "fastino/GLiNER2.5-Decide"


# (state text, heads) -> probabilities keyed by head task, then by option id.
Scorer = Callable[[str, list["_Head"]], dict[str, dict[str, float]]]


@dataclass
class GlinerRuntime:
    scorer: Scorer
    backbone: str
    device: str
    backend: str = "torch"

    def evaluate(self, state: object, questions: dict[str, Any]) -> dict[str, Any]:
        return evaluate_glinner(self, state, questions)


def _flatten(value: object, path: str) -> list[str]:
    if isinstance(value, dict):
        return [line for key, item in value.items() for line in _flatten(item, f"{path} {key}".strip())]
    if isinstance(value, list):
        return [line for item in value for line in _flatten(item, path)]
    return [f"{path}: {value}" if path else str(value)]


def _state_text(state: object) -> str:
    # Decide reads prose far better than JSON: flatten structured state into "path: value" lines.
    if isinstance(state, str):
        return state
    return "\n".join(_flatten(state, ""))


def _confidence(probabilities: dict[str, float]) -> float:
    ranked = sorted(probabilities.values(), reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return ranked[0]
    return ranked[0] - ranked[1]


def _estimate_tokens(text: str, labels: list[str]) -> int:
    joined = text + " ".join(labels)
    return max(1, len(joined) // 4)


def _resolve_device(requested: str) -> str:
    if requested == "cuda":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if requested == "mps":
        return "mps"
    return "cpu"


def _prepare_cache(settings: Settings) -> str:
    cache = str(settings.model_cache_dir)
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache
    os.environ.setdefault("USE_TF", "0")
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
    return cache


def _torch_scorer(extractor: Any) -> Scorer:
    def score(text: str, heads: list[_Head]) -> dict[str, dict[str, float]]:
        # multi_label + threshold 0 + softmax returns the full single-label distribution.
        tasks = {
            head.task: {
                "labels": dict(head.label_map),
                "multi_label": True,
                "cls_threshold": 0.0,
                "class_act": "softmax",
            }
            for head in heads
        }
        raw = extractor.classify_text(text, tasks, include_confidence=True)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Unexpected GLiNER2 output: {type(raw)!r}")
        return {head.task: _rows_to_probabilities(head, raw.get(head.task)) for head in heads}

    return score


def _onnx_scorer(model: Any) -> Scorer:
    from jev_api.glinner_onnx import OnnxTask

    def score(text: str, heads: list[_Head]) -> dict[str, dict[str, float]]:
        tasks = [OnnxTask(name=head.task, labels=dict(head.label_map)) for head in heads]
        return model.probabilities(text, tasks)

    return score


def _load_torch(settings: Settings, device: str) -> tuple[GlinerRuntime, int | None, str]:
    from gliner2 import AutoExtractor

    repo = settings.glinner_hf_repo
    logger.info("Loading Gliner (torch) repo=%s device=%s", repo, device)
    extractor = AutoExtractor.from_pretrained(repo)
    if device != "cpu":
        extractor = extractor.to(device)
    extractor.eval()
    params = sum(p.numel() for p in extractor.parameters())
    logger.info("Gliner ready: %.1fM parameters on %s", params / 1e6, device)
    runtime = GlinerRuntime(scorer=_torch_scorer(extractor), backbone=repo, device=device, backend="torch")
    return runtime, params, repo


ONNX_FILES = {"fp32": "model.onnx", "int8": "model_int8.onnx"}


def _load_onnx(settings: Settings, cache: str) -> tuple[GlinerRuntime, int | None, str]:
    from huggingface_hub import hf_hub_download

    from jev_api.glinner_onnx import GlinerOnnx

    variant = settings.glinner_onnx_variant
    if variant not in ONNX_FILES:
        raise ValueError(f"GLINNER_ONNX_VARIANT must be one of {sorted(ONNX_FILES)}, got {variant!r}")
    repo = settings.glinner_onnx_repo
    revision = settings.glinner_onnx_revision
    logger.info("Loading Gliner (onnx %s) repo=%s@%s", variant, repo, revision)
    model_path = hf_hub_download(repo, ONNX_FILES[variant], revision=revision, cache_dir=cache)
    tokenizer_path = hf_hub_download(repo, "tokenizer.json", revision=revision, cache_dir=cache)
    model = GlinerOnnx(Path(model_path), Path(tokenizer_path), threads=settings.glinner_threads)
    backbone = f"{repo}:{variant}"
    runtime = GlinerRuntime(scorer=_onnx_scorer(model), backbone=backbone, device="cpu", backend=f"onnx-{variant}")
    logger.info("Gliner ready: onnx %s on cpu", variant)
    return runtime, None, backbone


def load_glinner_runtime(settings: Settings) -> tuple[GlinerRuntime, int | None, str]:
    cache = _prepare_cache(settings)
    if settings.glinner_backend == "onnx":
        return _load_onnx(settings, cache)
    if settings.glinner_backend != "torch":
        raise ValueError(f"GLINNER_BACKEND must be torch or onnx, got {settings.glinner_backend!r}")
    return _load_torch(settings, _resolve_device(settings.device))


@dataclass
class _Head:
    qid: str
    qtype: str
    task: str
    label_map: dict[str, str]
    legend: dict[str, str] | None = None


def _build_head(qid: str, raw: object, used_tasks: set[str]) -> _Head:
    if not isinstance(raw, dict):
        raise ValueError(f"questions.{qid} must be an object")
    qtype = str(raw.get("type", "")).strip().lower()
    instructions = raw.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"questions.{qid} instructions must be a nonempty string")

    # GLiNER2 reads the task name as the question; keep it unique per request.
    task = instructions.strip()
    if task in used_tasks:
        # Parentheses are GLiNER2 structure tokens, so never add them here.
        task = f"{task} #{qid}"
    used_tasks.add(task)

    if qtype == "choice":
        criteria = raw.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"questions.{qid} choice criteria must be an object")
        label_map = {
            str(key): value.strip() if isinstance(value, str) and value.strip() else str(key)
            for key, value in criteria.items()
        }
        return _Head(qid=qid, qtype=qtype, task=task, label_map=label_map)

    if qtype == "score":
        levels = raw.get("criteria")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(f"questions.{qid} score criteria must be a list")
        legend = {str(index): str(label) for index, label in enumerate(levels)}
        return _Head(qid=qid, qtype=qtype, task=task, label_map=dict(legend), legend=legend)

    if qtype == "noul":
        criteria = raw.get("criteria")
        true_desc = "Yes"
        false_desc = "No"
        if isinstance(criteria, dict):
            if isinstance(criteria.get("true"), str) and criteria["true"].strip():
                true_desc = criteria["true"].strip()
            if isinstance(criteria.get("false"), str) and criteria["false"].strip():
                false_desc = criteria["false"].strip()
        return _Head(
            qid=qid,
            qtype=qtype,
            task=task,
            label_map={"true": true_desc, "false": false_desc},
        )

    raise ValueError(f"Unsupported question type for glinner: {qtype!r}")


def _rows_to_probabilities(head: _Head, rows: object) -> dict[str, float]:
    probs: dict[str, float] = {key: 0.0 for key in head.label_map}
    if not isinstance(rows, list):
        rows = [rows]
    for item in rows:
        if not isinstance(item, dict):
            continue
        key = str(item.get("label", ""))
        if key in probs:
            probs[key] = float(item.get("confidence", 0.0) or 0.0)
    total = sum(probs.values())
    if total <= 0:
        return {key: 1.0 / len(probs) for key in probs}
    return {key: value / total for key, value in probs.items()}


def evaluate_glinner(runtime: GlinerRuntime, state: object, questions: dict[str, Any]) -> dict[str, Any]:
    state_text = _state_text(state)
    used_tasks: set[str] = set()
    heads = [_build_head(str(qid), raw, used_tasks) for qid, raw in questions.items()]

    scored = runtime.scorer(state_text, heads)

    answers: dict[str, dict[str, Any]] = {}
    for head in heads:
        probs = scored[head.task]
        if head.qtype == "choice":
            answers[head.qid] = {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
                "confidence": _confidence(probs),
            }
        elif head.qtype == "score":
            answers[head.qid] = {
                "type": "score",
                "score": float(sum(int(key) * value for key, value in probs.items())),
                "legend": head.legend or {},
                "probabilities": probs,
                "confidence": _confidence(probs),
            }
        else:
            answers[head.qid] = {"type": "noul", "noul": float(probs.get("true", 0.0))}

    labels = [label for head in heads for label in (head.task, *head.label_map.values())]
    return {
        "answers": answers,
        "usage": {"input_tokens": _estimate_tokens(state_text, labels), "output_tokens": 0},
    }
