"""GLiClass (GLiNER-family) behind the Jev System One contract.

Named `glinner` in this API: Knowledgator GLiClass zero-shot classifier,
inspired by GLiNER's single-forward-pass label conditioning. Maps choice /
score / noul onto single-label classification over the supplied criteria.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from jev_api.config import Settings

logger = logging.getLogger("jev_api.glinner")

DEFAULT_GLINNER_REPO = "knowledgator/gliclass-base-v1.0"


@dataclass
class GlinerRuntime:
    pipeline: Any
    backbone: str
    device: str

    def evaluate(self, state: object, questions: dict[str, Any]) -> dict[str, Any]:
        return evaluate_glinner(self, state, questions)


def _state_text(state: object) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(state)


def _label_text(key: str, description: object | None) -> str:
    if isinstance(description, str) and description.strip():
        return f"{key}: {description.strip()}"
    return str(key)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    import math

    # Stabilize; GLiClass multi-label scores are independent sigmoids.
    peak = max(scores.values())
    exps = {key: math.exp(value - peak) for key, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


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


def _classify(
    runtime: GlinerRuntime,
    text: str,
    label_map: dict[str, str],
    prompt: str,
) -> dict[str, float]:
    """Return a probability distribution keyed by original option ids."""
    labels = list(label_map.values())
    reverse = {label: key for key, label in label_map.items()}
    raw = runtime.pipeline(
        text,
        labels,
        threshold=0.0,
        # multi-label returns a score per option; we softmax into a System One dist.
        classification_type="multi-label",
        prompt=prompt,
        batch_size=1,
    )
    # Pipeline returns list-of-lists for batch; one text → first element.
    rows = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected GLiClass output: {type(raw)!r}")

    scores: dict[str, float] = {key: 0.0 for key in label_map}
    for item in rows:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", ""))
        score = float(item.get("score", 0.0) or 0.0)
        key = reverse.get(label)
        if key is not None:
            scores[key] = score

    # Independent sigmoids → softmax so choice/score/noul stay calibrated-ish.
    return _softmax(scores)


def load_glinner_runtime(settings: Settings) -> tuple[GlinerRuntime, int | None, str]:
    cache = str(settings.model_cache_dir)
    os.makedirs(cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache
    os.environ.setdefault("USE_TF", "0")
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    repo = settings.glinner_hf_repo
    device = settings.device
    if device == "cuda":
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    elif device == "mps":
        device = "mps"
    else:
        device = "cpu"

    logger.info("Loading Gliner (GLiClass) repo=%s device=%s cache=%s", repo, device, cache)
    model = GLiClassModel.from_pretrained(repo, cache_dir=cache)
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache)
    pipeline = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="single-label",
        device=device,
        progress_bar=False,
        max_length=settings.glinner_max_length,
    )

    params: int | None = None
    if hasattr(model, "parameters"):
        params = sum(p.numel() for p in model.parameters())

    runtime = GlinerRuntime(pipeline=pipeline, backbone=repo, device=device)
    if params is not None:
        logger.info("Gliner ready: %.1fM parameters on %s", params / 1e6, device)
    else:
        logger.info("Gliner ready on %s", device)
    return runtime, params, repo


def evaluate_glinner(runtime: GlinerRuntime, state: object, questions: dict[str, Any]) -> dict[str, Any]:
    state_text = _state_text(state)
    answers: dict[str, dict[str, Any]] = {}
    input_tokens = 0

    for qid, raw in questions.items():
        if not isinstance(raw, dict):
            raise ValueError(f"questions.{qid} must be an object")
        qtype = str(raw.get("type", "")).strip().lower()
        instructions = raw.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"questions.{qid} instructions must be a nonempty string")

        prompt = instructions.strip()
        text = f"{state_text}\n\nQuestion: {prompt}"

        if qtype == "choice":
            criteria = raw.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"questions.{qid} choice criteria must be an object")
            label_map = {
                str(key): _label_text(str(key), value) for key, value in criteria.items()
            }
            probs = _classify(runtime, text, label_map, prompt)
            choice = max(probs, key=probs.get)
            answers[qid] = {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": _confidence(probs),
            }
            input_tokens += _estimate_tokens(text, list(label_map.values()))

        elif qtype == "score":
            levels = raw.get("criteria")
            if not isinstance(levels, list) or len(levels) < 2:
                raise ValueError(f"questions.{qid} score criteria must be a list")
            label_map = {
                str(index): _label_text(str(index), label)
                for index, label in enumerate(levels)
            }
            probs = _classify(runtime, text, label_map, prompt)
            legend = {str(index): str(label) for index, label in enumerate(levels)}
            score = sum(int(key) * value for key, value in probs.items())
            answers[qid] = {
                "type": "score",
                "score": float(score),
                "legend": legend,
                "probabilities": probs,
                "confidence": _confidence(probs),
            }
            input_tokens += _estimate_tokens(text, list(label_map.values()))

        elif qtype == "noul":
            criteria = raw.get("criteria")
            true_desc = "Yes"
            false_desc = "No"
            if isinstance(criteria, dict):
                if isinstance(criteria.get("true"), str) and criteria["true"].strip():
                    true_desc = criteria["true"].strip()
                if isinstance(criteria.get("false"), str) and criteria["false"].strip():
                    false_desc = criteria["false"].strip()
            label_map = {
                "true": _label_text("true", true_desc),
                "false": _label_text("false", false_desc),
            }
            probs = _classify(runtime, text, label_map, prompt)
            answers[qid] = {"type": "noul", "noul": float(probs.get("true", 0.0))}
            input_tokens += _estimate_tokens(text, list(label_map.values()))

        else:
            raise ValueError(f"Unsupported question type for glinner: {qtype!r}")

    return {
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
