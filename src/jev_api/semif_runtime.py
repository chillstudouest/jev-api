"""SemIf (TheoLeeCJ/SemIf) behind the Jev System One contract.

Official SemIf is a CLI/jsonl scorer: option-list rows, next-token logits.
Torch loading is CUDA/MPS only (Qwen3.5-4B BF16 ~8 GiB). On CPU this wrapper
uses SemIf's llama.cpp GGUF path (~3 GiB Q4) and does not pip-pin SemIf's
torch/transformers versions.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jev_api.config import Settings

logger = logging.getLogger("jev_api.semif")

SEMIF_GIT = "https://github.com/TheoLeeCJ/SemIf.git"
SEMIF_DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEMIF_MAX_OPTIONS = 16


@dataclass
class SemIfRuntime:
    backend: Literal["torch", "llamacpp"]
    model: Any
    tokenizer: Any
    metadata: dict[str, Any]
    score_direct: Any
    score_shared: Any
    mode: str
    max_tokens: int

    def evaluate(self, state: object, questions: dict[str, Any]) -> dict[str, Any]:
        return evaluate_semif(self, state, questions)


def _ensure_src_on_path(settings: Settings) -> Path:
    candidates: list[Path] = []
    if settings.semif_src:
        candidates.append(Path(settings.semif_src))
    candidates.append(Path("/opt/semif"))
    cache_clone = Path(settings.model_cache_dir) / "src" / "semif"
    candidates.append(cache_clone)

    def _activate(root: Path) -> Path | None:
        src = root / "src"
        marker = src / "semif_phase1" / "core.py"
        if not marker.is_file():
            marker = root / "semif_phase1" / "core.py"
            src = root
        if not marker.is_file():
            return None
        resolved = str(src)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        return root

    for path in candidates:
        found = _activate(path)
        if found is not None:
            return found

    cache_clone.parent.mkdir(parents=True, exist_ok=True)
    if not (cache_clone / "src" / "semif_phase1" / "core.py").is_file():
        logger.info("Cloning SemIf source into %s", cache_clone)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", SEMIF_GIT, str(cache_clone)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    found = _activate(cache_clone)
    if found is None:
        raise RuntimeError(f"SemIf clone at {cache_clone} is missing src/semif_phase1")
    return found


def _option_description(key: str, value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return key


def questions_to_semif(state: object, questions: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, raw in questions.items():
        if not isinstance(raw, dict):
            raise ValueError(f"questions.{qid} must be an object")
        qtype = str(raw.get("type", "")).strip().lower()
        question = raw.get("instructions")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"questions.{qid} instructions must be a nonempty string")

        if qtype == "noul":
            criteria = raw.get("criteria")
            true_desc = "Yes"
            false_desc = "No"
            if isinstance(criteria, dict):
                true_desc = _option_description("true", criteria.get("true"))
                false_desc = _option_description("false", criteria.get("false"))
            options = [
                {"id": "true", "description": true_desc},
                {"id": "false", "description": false_desc},
            ]
        elif qtype == "choice":
            criteria = raw.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"questions.{qid} choice criteria must be an object")
            options = [
                {"id": str(key), "description": _option_description(str(key), value)}
                for key, value in criteria.items()
            ]
        elif qtype == "score":
            levels = raw.get("criteria")
            if not isinstance(levels, list) or len(levels) < 2:
                raise ValueError(f"questions.{qid} score criteria must be a list")
            options = [
                {"id": str(index), "description": str(label)}
                for index, label in enumerate(levels)
            ]
        else:
            raise ValueError(f"Unsupported question type for SemIf: {qtype!r}")

        if len(options) > SEMIF_MAX_OPTIONS:
            raise ValueError(
                f"questions.{qid} has {len(options)} options; SemIf supports at most {SEMIF_MAX_OPTIONS}"
            )
        rows.append(
            {
                "id": qid,
                "state": state,
                "question": question,
                "options": options,
                "_jev_type": qtype,
            }
        )
    return rows


def _distribution(option_ids: list[str], probabilities: list[float]) -> dict[str, float]:
    if len(option_ids) != len(probabilities):
        raise RuntimeError("SemIf returned mismatched option_ids and probabilities")
    return {str(option_id): float(prob) for option_id, prob in zip(option_ids, probabilities)}


def _confidence(probabilities: list[float]) -> float:
    ranked = sorted((float(value) for value in probabilities), reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return ranked[0]
    return ranked[0] - ranked[1]


def semif_results_to_systemone(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_id = {str(item["id"]): item for item in results}
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["id"])
        result = by_id.get(qid)
        if result is None:
            raise RuntimeError(f"SemIf response is missing decision {qid}")
        qtype = str(row["_jev_type"])
        dist = _distribution(list(result["option_ids"]), list(result["probabilities"]))
        if qtype == "noul":
            mapped[qid] = {"type": "noul", "noul": float(dist.get("true", 0.0))}
        elif qtype == "choice":
            choice = max(dist, key=dist.get)
            mapped[qid] = {
                "type": "choice",
                "choice": choice,
                "probabilities": dist,
                "confidence": _confidence(list(dist.values())),
            }
        elif qtype == "score":
            legend = {option["id"]: option["description"] for option in row["options"]}
            score = sum(int(key) * value for key, value in dist.items())
            mapped[qid] = {
                "type": "score",
                "score": float(score),
                "legend": legend,
                "probabilities": dist,
                "confidence": _confidence(list(dist.values())),
            }
        else:
            raise ValueError(f"Unknown SemIf question type: {qtype!r}")
    return mapped


def _pick_backend(settings: Settings) -> Literal["torch", "llamacpp"]:
    requested = (settings.semif_backend or "auto").strip().lower()
    if requested in {"llamacpp", "llama.cpp", "gguf", "cpu"}:
        return "llamacpp"
    if requested == "torch":
        return "torch"
    if requested not in {"auto", ""}:
        raise ValueError(f"Unknown SEMIF_BACKEND: {settings.semif_backend!r}")

    device = settings.device.strip().lower()
    if device == "cpu":
        return "llamacpp"
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "torch"
        except ImportError:
            pass
    if device == "mps":
        return "torch"
    try:
        import torch

        if torch.cuda.is_available():
            return "torch"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "torch"
    except ImportError:
        pass
    return "llamacpp"


def _prepare_gguf(settings: Settings) -> Path:
    if settings.semif_gguf_path:
        local = Path(settings.semif_gguf_path)
        if not local.is_file():
            raise RuntimeError(f"SEMIF_GGUF_PATH is not a file: {local}")
        return local

    from huggingface_hub import hf_hub_download

    cache = Path(settings.model_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading SemIf GGUF %s/%s",
        settings.semif_gguf_repo,
        settings.semif_gguf_file,
    )
    return Path(
        hf_hub_download(
            settings.semif_gguf_repo,
            settings.semif_gguf_file,
            cache_dir=str(cache),
            token=settings.hf_token,
        )
    )


def load_semif_runtime(settings: Settings) -> tuple[SemIfRuntime, int | None, str]:
    _ensure_src_on_path(settings)
    backend = _pick_backend(settings)
    revision = settings.semif_revision or SEMIF_DEFAULT_REVISION
    mode = (settings.semif_mode or "auto").strip().lower()
    if mode not in {"auto", "direct", "shared"}:
        raise ValueError(f"Unknown SEMIF_MODE: {settings.semif_mode!r}")

    if backend == "torch":
        from semif_phase1.core import load_causal_model
        from semif_phase1.direct import score as score_direct
        from semif_phase1.shared import score_shared

        device = settings.device if settings.device in {"cuda", "mps"} else "auto"
        logger.info(
            "Loading SemIf torch model=%s revision=%s device=%s",
            settings.semif_hf_repo,
            revision,
            device,
        )
        try:
            model, tokenizer, metadata = load_causal_model(
                settings.semif_hf_repo,
                revision,
                device,
                settings.semif_dtype,
            )
        except ValueError as exc:
            raise RuntimeError(
                "SemIf torch backend needs CUDA or Apple MPS (Qwen3.5-4B BF16 ~8 GiB). "
                "On the CPU VPS set SEMIF_BACKEND=llamacpp and install llama-cpp-python "
                "(pip install -e '.[semif]')."
            ) from exc
        params: int | None = None
        if hasattr(model, "parameters"):
            params = sum(p.numel() for p in model.parameters())
        backbone = f"{settings.semif_hf_repo}@{revision[:12]}"
    else:
        from semif_phase1 import llamacpp_backend

        gguf = _prepare_gguf(settings)
        logger.info(
            "Loading SemIf llama.cpp gguf=%s tokenizer=%s revision=%s",
            gguf,
            settings.semif_hf_repo,
            revision,
        )
        try:
            model, tokenizer, metadata = llamacpp_backend.load_model(
                settings.semif_hf_repo,
                revision,
                gguf,
                threads=settings.semif_llama_threads,
                context_tokens=settings.semif_max_tokens,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SemIf CPU path needs llama-cpp-python. Install with: pip install -e '.[semif]'"
            ) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "llama.cpp extra" in message or "llama-cpp-python" in message:
                raise RuntimeError(
                    "SemIf CPU path needs llama-cpp-python. Install with: pip install -e '.[semif]'"
                ) from exc
            raise
        score_direct = llamacpp_backend.score
        score_shared = llamacpp_backend.score_shared
        params = None
        backbone = f"{settings.semif_gguf_repo}/{settings.semif_gguf_file}"

    runtime = SemIfRuntime(
        backend=backend,
        model=model,
        tokenizer=tokenizer,
        metadata=metadata,
        score_direct=score_direct,
        score_shared=score_shared,
        mode=mode,
        max_tokens=settings.semif_max_tokens,
    )
    logger.info("SemIf ready backend=%s backbone=%s", backend, backbone)
    return runtime, params, backbone


def evaluate_semif(runtime: SemIfRuntime, state: object, questions: dict[str, Any]) -> dict[str, Any]:
    rows = questions_to_semif(state, questions)
    use_shared = runtime.mode == "shared" or (runtime.mode == "auto" and len(rows) > 1)
    input_tokens = 0
    if use_shared:
        results, _timing = runtime.score_shared(
            runtime.model,
            runtime.tokenizer,
            [{key: value for key, value in row.items() if key != "_jev_type"} for row in rows],
            runtime.metadata,
            runtime.max_tokens,
        )
        input_tokens = sum(int(item.get("input_tokens", 0) or 0) for item in results)
    else:
        results = []
        for row in rows:
            payload = {key: value for key, value in row.items() if key != "_jev_type"}
            result = runtime.score_direct(
                runtime.model,
                runtime.tokenizer,
                payload,
                runtime.metadata,
                runtime.max_tokens,
            )
            results.append(result)
            input_tokens += int(result.get("input_tokens", 0) or 0)
    return {
        "answers": semif_results_to_systemone(rows, results),
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
