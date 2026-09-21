"""Verdict 2.0 engine: download checkpoint once, load once, infer exactly like upstream."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from jev_api.config import Settings
from jev_api.model import CorrectnessHead, VerdictModel, apply_temperature
from jev_api.preprocess import Item, build_decision_item, collate
from jev_api.schemas import DecideRequest, DecideResponse

logger = logging.getLogger("jev_api.engine")


@dataclass
class EngineStatus:
    ready: bool = False
    loading: bool = False
    error: str | None = None
    parameters: int | None = None
    backbone: str = ""
    device: str = "cpu"
    checkpoint_dir: str | None = None


class VerdictEngine:
    """Thread-safe single-model inference engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = EngineStatus(device=settings.device)
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.model: VerdictModel | None = None
        self.tokenizer: Any = None
        self.pad_id: int = 0

    def start_background_load(self) -> None:
        thread = threading.Thread(target=self._safe_load, name="jev-model-load", daemon=True)
        thread.start()

    def _safe_load(self) -> None:
        with self._load_lock:
            if self.status.ready or self.status.loading:
                return
            self.status.loading = True
            self.status.error = None
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001 — surface any load failure via /ready
                logger.exception("Model load failed")
                self.status.error = str(exc)
                self.status.ready = False
            finally:
                self.status.loading = False

    def ensure_checkpoint(self) -> Path:
        if self.settings.local_checkpoint_dir is not None:
            path = Path(self.settings.local_checkpoint_dir)
            if not path.exists():
                raise FileNotFoundError(f"LOCAL_CHECKPOINT_DIR not found: {path}")
            return path

        cache = Path(self.settings.model_cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading checkpoint %s into %s", self.settings.hf_repo, cache)
        local_dir = snapshot_download(
            repo_id=self.settings.hf_repo,
            revision=self.settings.hf_revision,
            token=self.settings.hf_token,
            cache_dir=str(cache),
            local_dir=str(cache / "snapshot"),
        )
        return Path(local_dir)

    def _resolve_checkpoint_file(self, checkpoint_dir: Path) -> Path:
        direct = checkpoint_dir / self.settings.checkpoint_filename
        if direct.is_file():
            return direct
        # Common layouts from the HF repo / upstream artifacts
        candidates = [
            checkpoint_dir / "artifacts" / "verdict2" / "model.pt",
            checkpoint_dir / "artifacts" / "verdict2-base" / "model.pt",
            checkpoint_dir / "model.pt",
        ]
        for path in candidates:
            if path.is_file():
                return path
        matches = list(checkpoint_dir.rglob("model.pt"))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Could not find {self.settings.checkpoint_filename} under {checkpoint_dir}"
        )

    def load(self) -> None:
        settings = self.settings
        device = settings.device
        backbone_dir = Path(settings.backbone_dir)
        if not backbone_dir.exists():
            raise FileNotFoundError(
                f"Vendored ModernBERT dir missing: {backbone_dir}. "
                "Expected config.json + tokenizer files."
            )

        checkpoint_dir = self.ensure_checkpoint()
        ckpt_path = self._resolve_checkpoint_file(checkpoint_dir)
        logger.info("Loading tokenizer from %s", backbone_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(backbone_dir), local_files_only=True)
        pad_id = tokenizer.pad_token_id or 0

        logger.info("Loading checkpoint %s on %s", ckpt_path, device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        backbone_name = checkpoint.get("backbone", str(backbone_dir))

        model = VerdictModel(str(backbone_dir)).to(device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading state_dict: %s", missing[:20])
        if unexpected:
            logger.warning("Unexpected keys when loading state_dict: %s", unexpected[:20])
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        self.model = model
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.status.parameters = params
        self.status.backbone = backbone_name
        self.status.device = device
        self.status.checkpoint_dir = str(checkpoint_dir)
        self.status.ready = True
        self.status.error = None
        logger.info("Model ready: %.1fM parameters on %s", params / 1e6, device)

    @torch.inference_mode()
    def decide(self, request: DecideRequest) -> DecideResponse:
        if not self.status.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError(self.status.error or "Model is not ready")

        started = time.perf_counter()
        item = build_decision_item(
            self.tokenizer,
            qtype_raw=request.type,
            question=request.question,
            state=request.state,
            options=[o.model_dump() for o in request.options],
            case_id=request.case_id,
            workflow=request.workflow,
            qid=request.qid,
        )
        result = self._infer_item(item)
        latency_ms = (time.perf_counter() - started) * 1000.0
        qtype_name = {0: "choice", 1: "score", 2: "noul"}[item.qtype]
        return DecideResponse(
            choice=result["choice"],
            scores=result["scores"],
            confidence=result["confidence"],
            latency_ms=round(latency_ms, 3),
            qtype=qtype_name,  # type: ignore[arg-type]
            expected_level=result["expected_level"],
            label_index=result["label_index"],
            option_keys=list(item.option_keys),
            model=self.settings.model_name,
        )

    def _infer_item(self, item: Item) -> dict[str, Any]:
        assert self.model is not None
        with self._lock:
            batch = collate([item], self.pad_id)
            device_batch = {k: v.to(self.status.device) for k, v in batch.items()}
            logits = self.model.option_logits(device_batch)
            k = device_batch["marker_mask"].sum(-1)
            logits = apply_temperature(logits, device_batch["qtype"], k, self.model.temperature)
            probs = torch.softmax(logits, dim=-1)
            feats = CorrectnessHead.features(probs, device_batch["marker_mask"], device_batch["qtype"])
            confidence = torch.sigmoid(self.model.correctness(feats))

            width = len(item.markers)
            p = probs[0, :width].float().cpu()
            p = p / max(float(p.sum()), 1e-9)
            conf = float(confidence[0].cpu())
            expected = float(sum(i * float(p[i]) for i in range(width)))
            label_index = int(torch.argmax(p).item())
            scores = {key: float(p[i]) for i, key in enumerate(item.option_keys)}
            choice = item.option_keys[label_index]
            return {
                "choice": choice,
                "scores": scores,
                "confidence": conf,
                "expected_level": expected,
                "label_index": label_index,
                "probs": [float(x) for x in p.tolist()],
            }
