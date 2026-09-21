"""Verdict 2.0 engine: download once, load once, batch-infer Jev System One requests."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from jev_api.adapter import build_jev_item, to_jev_answer
from jev_api.config import Settings
from jev_api.model import CorrectnessHead, VerdictModel, apply_temperature
from jev_api.preprocess import Item, collate
from jev_api.schemas import (
    Question,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    parse_questions,
    resolve_request_model,
)

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
            except Exception as exc:  # noqa: BLE001
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

    def systemone(self, request: SystemOneRequest) -> SystemOneResponse:
        if not self.status.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError(self.status.error or "Model is not ready")

        model_name = resolve_request_model(request.model)
        questions = parse_questions(request.questions)
        items: list[Item] = []
        qids: list[str] = []
        qdefs: list[Question] = []
        for qid, question in questions.items():
            item = build_jev_item(self.tokenizer, request.state, qid, question)
            items.append(item)
            qids.append(qid)
            qdefs.append(question)

        records = self._infer_batch(items)
        answers = {
            qid: to_jev_answer(qdef, record)
            for qid, qdef, record in zip(qids, qdefs, records, strict=True)
        }
        input_tokens = sum(len(it.ids) for it in items)
        return SystemOneResponse(
            model=model_name,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )

    @torch.inference_mode()
    def _infer_batch(self, items: list[Item]) -> list[dict[str, Any]]:
        assert self.model is not None
        with self._lock:
            batch = collate(items, self.pad_id)
            device_batch = {k: v.to(self.status.device) for k, v in batch.items()}
            logits = self.model.option_logits(device_batch)
            k = device_batch["marker_mask"].sum(-1)
            logits = apply_temperature(logits, device_batch["qtype"], k, self.model.temperature)
            probs = torch.softmax(logits, dim=-1)
            feats = CorrectnessHead.features(probs, device_batch["marker_mask"], device_batch["qtype"])
            confidence = torch.sigmoid(self.model.correctness(feats))

            records: list[dict[str, Any]] = []
            for row, item in enumerate(items):
                width = len(item.markers)
                p = probs[row, :width].float().cpu()
                p = p / max(float(p.sum()), 1e-9)
                conf = float(confidence[row].cpu())
                expected = float(sum(i * float(p[i]) for i in range(width)))
                label_index = int(torch.argmax(p).item())
                records.append(
                    {
                        "choice": item.option_keys[label_index],
                        "option_keys": list(item.option_keys),
                        "probs": [float(x) for x in p.tolist()],
                        "confidence": conf,
                        "expected_level": expected,
                        "label_index": label_index,
                    }
                )
            return records
