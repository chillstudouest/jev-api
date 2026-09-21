"""Von 395M (OptionMarker) engine behind the Jev-compatible HTTP contract."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from jev_api.config import Settings
from jev_api.schemas import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
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
    backbone: str = "wfzyx/von-1.0"
    device: str = "cpu"
    backend: str = "option-marker"


class VerdictEngine:
    """Thin wrapper: load Von once, expose systemone() for the HTTP layer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = EngineStatus(device=settings.device, backend=settings.von_backend)
        self._load_lock = threading.Lock()
        self._von: Any = None

    def start_background_load(self) -> None:
        thread = threading.Thread(target=self._safe_load, name="von-model-load", daemon=True)
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
                logger.exception("Von model load failed")
                self.status.error = str(exc)
                self.status.ready = False
            finally:
                self.status.loading = False

    def load(self) -> None:
        settings = self.settings
        # Hugging Face cache dir (public wfzyx/von-1.0)
        cache = str(settings.model_cache_dir)
        os.makedirs(cache, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache)
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

        os.environ["VON_DEVICE"] = settings.device
        os.environ["VON_BACKEND"] = settings.von_backend

        from von.engine import VonEngine

        logger.info("Loading Von backend=%s device=%s", settings.von_backend, settings.device)
        von = VonEngine(backend_name=settings.von_backend, device=settings.device)

        # Force weight download + load once at startup (not on first request).
        backend = von.backend
        if hasattr(backend, "_get_model"):
            model = backend._get_model()
            params = sum(p.numel() for p in model.parameters())
        elif hasattr(backend, "_get_model_and_tok"):
            model, _tok = backend._get_model_and_tok()
            params = sum(p.numel() for p in model.parameters())
        else:
            params = None

        self._von = von
        self.status.parameters = params
        self.status.backbone = settings.hf_repo
        self.status.device = settings.device
        self.status.backend = settings.von_backend
        self.status.ready = True
        self.status.error = None
        if params is not None:
            logger.info("Von ready: %.1fM parameters on %s", params / 1e6, settings.device)
        else:
            logger.info("Von ready on %s", settings.device)

    def systemone(self, request: SystemOneRequest) -> SystemOneResponse:
        if not self.status.ready or self._von is None:
            raise RuntimeError(self.status.error or "Model is not ready")

        # Validate Jev question shapes (raises ApiUsageError / ValidationError).
        model_name = resolve_request_model(request.model)
        parse_questions(request.questions)

        von_resp = self._von.evaluate(
            state=request.state,
            questions=request.questions,
            model=model_name,
        )

        answers: dict[str, Answer] = {}
        for qid, ans in von_resp.answers.items():
            data = ans.model_dump() if hasattr(ans, "model_dump") else dict(ans)
            qtype = data.get("type")
            if qtype == "choice":
                answers[qid] = ChoiceAnswer.model_validate(data)
            elif qtype == "score":
                answers[qid] = ScoreAnswer.model_validate(data)
            elif qtype == "noul":
                # Strip confidence if a backend ever adds it — Jev Noul has none.
                answers[qid] = NoulAnswer(noul=float(data["noul"]))
            else:
                raise ValueError(f"Unknown answer type from Von: {qtype!r}")

        usage_raw = von_resp.usage
        usage = Usage(
            input_tokens=int(getattr(usage_raw, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage_raw, "output_tokens", 0) or 0),
        )
        # Echo the client-facing Jev model alias (protocol compat), not the internal Von name.
        return SystemOneResponse(model=model_name, answers=answers, usage=usage)
