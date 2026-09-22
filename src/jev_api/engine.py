"""Von + Laya engines behind the Jev-compatible HTTP contract."""

from __future__ import annotations

import logging
import os
import threading
import time
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
    resolve_engine,
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
    backend: str = "von"
    name: str = "von"

    def not_ready_message(self) -> str:
        labels = {
            "von": ("Von OptionMarker", "~1.5GB"),
            "laya": ("Laya", "~0.8GB"),
            "agent-jev": ("AgentJev-0.6B", "~1.2GB"),
            "semif": ("SemIf Qwen3.5-4B", "~3GB GGUF / ~8GB BF16"),
            "djev": ("djev (Maisa, diffusion-gemma)", "remote API"),
        }
        label, size = labels.get(self.name, (self.name, "weights"))
        if self.error:
            return f"{label} failed to load: {self.error}"
        if self.loading:
            return (
                f"{label} is still loading (downloading / initializing {size}). "
                "Poll GET /ready or retry shortly."
            )
        return f"{label} is not ready"


class VerdictEngine:
    """Load Von on startup; load Laya lazily (or via PRELOAD_LAYA)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.von_status = EngineStatus(
            device=settings.device,
            backend=settings.von_backend,
            name="von",
            backbone=settings.hf_repo,
        )
        self.laya_status = EngineStatus(
            device=settings.device,
            backend="laya",
            name="laya",
            backbone=settings.laya_hf_repo,
        )
        self.agentjev_status = EngineStatus(
            device=settings.device,
            backend="agent-jev",
            name="agent-jev",
            backbone=settings.agentjev_hf_repo,
        )
        self.semif_status = EngineStatus(
            device=settings.device,
            backend="semif",
            name="semif",
            backbone=settings.semif_hf_repo,
        )
        self.djev_status = EngineStatus(
            device="remote",
            backend="djev-http",
            name="djev",
            backbone=settings.djev_base_url,
        )
        self._load_lock = threading.Lock()
        self._laya_lock = threading.Lock()
        self._agentjev_lock = threading.Lock()
        self._semif_lock = threading.Lock()
        self._djev_lock = threading.Lock()
        self._von: Any = None
        self._laya: Any = None
        self._agentjev: Any = None
        self._semif: Any = None
        self._djev: Any = None

    @property
    def status(self) -> EngineStatus:
        """Default readiness is Von — existing clients and Coolify keep working."""
        return self.von_status

    def start_background_load(self) -> None:
        thread = threading.Thread(target=self._safe_load, name="von-model-load", daemon=True)
        thread.start()
        if self.settings.preload_laya:
            threading.Thread(target=self._safe_load_laya, name="laya-model-load", daemon=True).start()
        if self.settings.preload_agentjev:
            threading.Thread(
                target=self._safe_load_agentjev, name="agentjev-model-load", daemon=True
            ).start()
        if self.settings.preload_semif:
            threading.Thread(target=self._safe_load_semif, name="semif-model-load", daemon=True).start()
        if self.settings.preload_djev:
            threading.Thread(target=self._safe_load_djev, name="djev-client-load", daemon=True).start()

    def load_blocking(self) -> None:
        """Load in the current thread (used during FastAPI lifespan startup)."""
        self._safe_load()
        if not self.von_status.ready:
            raise RuntimeError(self.von_status.not_ready_message())
        if self.settings.preload_laya:
            self._safe_load_laya()
            if not self.laya_status.ready:
                raise RuntimeError(self.laya_status.not_ready_message())
        if self.settings.preload_agentjev:
            self._safe_load_agentjev()
            if not self.agentjev_status.ready:
                raise RuntimeError(self.agentjev_status.not_ready_message())
        if self.settings.preload_semif:
            self._safe_load_semif()
            if not self.semif_status.ready:
                raise RuntimeError(self.semif_status.not_ready_message())
        if self.settings.preload_djev:
            self._safe_load_djev()
            if not self.djev_status.ready:
                raise RuntimeError(self.djev_status.not_ready_message())

    def _safe_load(self) -> None:
        with self._load_lock:
            if self.von_status.ready:
                return
            if self.von_status.loading:
                return
            self.von_status.loading = True
            self.von_status.error = None
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Von model load failed")
                self.von_status.error = str(exc)
                self.von_status.ready = False
                self._von = None
            finally:
                self.von_status.loading = False

    def _safe_load_laya(self) -> None:
        with self._laya_lock:
            if self.laya_status.ready:
                return
            if self.laya_status.loading:
                return
            self.laya_status.loading = True
            self.laya_status.error = None
            try:
                self._load_laya()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Laya model load failed")
                self.laya_status.error = str(exc)
                self.laya_status.ready = False
                self._laya = None
            finally:
                self.laya_status.loading = False

    def _safe_load_agentjev(self) -> None:
        with self._agentjev_lock:
            if self.agentjev_status.ready:
                return
            if self.agentjev_status.loading:
                return
            self.agentjev_status.loading = True
            self.agentjev_status.error = None
            try:
                self._load_agentjev()
            except Exception as exc:  # noqa: BLE001
                logger.exception("AgentJev model load failed")
                self.agentjev_status.error = str(exc)
                self.agentjev_status.ready = False
                self._agentjev = None
            finally:
                self.agentjev_status.loading = False

    def _safe_load_semif(self) -> None:
        with self._semif_lock:
            if self.semif_status.ready:
                return
            if self.semif_status.loading:
                return
            self.semif_status.loading = True
            self.semif_status.error = None
            try:
                self._load_semif()
            except Exception as exc:  # noqa: BLE001
                logger.exception("SemIf model load failed")
                self.semif_status.error = str(exc)
                self.semif_status.ready = False
                self._semif = None
            finally:
                self.semif_status.loading = False

    def _safe_load_djev(self) -> None:
        with self._djev_lock:
            if self.djev_status.ready:
                return
            if self.djev_status.loading:
                return
            self.djev_status.loading = True
            self.djev_status.error = None
            try:
                self._load_djev()
            except Exception as exc:  # noqa: BLE001
                logger.exception("djev client load failed")
                self.djev_status.error = str(exc)
                self.djev_status.ready = False
                self._djev = None
            finally:
                self.djev_status.loading = False

    def _prepare_hf_cache(self) -> str:
        settings = self.settings
        cache = str(settings.model_cache_dir)
        os.makedirs(cache, exist_ok=True)
        os.environ["HF_HOME"] = cache
        os.environ["TRANSFORMERS_CACHE"] = cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
        return cache

    def _open_von(self) -> Any:
        from von.engine import VonEngine

        settings = self.settings
        candidates: list[str] = []
        for name in (settings.von_backend, "von", "von-1.1", "option-marker"):
            if name and name not in candidates:
                candidates.append(name)

        last_error: Exception | None = None
        for backend_name in candidates:
            try:
                von = VonEngine(backend_name=backend_name, device=settings.device)
                self.von_status.backend = backend_name
                return von
            except ValueError as exc:
                last_error = exc
                logger.warning("Von backend %s rejected: %s", backend_name, exc)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to construct VonEngine")

    def load(self) -> None:
        settings = self.settings
        cache = self._prepare_hf_cache()
        os.environ["VON_DEVICE"] = settings.device
        os.environ["VON_BACKEND"] = settings.von_backend

        logger.info("Loading Von backend=%s device=%s cache=%s", settings.von_backend, settings.device, cache)
        von = self._open_von()

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
        self.von_status.parameters = params
        self.von_status.backbone = settings.hf_repo
        self.von_status.device = settings.device
        self.von_status.ready = True
        self.von_status.error = None
        if params is not None:
            logger.info("Von ready: %.1fM parameters on %s", params / 1e6, settings.device)
        else:
            logger.info("Von ready on %s", settings.device)

    def _load_laya(self) -> None:
        from jev_api.laya_runtime import load_laya_agent

        self._prepare_hf_cache()
        agent, params, backbone = load_laya_agent(self.settings)
        self._laya = agent
        self.laya_status.parameters = params
        self.laya_status.backbone = backbone
        self.laya_status.device = self.settings.device
        self.laya_status.backend = "laya"
        self.laya_status.ready = True
        self.laya_status.error = None

    def _load_agentjev(self) -> None:
        from jev_api.agentjev_runtime import load_agentjev_engine

        self._prepare_hf_cache()
        engine, params, backbone = load_agentjev_engine(self.settings)
        self._agentjev = engine
        self.agentjev_status.parameters = params
        self.agentjev_status.backbone = backbone
        self.agentjev_status.device = self.settings.device
        self.agentjev_status.backend = "agent-jev"
        self.agentjev_status.ready = True
        self.agentjev_status.error = None

    def _load_semif(self) -> None:
        from jev_api.semif_runtime import load_semif_runtime

        self._prepare_hf_cache()
        runtime, params, backbone = load_semif_runtime(self.settings)
        self._semif = runtime
        self.semif_status.parameters = params
        self.semif_status.backbone = backbone
        self.semif_status.device = self.settings.device
        self.semif_status.backend = runtime.backend
        self.semif_status.ready = True
        self.semif_status.error = None

    def _load_djev(self) -> None:
        from jev_api.djev_runtime import load_djev_client

        client, params, backbone = load_djev_client(self.settings)
        self._djev = client
        self.djev_status.parameters = params
        self.djev_status.backbone = backbone
        self.djev_status.device = "remote"
        self.djev_status.backend = "djev-http"
        self.djev_status.ready = True
        self.djev_status.error = None

    def ensure_laya(self) -> None:
        if self.laya_status.ready and self._laya is not None:
            return
        self._safe_load_laya()
        if not self.laya_status.ready or self._laya is None:
            raise RuntimeError(self.laya_status.not_ready_message())

    def ensure_agentjev(self) -> None:
        if self.agentjev_status.ready and self._agentjev is not None:
            return
        self._safe_load_agentjev()
        if not self.agentjev_status.ready or self._agentjev is None:
            raise RuntimeError(self.agentjev_status.not_ready_message())

    def ensure_semif(self) -> None:
        if self.semif_status.ready and self._semif is not None:
            return
        self._safe_load_semif()
        if not self.semif_status.ready or self._semif is None:
            raise RuntimeError(self.semif_status.not_ready_message())

    def ensure_djev(self) -> None:
        if self.djev_status.ready and self._djev is not None:
            return
        self._safe_load_djev()
        if not self.djev_status.ready or self._djev is None:
            raise RuntimeError(self.djev_status.not_ready_message())

    def unload(self, name: str) -> None:
        """Drop an engine from RAM (sequential benchmarks)."""
        if name == "von":
            with self._load_lock:
                self._von = None
                self.von_status.ready = False
                self.von_status.loading = False
        elif name == "laya":
            with self._laya_lock:
                self._laya = None
                self.laya_status.ready = False
                self.laya_status.loading = False
        elif name == "agent-jev":
            with self._agentjev_lock:
                self._agentjev = None
                self.agentjev_status.ready = False
                self.agentjev_status.loading = False
        elif name == "semif":
            with self._semif_lock:
                self._semif = None
                self.semif_status.ready = False
                self.semif_status.loading = False
        elif name == "djev":
            with self._djev_lock:
                self._djev = None
                self.djev_status.ready = False
                self.djev_status.loading = False
        else:
            raise ValueError(f"Cannot unload engine {name!r}")

    def systemone(self, request: SystemOneRequest) -> SystemOneResponse:
        model_name = resolve_request_model(request.model)
        engine_name = resolve_engine(model_name)
        parse_questions(request.questions)

        started = time.perf_counter()
        if engine_name == "laya":
            answers, usage = self._evaluate_laya(request)
        elif engine_name == "agent-jev":
            answers, usage = self._evaluate_agentjev(request)
        elif engine_name == "semif":
            answers, usage = self._evaluate_semif(request)
        elif engine_name == "djev":
            answers, usage = self._evaluate_djev(request)
        else:
            answers, usage = self._evaluate_von(request, model_name)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

        return SystemOneResponse(
            model=model_name,
            answers=answers,
            usage=usage,
            duration_ms=duration_ms,
        )

    def _evaluate_von(self, request: SystemOneRequest, model_name: str) -> tuple[dict[str, Answer], Usage]:
        if not self.von_status.ready or self._von is None:
            raise RuntimeError(self.von_status.not_ready_message())

        von_resp = self._von.evaluate(
            state=request.state,
            questions=request.questions,
            model=model_name,
        )
        answers = answers_from_mapping(von_resp.answers, source="Von")
        usage_raw = von_resp.usage
        usage = Usage(
            input_tokens=int(getattr(usage_raw, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage_raw, "output_tokens", 0) or 0),
        )
        return answers, usage

    def _evaluate_laya(self, request: SystemOneRequest) -> tuple[dict[str, Answer], Usage]:
        from jev_api.laya_runtime import laya_system_one

        self.ensure_laya()
        if self._laya is None:
            raise RuntimeError(self.laya_status.not_ready_message())

        raw = laya_system_one(self._laya, request.state, request.questions)
        raw_answers = raw.get("answers")
        if not isinstance(raw_answers, dict):
            raise ValueError("Laya response is missing answers")
        answers = answers_from_mapping(raw_answers, source="Laya")
        usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
        )
        return answers, usage

    def _evaluate_agentjev(self, request: SystemOneRequest) -> tuple[dict[str, Answer], Usage]:
        from jev_api.agentjev_runtime import evaluate_agentjev

        self.ensure_agentjev()
        if self._agentjev is None:
            raise RuntimeError(self.agentjev_status.not_ready_message())

        raw = evaluate_agentjev(self._agentjev, request.state, request.questions)
        answers = answers_from_mapping(raw["answers"], source="AgentJev")
        usage_raw = raw["usage"]
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
        )
        return answers, usage

    def _evaluate_semif(self, request: SystemOneRequest) -> tuple[dict[str, Answer], Usage]:
        self.ensure_semif()
        if self._semif is None:
            raise RuntimeError(self.semif_status.not_ready_message())

        raw = self._semif.evaluate(request.state, request.questions)
        answers = answers_from_mapping(raw["answers"], source="SemIf")
        usage_raw = raw["usage"]
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
        )
        return answers, usage

    def _evaluate_djev(self, request: SystemOneRequest) -> tuple[dict[str, Answer], Usage]:
        self.ensure_djev()
        if self._djev is None:
            raise RuntimeError(self.djev_status.not_ready_message())

        raw = self._djev.evaluate(request.state, request.questions)
        answers = answers_from_mapping(raw["answers"], source="djev")
        usage_raw = raw["usage"]
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
        )
        return answers, usage


def answers_from_mapping(raw_answers: object, source: str) -> dict[str, Answer]:
    if not isinstance(raw_answers, dict):
        raise ValueError(f"Unknown answer payload from {source}")

    answers: dict[str, Answer] = {}
    for qid, ans in raw_answers.items():
        if hasattr(ans, "model_dump"):
            data = ans.model_dump()
        elif isinstance(ans, dict):
            data = dict(ans)
        else:
            raise ValueError(f"Unknown answer type from {source}: {type(ans).__name__}")
        qtype = data.get("type")
        if qtype == "choice":
            answers[str(qid)] = ChoiceAnswer.model_validate(data)
        elif qtype == "score":
            answers[str(qid)] = ScoreAnswer.model_validate(data)
        elif qtype == "noul":
            answers[str(qid)] = NoulAnswer(noul=float(data["noul"]))
        else:
            raise ValueError(f"Unknown answer type from {source}: {qtype!r}")
    return answers
