"""gliner-von (Gliner + Von) and the official Jev proxy, behind the Jev-compatible HTTP contract."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

from jev_api.config import Settings
from jev_api.schemas import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    QuestionTiming,
    RequestTimings,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    parse_questions,
    resolve_engine,
    resolve_request_model,
)
from jev_api.timing import Stopwatch

logger = logging.getLogger("jev_api.engine")


def _time_inference(run: Callable[[], T]) -> tuple[T, float, float | None]:
    """Return (result, inference_ms wall, gpu_duration_ms CUDA-or-None)."""
    started = time.perf_counter()
    gpu_ms: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = run()
            end.record()
            torch.cuda.synchronize()
            gpu_ms = round(float(start.elapsed_time(end)), 3)
        else:
            result = run()
    except ImportError:
        result = run()
    inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return result, inference_ms, gpu_ms


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
            "glinner": ("Gliner (GLiNER2.5-Decide)", "~0.6–1.8GB"),
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
    """Load Von on startup; load Gliner with it (PRELOAD_GLINNER) or on first request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.von_status = EngineStatus(
            device=settings.device,
            backend=settings.von_backend,
            name="von",
            backbone=settings.hf_repo,
        )
        self.glinner_status = EngineStatus(
            device=settings.device,
            backend="glinner",
            name="glinner",
            backbone=settings.glinner_hf_repo,
        )
        self._load_lock = threading.Lock()
        self._glinner_lock = threading.Lock()
        self._von: Any = None
        self._glinner: Any = None
        self._last_activity = time.monotonic()

    @property
    def status(self) -> EngineStatus:
        """Default readiness is Von — existing clients and Coolify keep working."""
        return self.von_status

    def start_background_load(self) -> None:
        thread = threading.Thread(target=self._safe_load, name="von-model-load", daemon=True)
        thread.start()
        if self.settings.preload_glinner:
            threading.Thread(target=self._safe_load_glinner, name="glinner-model-load", daemon=True).start()
        if self.settings.warmup_on_startup:
            threading.Thread(target=self._warm_loop, name="engine-warmup", daemon=True).start()

    def _warm_loop(self) -> None:
        """Warm once loads finish, then keep warm while idle. Never raises."""
        try:
            while not self._loads_settled():
                time.sleep(1.0)
            self.warmup()
            interval = self.settings.keep_warm_interval_s
            while interval > 0:
                time.sleep(interval)
                if time.monotonic() - self._last_activity >= interval:
                    self.warmup()
        except Exception:  # noqa: BLE001
            logger.exception("Warm loop stopped")

    def _loads_settled(self) -> bool:
        statuses = [self.von_status]
        if self.settings.preload_glinner:
            statuses.append(self.glinner_status)
        return all(s.ready or s.error for s in statuses)

    def warmup(self) -> None:
        """Run a dummy decision through every loaded engine (mixed question types)."""
        if not self.von_status.ready:
            return
        request = SystemOneRequest(
            state="Charged twice for September and cancelling Friday unless refunded.",
            model="gliner-von",
            questions={
                "route": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
                },
                "urgency": {
                    "type": "noul",
                    "instructions": "Does this need a reply today?",
                    "criteria": {"true": "Time-sensitive", "false": "Can wait"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is this?",
                    "criteria": ["Low", "Medium", "High", "Critical"],
                },
            },
        )
        started = time.perf_counter()
        try:
            self.systemone(request)
        except Exception:  # noqa: BLE001
            logger.exception("Warmup failed")
            return
        logger.info("Warmup gliner-von done in %.0f ms", (time.perf_counter() - started) * 1000.0)

    def load_blocking(self) -> None:
        """Load in the current thread (used during FastAPI lifespan startup)."""
        self._safe_load()
        if not self.von_status.ready:
            raise RuntimeError(self.von_status.not_ready_message())
        if self.settings.preload_glinner:
            self._safe_load_glinner()
            if not self.glinner_status.ready:
                raise RuntimeError(self.glinner_status.not_ready_message())

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

    def _safe_load_glinner(self) -> None:
        with self._glinner_lock:
            if self.glinner_status.ready:
                return
            if self.glinner_status.loading:
                return
            self.glinner_status.loading = True
            self.glinner_status.error = None
            try:
                self._load_glinner()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Gliner model load failed")
                self.glinner_status.error = str(exc)
                self.glinner_status.ready = False
                self._glinner = None
            finally:
                self.glinner_status.loading = False


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

    def _load_glinner(self) -> None:
        from jev_api.glinner_runtime import load_glinner_runtime

        self._prepare_hf_cache()
        runtime, params, backbone = load_glinner_runtime(self.settings)
        self._glinner = runtime
        self.glinner_status.parameters = params
        self.glinner_status.backbone = backbone
        self.glinner_status.device = runtime.device
        self.glinner_status.backend = runtime.backend
        self.glinner_status.ready = True
        self.glinner_status.error = None


    def ensure_glinner(self) -> None:
        if self.glinner_status.ready and self._glinner is not None:
            return
        self._safe_load_glinner()
        if not self.glinner_status.ready or self._glinner is None:
            raise RuntimeError(self.glinner_status.not_ready_message())


    def unload(self, name: str) -> None:
        """Drop an engine from RAM (sequential benchmarks)."""
        if name == "von":
            with self._load_lock:
                self._von = None
                self.von_status.ready = False
                self.von_status.loading = False
        elif name == "glinner":
            with self._glinner_lock:
                self._glinner = None
                self.glinner_status.ready = False
                self.glinner_status.loading = False
        else:
            raise ValueError(f"Cannot unload engine {name!r}")

    def systemone(self, request: SystemOneRequest) -> SystemOneResponse:
        watch = Stopwatch()
        model_name = resolve_request_model(request.model)
        engine_name = resolve_engine(model_name)
        watch.mark("resolve")
        parse_questions(request.questions)
        watch.mark("parse")

        if engine_name == "jev":
            answers, usage, inference_ms, gpu_duration_ms = self._evaluate_jev(request, watch)
        else:
            answers, usage, inference_ms, gpu_duration_ms = self._evaluate_gliner_von(request, model_name, watch)

        self._last_activity = time.monotonic()
        duration_ms = watch.total_ms()
        timings = RequestTimings(
            engine=engine_name,
            question_count=len(request.questions),
            state_chars=_state_chars(request.state),
            resolve_ms=watch.steps.get("resolve", 0.0),
            parse_ms=watch.steps.get("parse", 0.0),
            format_state_ms=watch.steps.get("format_state", 0.0),
            load_ms=watch.steps.get("load", 0.0),
            http_ms=watch.steps.get("http", 0.0),
            questions=[QuestionTiming.model_validate(row) for row in watch.questions],
            map_answers_ms=watch.steps.get("map_answers", 0.0),
            total_ms=duration_ms,
        )
        logger.info(
            "systemone engine=%s questions=%s state_chars=%s duration_ms=%.1f "
            "inference_ms=%.1f load_ms=%.1f resolve_ms=%.1f parse_ms=%.1f "
            "format_state_ms=%.1f http_ms=%.1f map_answers_ms=%.1f per_question=%s",
            engine_name,
            timings.question_count,
            timings.state_chars,
            duration_ms,
            inference_ms,
            timings.load_ms,
            timings.resolve_ms,
            timings.parse_ms,
            timings.format_state_ms,
            timings.http_ms,
            timings.map_answers_ms,
            [(row.id, row.type, row.ms) for row in timings.questions],
        )
        return SystemOneResponse(
            model=model_name,
            answers=answers,
            usage=usage,
            duration_ms=duration_ms,
            inference_ms=inference_ms,
            gpu_duration_ms=gpu_duration_ms,
            timings=timings,
        )

    def _evaluate_jev(
        self, request: SystemOneRequest, watch: Stopwatch
    ) -> tuple[dict[str, Answer], Usage, float, float | None]:
        from jev_api.typesafe_runtime import call_typesafe_jev

        raw = watch.measure(
            "http",
            lambda: call_typesafe_jev(self.settings, request.state, request.questions),
        )
        http_ms = watch.steps.get("http", 0.0)
        answers = raw["answers"]
        usage = raw["usage"]
        gpu_raw = raw.get("gpu_duration_ms")
        gpu_duration_ms = float(gpu_raw) if isinstance(gpu_raw, (int, float)) else None
        for qid, raw_q in request.questions.items():
            qtype = str(raw_q.get("type", "")) if isinstance(raw_q, dict) else ""
            watch.questions.append({"id": str(qid), "type": qtype, "ms": 0.0})
        return answers, usage, http_ms, gpu_duration_ms

    def _evaluate_von(
        self,
        request: SystemOneRequest,
        model_name: str,
        watch: Stopwatch,
    ) -> tuple[dict[str, Answer], Usage, float, float | None]:
        if not self.von_status.ready or self._von is None:
            raise RuntimeError(self.von_status.not_ready_message())

        from von.types import Choice, Noul, Score

        backend = self._von.backend
        state_str = watch.measure("format_state", lambda: _format_von_state(request.state))
        started = time.perf_counter()
        raw_answers: dict[str, object] = {}
        total_q_chars = 0
        gpu_ms: float | None = None
        for qid, raw in request.questions.items():
            qtype = str(raw.get("type", "choice")).strip().lower() if isinstance(raw, dict) else getattr(raw, "type", "choice")

            def _run(
                current: object = raw,
                kind: str = qtype,
                question_id: str = str(qid),
            ) -> object:
                if isinstance(current, dict):
                    if kind == "choice":
                        question = Choice(**current)
                    elif kind == "noul":
                        question = Noul(**current)
                    elif kind == "score":
                        question = Score(**current)
                    else:
                        raise ValueError(f"Unknown question type '{kind}'")
                else:
                    question = current
                if kind == "choice":
                    return backend.evaluate_choice(question_id, state_str, question)
                if kind == "noul":
                    return backend.evaluate_noul(question_id, state_str, question)
                if kind == "score":
                    return backend.evaluate_score(question_id, state_str, question)
                raise ValueError(f"Unknown question type '{kind}'")

            raw_answers[str(qid)] = watch.measure_question(str(qid), qtype, _run)
            instructions = raw.get("instructions") if isinstance(raw, dict) else getattr(raw, "instructions", "")
            total_q_chars += len(instructions or "")

        inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
        answers = watch.measure("map_answers", lambda: answers_from_mapping(raw_answers, source="Von"))
        usage = Usage(
            input_tokens=max(1, len(state_str) // 4) + max(1, total_q_chars // 4),
            output_tokens=len(answers),
        )
        return answers, usage, inference_ms, gpu_ms

    def _evaluate_gliner_von(
        self,
        request: SystemOneRequest,
        model_name: str,
        watch: Stopwatch,
    ) -> tuple[dict[str, Answer], Usage, float, float | None]:
        """noul goes to Von, choice / score to Gliner; answers keep the request order."""
        noul: dict[str, Any] = {}
        rest: dict[str, Any] = {}
        for qid, raw in request.questions.items():
            qtype = str(raw.get("type", "")).strip().lower() if isinstance(raw, dict) else ""
            (noul if qtype == "noul" else rest)[qid] = raw

        answers: dict[str, Answer] = {}
        input_tokens = output_tokens = 0
        inference_ms = 0.0
        if noul:
            von_answers, von_usage, von_ms, _ = self._evaluate_von(
                request.model_copy(update={"questions": noul}), model_name, watch
            )
            answers.update(von_answers)
            input_tokens += von_usage.input_tokens
            output_tokens += von_usage.output_tokens
            inference_ms += von_ms
        if rest:
            gliner_answers, gliner_usage, gliner_ms, _ = self._evaluate_glinner(
                request.model_copy(update={"questions": rest}), watch
            )
            answers.update(gliner_answers)
            input_tokens += gliner_usage.input_tokens
            output_tokens += gliner_usage.output_tokens
            inference_ms += gliner_ms

        ordered = {str(qid): answers[str(qid)] for qid in request.questions}
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        return ordered, usage, round(inference_ms, 3), None

    def _evaluate_glinner(
        self, request: SystemOneRequest, watch: Stopwatch
    ) -> tuple[dict[str, Answer], Usage, float, float | None]:
        watch.measure("load", self.ensure_glinner)
        if self._glinner is None:
            raise RuntimeError(self.glinner_status.not_ready_message())

        raw, inference_ms, gpu_duration_ms = _time_inference(
            lambda: self._glinner.evaluate(request.state, request.questions)
        )
        watch.steps["forward"] = inference_ms
        answers = watch.measure("map_answers", lambda: answers_from_mapping(raw["answers"], source="Gliner"))
        for qid, raw_q in request.questions.items():
            qtype = str(raw_q.get("type", "")) if isinstance(raw_q, dict) else ""
            watch.questions.append({"id": str(qid), "type": qtype, "ms": 0.0})
        usage_raw = raw["usage"]
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
        )
        return answers, usage, inference_ms, gpu_duration_ms


def _format_von_state(state: object) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{key}: {value}" for key, value in state.items())
    return str(state)


def _state_chars(state: object) -> int:
    if isinstance(state, str):
        return len(state)
    try:
        import json

        return len(json.dumps(state, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(state))


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
