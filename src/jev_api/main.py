"""jev-api — Jev-compatible System One HTTP server (Von + Laya)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from jev_api.auth import require_api_key
from jev_api.config import Settings, get_settings
from jev_api.engine import VerdictEngine
from jev_api.errors import (
    http_exception_handler,
    raise_usage,
    validation_exception_handler,
)
from jev_api.schemas import (
    ApiUsageError,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    SystemOneRequest,
    SystemOneResponse,
    resolve_engine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("jev_api")


def create_app(settings: Settings | None = None, engine: VerdictEngine | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or VerdictEngine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _app.state.settings = settings
        _app.state.engine = engine
        # Background load so /health becomes ready immediately (Coolify).
        # Poll GET /ready before calling /v1/systemone.
        if settings.download_on_startup:
            logger.info("Starting Von model load in background…")
            engine.start_background_load()
        yield

    app = FastAPI(
        title="jev-api",
        version=settings.api_version,
        description=(
            "Self-hosted Jev-compatible System One API. "
            "Protocol: TypeSafe / jev-agent POST /v1/systemone. "
            "Engines: von (default), laya, agent-jev, semif, or djev (Maisa/DiffusionGemma). "
            "Switch with the request body `model` field."
        ),
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready(request: Request) -> ReadyResponse:
        eng: VerdictEngine = request.app.state.engine
        if eng.status.ready:
            return ReadyResponse(status="ready", ready=True)
        if eng.status.loading:
            return ReadyResponse(
                status="loading",
                ready=False,
                detail="Downloading / initializing Von OptionMarker (~1.5GB). Retry shortly.",
            )
        if eng.status.error:
            return ReadyResponse(status="error", ready=False, detail=eng.status.error)
        return ReadyResponse(status="loading", ready=False, detail="Model not started")

    @app.get(
        "/v1/model",
        response_model=ModelInfoResponse,
        dependencies=[Depends(require_api_key)],
        tags=["ops"],
    )
    def model_info(request: Request) -> ModelInfoResponse:
        eng: VerdictEngine = request.app.state.engine
        cfg: Settings = request.app.state.settings
        params = eng.status.parameters
        human = f"{params / 1e6:.1f}M" if params is not None else None
        return ModelInfoResponse(
            name=cfg.model_name,
            version=cfg.model_version,
            runtime=cfg.runtime_name,
            parameters=params,
            parameters_human=human,
            device=eng.status.device,
            api_version=cfg.api_version,
            backbone=eng.status.backbone,
            checkpoint_repo=cfg.hf_repo,
            ready=eng.status.ready,
            extras={
                "default_model": "von",
                "von_backend": eng.von_status.backend,
                "von": {
                    "ready": eng.von_status.ready,
                    "backend": eng.von_status.backend,
                    "backbone": eng.von_status.backbone,
                    "error": eng.von_status.error,
                },
                "laya": {
                    "ready": eng.laya_status.ready,
                    "backend": eng.laya_status.backend,
                    "backbone": eng.laya_status.backbone,
                    "error": eng.laya_status.error,
                    "runtime": "python+transformers",
                },
                "agent-jev": {
                    "ready": eng.agentjev_status.ready,
                    "backend": eng.agentjev_status.backend,
                    "backbone": eng.agentjev_status.backbone,
                    "error": eng.agentjev_status.error,
                    "runtime": "qwen3-0.6b+candidate-head",
                },
                "semif": {
                    "ready": eng.semif_status.ready,
                    "backend": eng.semif_status.backend,
                    "backbone": eng.semif_status.backbone,
                    "error": eng.semif_status.error,
                    "runtime": "qwen3.5-4b-option-logits",
                },
                "djev": {
                    "ready": eng.djev_status.ready,
                    "backend": eng.djev_status.backend,
                    "backbone": eng.djev_status.backbone,
                    "error": eng.djev_status.error,
                    "runtime": "maisa-diffusion-gemma-http",
                    "upstream": "https://github.com/Davipar/djev-dev",
                    "note": "JevBench: djev (Maisa, diffusion-gemma) structured one-step read — not thinking path",
                },
                "note": (
                    "Switch engine with body.model: von (default), laya, agent-jev, semif, or djev. "
                    "laya-mlx is Apple Silicon only; this service uses official laya on CPU. "
                    "SemIf torch needs CUDA/MPS; CPU uses llama.cpp GGUF (~3GB). "
                    "djev proxies Davipar/djev-dev (Maisa DiffusionGemma). "
                    "Self-hosted needs no API key; DJEV_API_KEY is only for api.djev.dev."
                ),
            },
        )

    def _systemone(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
        eng: VerdictEngine = request.app.state.engine
        try:
            engine_name = resolve_engine(payload.model)
        except ApiUsageError as exc:
            raise_usage(400, exc.message, exc.error_type)
            raise
        if engine_name == "von" and not eng.von_status.ready:
            raise_usage(503, eng.von_status.not_ready_message(), error_type="server_error")
        try:
            return eng.systemone(payload)
        except ApiUsageError as exc:
            raise_usage(400, exc.message, exc.error_type)
        except ValidationError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except ValueError as exc:
            raise_usage(400, str(exc) or "Invalid request.")
        except RuntimeError as exc:
            raise_usage(503, str(exc), error_type="server_error")

    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        dependencies=[Depends(require_api_key)],
        tags=["systemone"],
    )
    def systemone(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
        return _systemone(payload, request)

    @app.post(
        "/api/v1/systemone",
        response_model=SystemOneResponse,
        dependencies=[Depends(require_api_key)],
        tags=["systemone"],
        include_in_schema=False,
    )
    def systemone_api_alias(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
        return _systemone(payload, request)

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "jev_api.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    run()
