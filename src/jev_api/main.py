"""jev-api — Jev-compatible System One HTTP server powered by Von 395M."""

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
        if settings.download_on_startup:
            # Block until Von is loaded so Traefik does not route traffic early.
            # First boot downloads ~1.5GB — Coolify start_period must be long enough.
            logger.info("Loading Von model before accepting traffic…")
            try:
                engine.load_blocking()
            except Exception:
                logger.exception("Startup model load failed — /ready will report error")
        yield

    app = FastAPI(
        title="jev-api",
        version=settings.api_version,
        description=(
            "Self-hosted Jev-compatible System One API. "
            "Protocol: TypeSafe / jev-agent POST /v1/systemone. "
            "Engine: Von OptionMarker 395M (wfzyx/von-1.0)."
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
                "von_backend": eng.status.backend,
                "note": "Protocol-compatible with Jev; predictions come from Von 395M OptionMarker",
            },
        )

    def _systemone(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
        eng: VerdictEngine = request.app.state.engine
        if not eng.status.ready:
            raise_usage(503, eng.status.not_ready_message(), error_type="server_error")
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
