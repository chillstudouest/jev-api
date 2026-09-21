"""jev-api FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from jev_api.auth import require_api_key
from jev_api.config import Settings, get_settings
from jev_api.engine import VerdictEngine
from jev_api.schemas import (
    DecideRequest,
    DecideResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
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
            engine.start_background_load()
        yield

    app = FastAPI(
        title="jev-api",
        version=settings.api_version,
        description="HTTP microservice for openJev Verdict 2.0 decision inference",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready(request: Request) -> ReadyResponse:
        eng: VerdictEngine = request.app.state.engine
        if eng.status.ready:
            return ReadyResponse(status="ready", ready=True)
        if eng.status.loading:
            return ReadyResponse(status="loading", ready=False, detail="Model is loading")
        if eng.status.error:
            return ReadyResponse(status="error", ready=False, detail=eng.status.error)
        return ReadyResponse(status="loading", ready=False, detail="Model not started")

    @app.get(
        "/v1/model",
        response_model=ModelInfoResponse,
        dependencies=[Depends(require_api_key)],
        tags=["v1"],
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
            backbone=eng.status.backbone or str(cfg.backbone_dir),
            checkpoint_repo=cfg.hf_repo,
            ready=eng.status.ready,
            extras={
                "temperature_shape": [3, 8],
                "heads": ["marker_pointer", "correctness"],
                "max_len": 512,
                "head_max_len": 192,
                "option_token_cap": 48,
            },
        )

    @app.post(
        "/v1/decide",
        response_model=DecideResponse,
        dependencies=[Depends(require_api_key)],
        tags=["v1"],
    )
    def decide(payload: DecideRequest, request: Request) -> DecideResponse:
        eng: VerdictEngine = request.app.state.engine
        if not eng.status.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=eng.status.error or "Model is not ready",
            )
        try:
            return eng.decide(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    # Single worker: one in-memory model copy. Multiple workers would multiply RAM.
    uvicorn.run(
        "jev_api.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    run()
