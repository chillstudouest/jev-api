"""Runtime settings for jev-api. Secrets come from the environment only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_version: str = "1.3.0"
    model_name: str = "von"
    model_version: str = "1.0"
    runtime_name: str = "pytorch+von+glinner+jev"

    host: str = "0.0.0.0"
    port: int = 8000

    jev_api_key: str = Field(default="", alias="JEV_API_KEY")
    # Optional — wfzyx/von-1.0 is public; token only needed for higher HF rate limits.
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_repo: str = Field(default="wfzyx/von-1.0", alias="HF_REPO")

    model_cache_dir: Path = Field(default=Path("/data/models"), alias="MODEL_CACHE_DIR")
    # Latest von-sdk accepts von / von-1.1 / latest. Older images used option-marker.
    von_backend: str = Field(default="von", alias="VON_BACKEND")
    device: str = Field(default="cpu", alias="JEV_DEVICE")
    download_on_startup: bool = Field(default=True, alias="DOWNLOAD_ON_STARTUP")

    glinner_hf_repo: str = Field(
        default="fastino/GLiNER2.5-Decide", alias="GLINNER_HF_REPO"
    )
    preload_glinner: bool = Field(default=False, alias="PRELOAD_GLINNER")
    # One dummy decision per loaded engine after load, so the first real request is not cold.
    warmup_on_startup: bool = Field(default=True, alias="WARMUP_ON_STARTUP")
    # Re-run the dummy decision when no request came in for this many seconds, so the OS
    # keeps the weights resident and the CPU awake. 0 disables.
    keep_warm_interval_s: float = Field(default=45.0, alias="KEEP_WARM_INTERVAL_S")
    # torch (gliner2 library) or onnx (onnxruntime, CPU only).
    glinner_backend: str = Field(default="onnx", alias="GLINNER_BACKEND")
    glinner_onnx_repo: str = Field(
        default="nishparadox/gliner2.5-decide-onnx", alias="GLINNER_ONNX_REPO"
    )
    glinner_onnx_revision: str = Field(
        default="bbbcdb01c406b4f9a44b3a8102c96edb91c1f47d", alias="GLINNER_ONNX_REVISION"
    )
    glinner_onnx_variant: str = Field(default="fp32", alias="GLINNER_ONNX_VARIANT")
    glinner_threads: int | None = Field(default=None, alias="GLINNER_THREADS")

    typesafe_api_key: str = Field(default="", alias="TYPESAFE_API_KEY")
    typesafe_base_url: str = Field(default="https://api.typesafe.ai", alias="TYPESAFE_BASE_URL")
    typesafe_model: str = Field(default="jev-1.13.0", alias="TYPESAFE_MODEL")
    typesafe_timeout: float = Field(default=60.0, alias="TYPESAFE_TIMEOUT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
