"""Runtime settings for jev-api. Secrets come from the environment only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_version: str = "1.1.0"
    model_name: str = "von-option-marker"
    model_version: str = "1.0"
    runtime_name: str = "pytorch+von-option-marker"

    host: str = "0.0.0.0"
    port: int = 8000

    jev_api_key: str = Field(default="", alias="JEV_API_KEY")
    # Optional — wfzyx/von-1.0 is public; token only needed for higher HF rate limits.
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_repo: str = Field(default="wfzyx/von-1.0", alias="HF_REPO")

    model_cache_dir: Path = Field(default=Path("/data/models"), alias="MODEL_CACHE_DIR")
    # option-marker = Von 395M (ModernBERT-Large). Also: von-1.0 (NLI encoder path).
    von_backend: str = Field(default="option-marker", alias="VON_BACKEND")
    device: str = Field(default="cpu", alias="JEV_DEVICE")
    download_on_startup: bool = Field(default=True, alias="DOWNLOAD_ON_STARTUP")


@lru_cache
def get_settings() -> Settings:
    return Settings()
