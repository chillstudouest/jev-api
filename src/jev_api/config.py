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

    api_version: str = "1.2.0"
    model_name: str = "von"
    model_version: str = "1.0"
    runtime_name: str = "pytorch+von+laya+agent-jev"

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

    laya_hf_repo: str = Field(default="convaiinnovations/laya", alias="LAYA_HF_REPO")
    laya_subfolder: str | None = Field(default=None, alias="LAYA_SUBFOLDER")
    preload_laya: bool = Field(default=False, alias="PRELOAD_LAYA")

    agentjev_hf_repo: str = Field(default="aimeigaoshou/agent-jev", alias="AGENTJEV_HF_REPO")
    agentjev_backbone: str = Field(default="Qwen/Qwen3-0.6B", alias="AGENTJEV_BACKBONE")
    agentjev_src: Path | None = Field(default=None, alias="AGENTJEV_SRC")
    preload_agentjev: bool = Field(default=False, alias="PRELOAD_AGENTJEV")


@lru_cache
def get_settings() -> Settings:
    return Settings()
