"""Runtime settings for jev-api. Secrets come from the environment only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKBONE = ROOT / "vendor" / "modernbert"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_version: str = "1.0.0"
    model_name: str = "openJev-verdict-2.0"
    model_version: str = "2.0"
    runtime_name: str = "pytorch+transformers"

    host: str = "0.0.0.0"
    port: int = 8000

    jev_api_key: str = Field(default="", alias="JEV_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_repo: str = Field(default="heman10x/openJev-verdict-2.0", alias="HF_REPO")
    hf_revision: str | None = Field(default=None, alias="HF_REVISION")

    model_cache_dir: Path = Field(default=Path("/data/models"), alias="MODEL_CACHE_DIR")
    backbone_dir: Path = Field(default=DEFAULT_BACKBONE, alias="BACKBONE_DIR")
    checkpoint_filename: str = Field(default="model.pt", alias="CHECKPOINT_FILENAME")

    device: str = Field(default="cpu", alias="JEV_DEVICE")
    download_on_startup: bool = Field(default=True, alias="DOWNLOAD_ON_STARTUP")
    # Local/dev: point at an already-downloaded checkpoint directory.
    local_checkpoint_dir: Path | None = Field(default=None, alias="LOCAL_CHECKPOINT_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()
