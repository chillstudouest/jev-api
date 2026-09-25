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
    runtime_name: str = "pytorch+von+laya+semif+glinner+jev"

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


    semif_hf_repo: str = Field(default="Qwen/Qwen3.5-4B", alias="SEMIF_HF_REPO")
    semif_revision: str = Field(
        default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", alias="SEMIF_REVISION"
    )
    semif_backend: str = Field(default="auto", alias="SEMIF_BACKEND")
    semif_mode: str = Field(default="auto", alias="SEMIF_MODE")
    semif_dtype: str = Field(default="bfloat16", alias="SEMIF_DTYPE")
    semif_gguf_repo: str = Field(
        default="bartowski/Qwen_Qwen3.5-4B-GGUF", alias="SEMIF_GGUF_REPO"
    )
    semif_gguf_file: str = Field(
        default="Qwen_Qwen3.5-4B-Q4_K_M.gguf", alias="SEMIF_GGUF_FILE"
    )
    semif_gguf_path: Path | None = Field(default=None, alias="SEMIF_GGUF_PATH")
    semif_src: Path | None = Field(default=None, alias="SEMIF_SRC")
    semif_max_tokens: int = Field(default=4096, alias="SEMIF_MAX_TOKENS")
    semif_llama_threads: int | None = Field(default=None, alias="SEMIF_LLAMA_THREADS")
    preload_semif: bool = Field(default=False, alias="PRELOAD_SEMIF")

    glinner_hf_repo: str = Field(
        default="knowledgator/gliclass-base-v1.0", alias="GLINNER_HF_REPO"
    )
    glinner_max_length: int = Field(default=1024, alias="GLINNER_MAX_LENGTH")
    preload_glinner: bool = Field(default=False, alias="PRELOAD_GLINNER")

    typesafe_api_key: str = Field(default="", alias="TYPESAFE_API_KEY")
    typesafe_base_url: str = Field(default="https://api.typesafe.ai", alias="TYPESAFE_BASE_URL")
    typesafe_model: str = Field(default="jev-1.13.0", alias="TYPESAFE_MODEL")
    typesafe_timeout: float = Field(default=60.0, alias="TYPESAFE_TIMEOUT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
