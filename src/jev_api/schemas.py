"""HTTP request / response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Option(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)


class DecideRequest(BaseModel):
    type: str = Field(
        description="Decision type. 'workflow' maps to Verdict choice. Also: choice, score, noul."
    )
    question: str = Field(min_length=1, max_length=4096, description="Instructions / question text")
    state: str = Field(min_length=1, max_length=16384, description="Application / workflow state")
    options: list[Option] = Field(default_factory=list)
    case_id: str = Field(default="", max_length=256)
    workflow: str = Field(default="", max_length=256)
    qid: str = Field(default="decide", max_length=128)

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def validate_options(self) -> DecideRequest:
        qtype = self.type
        if qtype in {"workflow", "choice"} and len(self.options) < 2:
            raise ValueError("choice/workflow requires at least 2 options")
        if qtype == "score" and len(self.options) < 2:
            raise ValueError("score requires at least 2 level options")
        if qtype == "noul" and self.options and len(self.options) not in {0, 2}:
            raise ValueError("noul accepts 0 options (defaults) or exactly false/true")
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option ids must be unique")
        return self


class DecideResponse(BaseModel):
    choice: str
    scores: dict[str, float]
    confidence: float
    latency_ms: float
    qtype: Literal["choice", "score", "noul"]
    expected_level: float | None = None
    label_index: int
    option_keys: list[str]
    model: str = "openJev-verdict-2.0"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "loading", "error"]
    ready: bool
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    runtime: str
    parameters: int | None
    parameters_human: str | None
    device: str
    api_version: str
    backbone: str
    checkpoint_repo: str
    ready: bool
    extras: dict[str, Any] = Field(default_factory=dict)
