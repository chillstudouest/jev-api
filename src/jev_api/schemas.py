"""Jev / TypeSafe System One HTTP contract schemas.

Protocol-compatible with POST /v1/systemone.
Engines: `von`, `laya`, `agent-jev` (AgentJev-0.6B), and `semif` (SemIf / Qwen3.5-4B).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VON_MODELS = (
    "von",
    "von-latest",
    "von-preview",
    "von-1.0.0",
    "von-option-marker",
    "jev-latest",
    "jev-preview",
    "jev-1.13.0",
)
LAYA_MODELS = (
    "laya",
    "laya-latest",
    "laya-1.0",
)
AGENTJEV_MODELS = (
    "agent-jev",
    "agentjev",
    "agent-jev-0.6b",
)
SEMIF_MODELS = (
    "semif",
    "semif-latest",
    "semif-phase1",
    "openjev",
)
SUPPORTED_MODELS = VON_MODELS + LAYA_MODELS + AGENTJEV_MODELS + SEMIF_MODELS
DEFAULT_MODEL = "von"
EngineName = Literal["von", "laya", "agent-jev", "semif"]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2)


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: list[str] = Field(min_length=2)


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] | None = None

    @model_validator(mode="after")
    def validate_noul_criteria(self) -> NoulQuestion:
        if self.criteria is None:
            return self
        keys = set(self.criteria.keys())
        if keys and keys != {"true", "false"}:
            raise ValueError("noul criteria must be a {true, false} mapping")
        return self


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


class ApiUsageError(Exception):
    def __init__(self, message: str, error_type: str = "api_usage_error") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: Any = Field(..., description="Shared state (string or JSON object)")
    questions: dict[str, Any] = Field(min_length=1)
    model: str | None = None

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @field_validator("state")
    @classmethod
    def state_not_empty(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("state is required")
        if isinstance(value, str) and not value.strip():
            raise ValueError("state must not be empty")
        if isinstance(value, (list, dict)) and len(value) == 0:
            raise ValueError("state must not be empty")
        return value


def parse_questions(raw_questions: dict[str, Any]) -> dict[str, Question]:
    if not raw_questions:
        raise ApiUsageError("Invalid request. questions must not be empty")

    parsed: dict[str, Question] = {}
    for qid, raw in raw_questions.items():
        if not isinstance(raw, dict):
            raise ApiUsageError(f"Invalid request. questions.{qid} must be an object")
        qtype = str(raw.get("type", "")).strip().lower()
        try:
            if qtype == "choice":
                parsed[qid] = ChoiceQuestion.model_validate(raw)
            elif qtype == "score":
                parsed[qid] = ScoreQuestion.model_validate(raw)
            elif qtype == "noul":
                parsed[qid] = NoulQuestion.model_validate(raw)
            else:
                raise ApiUsageError("Invalid request.")
        except ApiUsageError:
            raise
        except Exception:
            from pydantic import ValidationError

            raise  # ValidationError → 422
    return parsed


def resolve_request_model(model: str | None) -> str:
    resolved = model or DEFAULT_MODEL
    if resolved not in SUPPORTED_MODELS:
        raise ApiUsageError(f"Unknown model: {resolved}")
    return resolved


def resolve_engine(model: str | None) -> EngineName:
    resolved = resolve_request_model(model)
    if resolved in LAYA_MODELS:
        return "laya"
    if resolved in AGENTJEV_MODELS:
        return "agent-jev"
    if resolved in SEMIF_MODELS:
        return "semif"
    return "von"


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    duration_ms: float = Field(description="Server-side inference duration in milliseconds")


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
    protocol: str = "jev-systemone"
    engine: str = "von-option-marker-395m"
    accepted_models: list[str] = Field(default_factory=lambda: list(SUPPORTED_MODELS))
    extras: dict[str, Any] = Field(default_factory=dict)
