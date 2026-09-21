"""Jev / TypeSafe System One HTTP contract schemas.

Protocol-compatible with POST /v1/systemone as documented at
https://jev-agent.com/api-reference — OpenJev is the inference engine only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_MODELS = ("jev-latest", "jev-preview", "jev-1.13.0")
DEFAULT_MODEL = "jev-1.13.0"
MODEL_ALIASES = {name: "openJev-verdict-2.0" for name in SUPPORTED_MODELS}

QuestionType = Literal["choice", "score", "noul"]


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
    """Maps to HTTP 400 api_usage_error (TypeSafe)."""

    def __init__(self, message: str, error_type: str = "api_usage_error") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class SystemOneRequest(BaseModel):
    """Jev System One request body (loose parse; deep validation in parse_questions)."""

    model_config = ConfigDict(extra="ignore")

    state: str = Field(min_length=1)
    questions: dict[str, Any] = Field(min_length=1)
    model: str | None = None

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODEL


def parse_questions(raw_questions: dict[str, Any]) -> dict[str, Question]:
    """Validate each question; unknown types → ApiUsageError (HTTP 400)."""
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
        except Exception as exc:  # pydantic ValidationError etc.
            # Missing fields → re-raise as validation-like; treat criteria/type issues as 400
            from pydantic import ValidationError

            if isinstance(exc, ValidationError):
                raise  # becomes 422
            raise ApiUsageError(str(exc) or "Invalid request.") from exc
    return parsed


def resolve_request_model(model: str | None) -> str:
    resolved = model or DEFAULT_MODEL
    if resolved not in MODEL_ALIASES:
        raise ApiUsageError(f"Unknown model: {resolved}")
    return resolved


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
    engine: str = "openJev-verdict-2.0"
    accepted_models: list[str] = Field(default_factory=lambda: list(SUPPORTED_MODELS))
    extras: dict[str, Any] = Field(default_factory=dict)
