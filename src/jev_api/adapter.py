"""Adapt Jev System One questions ↔ openJev Verdict 2.0 items / answers."""

from __future__ import annotations

from typing import Any

from jev_api.preprocess import Item, build_item
from jev_api.schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


def question_to_qdef(question: Question) -> dict[str, Any]:
    """Convert a validated Jev question into upstream verdict2 qdef."""
    if isinstance(question, ChoiceQuestion):
        criteria = {k: ("" if v is None else v) for k, v in question.criteria.items()}
        return {"type": "choice", "instructions": question.instructions, "criteria": criteria}
    if isinstance(question, ScoreQuestion):
        return {"type": "score", "instructions": question.instructions, "criteria": list(question.criteria)}
    # Noul
    criteria = question.criteria or {}
    return {
        "type": "noul",
        "instructions": question.instructions,
        "criteria": {k: ("" if v is None else v) for k, v in criteria.items()},
    }


def build_jev_item(tokenizer: Any, state: str, qid: str, question: Question) -> Item:
    qdef = question_to_qdef(question)
    keys = _option_keys(question)
    gold = {
        "probabilities": {k: 0.0 for k in keys},
        "label": keys[0] if keys else "",
        "score": 0.0,
    }
    row = {"state": state, "id": "", "workflow": ""}
    item = build_item(tokenizer, row, qid, qdef, gold)
    if item is None:
        raise ValueError("max_tokens_exceeded")
    return item


def _option_keys(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria.keys())
    if isinstance(question, ScoreQuestion):
        return [str(i) for i in range(len(question.criteria))]
    return ["false", "true"]


def to_jev_answer(question: Question, infer: dict[str, Any]) -> Answer:
    """Map an OpenJev inference record onto a Jev answer primitive."""
    probs_list: list[float] = infer["probs"]
    keys: list[str] = list(infer["option_keys"])
    confidence = float(infer["confidence"])

    if isinstance(question, ChoiceQuestion):
        probabilities = {k: float(probs_list[i]) for i, k in enumerate(keys)}
        return ChoiceAnswer(
            choice=str(infer["choice"]),
            probabilities=probabilities,
            confidence=confidence,
        )

    if isinstance(question, ScoreQuestion):
        # Jev score = weighted mean of level indices 0..n-1 (NOT a 0-1 float).
        legend = {str(i): label for i, label in enumerate(question.criteria)}
        probabilities = {str(i): float(probs_list[i]) for i in range(len(keys))}
        score = float(sum(i * probabilities[str(i)] for i in range(len(keys))))
        return ScoreAnswer(
            score=score,
            legend=legend,
            probabilities=probabilities,
            confidence=confidence,
        )

    # Noul: probability of "true" (yes). No confidence field in the Jev contract.
    assert isinstance(question, NoulQuestion)
    true_idx = keys.index("true") if "true" in keys else 1
    return NoulAnswer(noul=float(probs_list[true_idx]))
