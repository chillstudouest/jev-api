#!/usr/bin/env python3
"""Sequential von / laya / agent-jev benchmark on identical System One requests."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from jev_api.config import Settings
from jev_api.engine import VerdictEngine
from jev_api.schemas import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest, resolve_engine

CASES: list[dict[str, Any]] = [
    {
        "id": "billing_triage",
        "state": "Charged twice for September and cancelling Friday unless refunded.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Bugs and outages",
                },
            },
            "urgency": {
                "type": "noul",
                "instructions": "Does this need a reply today?",
                "criteria": {"true": "Time-sensitive", "false": "Can wait"},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": ["Low", "Medium", "High", "Critical"],
            },
        },
        "expected": {
            "route": "billing",
            "urgency_min": 0.5,
            "severity_min": 1.0,
        },
    },
    {
        "id": "outage",
        "state": "Production API latency is 3x baseline for one region; no data loss reported.",
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "How severe is this incident?",
                "criteria": ["Low", "Medium", "High", "Critical"],
            },
            "page_now": {
                "type": "noul",
                "instructions": "Should on-call be paged immediately?",
                "criteria": {"true": "Page now", "false": "Can wait for business hours"},
            },
        },
        "expected": {
            "severity_min": 1.0,
            "page_now_min": 0.4,
        },
    },
    {
        "id": "quote_request",
        "state": "Customer wants a bathroom renovation quote.",
        "questions": {
            "next": {
                "type": "choice",
                "instructions": "What should happen next?",
                "criteria": {
                    "create_client": "Create a client",
                    "create_job": "Create a job",
                    "create_quote": "Create a quote",
                },
            }
        },
        "expected": {"next": "create_quote"},
    },
]


def _summarize_answers(answers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, ans in answers.items():
        if isinstance(ans, ChoiceAnswer):
            out[qid] = {"type": "choice", "choice": ans.choice, "confidence": ans.confidence}
        elif isinstance(ans, ScoreAnswer):
            out[qid] = {"type": "score", "score": ans.score, "confidence": ans.confidence}
        elif isinstance(ans, NoulAnswer):
            out[qid] = {"type": "noul", "noul": ans.noul}
        else:
            out[qid] = ans.model_dump()
    return out


def _quality(case: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected", {})
    checks: dict[str, bool] = {}
    if "route" in expected and "route" in answers and isinstance(answers["route"], ChoiceAnswer):
        checks["route_billing"] = answers["route"].choice == expected["route"]
    if "next" in expected and "next" in answers and isinstance(answers["next"], ChoiceAnswer):
        checks["next_quote"] = answers["next"].choice == expected["next"]
    if "urgency_min" in expected and "urgency" in answers and isinstance(answers["urgency"], NoulAnswer):
        checks["urgency_high"] = answers["urgency"].noul >= float(expected["urgency_min"])
    if "severity_min" in expected and "severity" in answers and isinstance(answers["severity"], ScoreAnswer):
        checks["severity_raised"] = answers["severity"].score >= float(expected["severity_min"])
    if "page_now_min" in expected and "page_now" in answers and isinstance(answers["page_now"], NoulAnswer):
        checks["page_now"] = answers["page_now"].noul >= float(expected["page_now_min"])
    return {
        "passed": sum(1 for ok in checks.values() if ok),
        "total": len(checks),
        "checks": checks,
    }


def run_model(engine: VerdictEngine, model: str, repeats: int, warmup: int) -> dict[str, Any]:
    engine_name = resolve_engine(model)
    if engine_name == "von":
        engine.load_blocking()
    elif engine_name == "laya":
        engine.ensure_laya()
    else:
        engine.ensure_agentjev()

    latencies: list[float] = []
    case_rows: list[dict[str, Any]] = []
    quality_scores: list[float] = []

    for case in CASES:
        request = SystemOneRequest(state=case["state"], questions=case["questions"], model=model)
        for _ in range(warmup):
            engine.systemone(request)
        last = None
        times: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            last = engine.systemone(request)
            times.append((time.perf_counter() - started) * 1000.0)
        assert last is not None
        latencies.extend(times)
        quality = _quality(case, last.answers)
        if quality["total"]:
            quality_scores.append(quality["passed"] / quality["total"])
        case_rows.append(
            {
                "id": case["id"],
                "duration_ms": last.duration_ms,
                "client_ms": {
                    "p50": round(statistics.median(times), 2),
                    "mean": round(statistics.fmean(times), 2),
                },
                "answers": _summarize_answers(last.answers),
                "quality": quality,
            }
        )

    latencies.sort()
    return {
        "model": model,
        "engine": engine_name,
        "repeats_per_case": repeats,
        "cases": case_rows,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "mean": round(statistics.fmean(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "quality_rate": round(statistics.fmean(quality_scores), 3) if quality_scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="von,laya,agent-jev")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-dir", default="./data/models")
    parser.add_argument("--out", default="")
    parser.add_argument("--unload", action="store_true", default=True)
    parser.add_argument("--keep-loaded", action="store_true")
    args = parser.parse_args()

    settings = Settings(
        jev_api_key="compare",
        download_on_startup=False,
        model_cache_dir=Path(args.cache_dir),
    )
    engine = VerdictEngine(settings)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    for model in models:
        print(f"=== {model} ===", flush=True)
        row = run_model(engine, model, args.repeats, args.warmup)
        results.append(row)
        print(json.dumps({"model": model, "latency_ms": row["latency_ms"], "quality_rate": row["quality_rate"]}))
        if not args.keep_loaded:
            engine.unload(resolve_engine(model))

    report = {"models": results}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
