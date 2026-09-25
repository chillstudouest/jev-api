#!/usr/bin/env python3
"""Sequential engine benchmark on identical System One requests.

Default suite: first N public JevBench decisions (easy + original), MIT.
Not a full JevBench score (no hard tier / calibration / cost axes).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jev_api.config import Settings
from jev_api.engine import VerdictEngine, answers_from_mapping
from jev_api.schemas import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    resolve_engine,
)

REMOTE_JEV_ALIASES = {"jev", "jev-official", "typesafe", "typesafe-jev"}
DEFAULT_TYPESAFE_BASE = "https://api.typesafe.ai"
DEFAULT_TYPESAFE_MODEL = "jev-1.13.0"

JEVBENCH_SOURCES = (
    (
        "easy.jsonl",
        "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/easy.jsonl",
    ),
    (
        "original.jsonl",
        "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/original.jsonl",
    ),
)


def _download_jevbench(cache_dir: Path) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, url in JEVBENCH_SOURCES:
        path = cache_dir / name
        if not path.is_file():
            print(f"Downloading {name}…", flush=True)
            with urllib.request.urlopen(url, timeout=60) as resp:
                path.write_bytes(resp.read())
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _noul_criteria(raw: dict[str, Any]) -> dict[str, str | None]:
    criteria = raw.get("criteria")
    if isinstance(criteria, dict) and set(criteria.keys()) == {"true", "false"}:
        return {
            "true": criteria.get("true") if isinstance(criteria.get("true"), str) else None,
            "false": criteria.get("false") if isinstance(criteria.get("false"), str) else None,
        }
    # Some rows use yes/no label names; map to System One noul.
    return {"true": "Yes", "false": "No"}


def jevbench_row_to_case(row: dict[str, Any]) -> dict[str, Any]:
    qid = "q"
    question = dict(row["question"])
    qtype = str(question.get("type", "")).strip().lower()
    if qtype == "noul":
        question["criteria"] = _noul_criteria(question)
    case: dict[str, Any] = {
        "id": str(row["id"]),
        "family": row.get("family"),
        "state": row["state"],
        "questions": {qid: question},
        "expected": {"qid": qid, "type": qtype, "value": row.get("expected")},
    }
    return case


def load_cases(limit: int, cache_dir: Path) -> list[dict[str, Any]]:
    rows = _download_jevbench(cache_dir)
    cases = [jevbench_row_to_case(row) for row in rows[:limit]]
    if len(cases) < limit:
        print(f"Warning: only {len(cases)} public rows available (requested {limit})", flush=True)
    return cases


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


def _choice_from_score(ans: ScoreAnswer) -> str:
    if ans.probabilities:
        return max(ans.probabilities, key=ans.probabilities.get)  # type: ignore[arg-type]
    return str(int(round(ans.score)))


def _quality(case: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") or {}
    qid = str(expected.get("qid", "q"))
    qtype = str(expected.get("type", ""))
    value = expected.get("value")
    ans = answers.get(qid)
    ok = False
    detail = ""
    if qtype == "choice" and isinstance(ans, ChoiceAnswer):
        ok = ans.choice == str(value)
        detail = f"got={ans.choice} want={value}"
    elif qtype == "noul" and isinstance(ans, NoulAnswer):
        want_yes = str(value).strip().lower() in {"yes", "true", "1"}
        pred_yes = ans.noul >= 0.5
        ok = pred_yes == want_yes
        detail = f"noul={ans.noul:.3f} want={'yes' if want_yes else 'no'}"
    elif qtype == "score" and isinstance(ans, ScoreAnswer):
        pred = _choice_from_score(ans)
        ok = pred == str(value)
        detail = f"argmax={pred} score={ans.score:.3f} want={value}"
    else:
        detail = f"missing answer for {qid}/{qtype}"
    return {"passed": 1 if ok else 0, "total": 1, "ok": ok, "detail": detail}


def _call_typesafe(
    base_url: str,
    api_key: str,
    model: str,
    state: object,
    questions: dict[str, Any],
    timeout: float,
) -> SystemOneResponse:
    payload = {"state": state, "questions": questions, "model": model}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/systemone",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answers = answers_from_mapping(data.get("answers") or {}, source="Jev")
    usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage = Usage(
        input_tokens=int(usage_raw.get("input_tokens", 0) or 0),
        output_tokens=int(usage_raw.get("output_tokens", 0) or 0),
    )
    duration_ms = float(data.get("duration_ms") or 0.0)
    inference_ms = float(data.get("inference_ms") or duration_ms)
    return SystemOneResponse(
        model=str(data.get("model") or model),
        answers=answers,
        usage=usage,
        duration_ms=duration_ms,
        inference_ms=inference_ms,
        gpu_duration_ms=data.get("gpu_duration_ms"),
    )


def run_model(
    engine: VerdictEngine | None,
    model: str,
    cases: list[dict[str, Any]],
    repeats: int,
    warmup: int,
    *,
    typesafe_base: str,
    typesafe_key: str,
    typesafe_model: str,
    timeout: float,
) -> dict[str, Any]:
    remote = model in REMOTE_JEV_ALIASES
    if remote:
        if not typesafe_key:
            raise SystemExit(
                "model=jev needs TYPESAFE_API_KEY (official TypeSafe cloud). "
                "Put it in .env.local — JEV_API_KEY is only for self-hosted jev-api."
            )
        engine_name = "jev"
        call = lambda request: _call_typesafe(
            typesafe_base,
            typesafe_key,
            typesafe_model,
            request.state,
            request.questions,
            timeout,
        )
    else:
        assert engine is not None
        engine_name = resolve_engine(model)
        if engine_name == "von":
            engine.load_blocking()
        elif engine_name == "laya":
            engine.ensure_laya()
        elif engine_name == "semif":
            engine.ensure_semif()
        elif engine_name == "glinner":
            engine.ensure_glinner()
        else:
            raise ValueError(f"Unknown engine {engine_name!r}")
        call = engine.systemone

    latencies: list[float] = []
    case_rows: list[dict[str, Any]] = []
    quality_scores: list[float] = []
    passed = 0
    total = 0

    if warmup > 0 and cases:
        warm = SystemOneRequest(state=cases[0]["state"], questions=cases[0]["questions"], model=None if remote else model)
        for _ in range(warmup):
            call(warm)

    for index, case in enumerate(cases, start=1):
        request = SystemOneRequest(
            state=case["state"],
            questions=case["questions"],
            model=None if remote else model,
        )
        last = None
        times: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            last = call(request)
            times.append((time.perf_counter() - started) * 1000.0)
        assert last is not None
        latencies.extend(times)
        quality = _quality(case, last.answers)
        passed += quality["passed"]
        total += quality["total"]
        if quality["total"]:
            quality_scores.append(quality["passed"] / quality["total"])
        case_rows.append(
            {
                "id": case["id"],
                "family": case.get("family"),
                "duration_ms": last.duration_ms,
                "client_ms": {
                    "p50": round(statistics.median(times), 2),
                    "mean": round(statistics.fmean(times), 2),
                },
                "answers": _summarize_answers(last.answers),
                "quality": quality,
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(
                f"  [{model}] {index}/{len(cases)} "
                f"acc={passed / total:.3f} last_ms={times[-1]:.0f}",
                flush=True,
            )

    latencies.sort()
    return {
        "model": model,
        "engine": engine_name,
        "n_cases": len(cases),
        "repeats_per_case": repeats,
        "accuracy": round(passed / total, 4) if total else None,
        "passed": passed,
        "total_checks": total,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 2),
            "mean": round(statistics.fmean(latencies), 2),
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "quality_rate": round(statistics.fmean(quality_scores), 3) if quality_scores else None,
        "cases": case_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="jev,von,semif,glinner")
    parser.add_argument("--limit", type=int, default=100, help="Number of JevBench public decisions")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-dir", default="./data/models")
    parser.add_argument("--jevbench-dir", default="./data/jevbench")
    parser.add_argument("--out", default="")
    parser.add_argument("--keep-loaded", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--typesafe-base",
        default=os.environ.get("TYPESAFE_BASE_URL", DEFAULT_TYPESAFE_BASE),
    )
    parser.add_argument(
        "--typesafe-key",
        default=os.environ.get("TYPESAFE_API_KEY", ""),
    )
    parser.add_argument(
        "--typesafe-model",
        default=os.environ.get("TYPESAFE_MODEL", DEFAULT_TYPESAFE_MODEL),
    )
    args = parser.parse_args()

    # Load .env.local keys if present (without requiring python-dotenv).
    for env_path in (Path(".env.local"), Path(".env")):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if not args.typesafe_key:
        args.typesafe_key = os.environ.get("TYPESAFE_API_KEY", "")

    cases = load_cases(args.limit, Path(args.jevbench_dir))
    print(f"Loaded {len(cases)} cases", flush=True)

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
        row = run_model(
            engine,
            model,
            cases,
            args.repeats,
            args.warmup,
            typesafe_base=args.typesafe_base,
            typesafe_key=args.typesafe_key,
            typesafe_model=args.typesafe_model,
            timeout=args.timeout,
        )
        results.append(row)
        print(
            json.dumps(
                {
                    "model": model,
                    "n_cases": row["n_cases"],
                    "accuracy": row["accuracy"],
                    "latency_ms": row["latency_ms"],
                }
            ),
            flush=True,
        )
        if not args.keep_loaded and model not in REMOTE_JEV_ALIASES:
            engine.unload(resolve_engine(model))
            import gc

            gc.collect()
            try:
                import torch

                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    report = {
        "suite": "jevbench-public-prefix",
        "n_cases": len(cases),
        "note": (
            "Accuracy on first N public JevBench decisions (easy+original). "
            "Not the official JevBench Score (missing hard/calibration/speed/cost axes). "
            "model=jev hits TypeSafe cloud; others are local."
        ),
        "models": [
            {
                "model": row["model"],
                "engine": row["engine"],
                "n_cases": row["n_cases"],
                "accuracy": row["accuracy"],
                "passed": row["passed"],
                "total_checks": row["total_checks"],
                "latency_ms": row["latency_ms"],
                "cases": row["cases"],
            }
            for row in results
        ],
    }
    text = json.dumps(report, indent=2)
    summary = {
        "n_cases": len(cases),
        "models": [
            {
                "model": row["model"],
                "accuracy": row["accuracy"],
                "latency_ms": row["latency_ms"],
            }
            for row in results
        ],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
