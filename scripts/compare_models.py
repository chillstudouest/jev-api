#!/usr/bin/env python3
"""Sequential engine benchmark on identical System One requests.

Default suite: first N public JevBench decisions (easy + original), MIT.
Not a full JevBench score (no hard tier / calibration / cost axes).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from jev_api.config import Settings
from jev_api.engine import VerdictEngine
from jev_api.schemas import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest, resolve_engine

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


def run_model(
    engine: VerdictEngine,
    model: str,
    cases: list[dict[str, Any]],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    engine_name = resolve_engine(model)
    if engine_name == "von":
        engine.load_blocking()
    elif engine_name == "laya":
        engine.ensure_laya()
    elif engine_name == "semif":
        engine.ensure_semif()
    elif engine_name == "autojev":
        engine.ensure_autojev()
    else:
        raise ValueError(f"Unknown engine {engine_name!r}")

    latencies: list[float] = []
    case_rows: list[dict[str, Any]] = []
    quality_scores: list[float] = []
    passed = 0
    total = 0

    if warmup > 0 and cases:
        warm = SystemOneRequest(state=cases[0]["state"], questions=cases[0]["questions"], model=model)
        for _ in range(warmup):
            engine.systemone(warm)

    for index, case in enumerate(cases, start=1):
        request = SystemOneRequest(state=case["state"], questions=case["questions"], model=model)
        last = None
        times: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            last = engine.systemone(request)
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
    parser.add_argument("--models", default="von,laya,semif")
    parser.add_argument("--limit", type=int, default=100, help="Number of JevBench public decisions")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-dir", default="./data/models")
    parser.add_argument("--jevbench-dir", default="./data/jevbench")
    parser.add_argument("--out", default="")
    parser.add_argument("--keep-loaded", action="store_true")
    args = parser.parse_args()

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
        row = run_model(engine, model, cases, args.repeats, args.warmup)
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
        if not args.keep_loaded:
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
            "Not the official JevBench Score (missing hard/calibration/speed/cost axes)."
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
