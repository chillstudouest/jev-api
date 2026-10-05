#!/usr/bin/env python3
"""Bench a new engine against the frozen Jev / Von baselines.

Jev and Von are never re-run: their numbers live in benchmarks/baselines.json.
Runs the public JevBench decisions and benchmarks/support_ticket.json, either
in-process (default) or against a deployed jev-api (--url).

    PYTHONPATH=src python scripts/bench_new_model.py --model gliner-von --label onnx-fp32-mac
    PYTHONPATH=src python scripts/bench_new_model.py --model gliner-von --url https://jev-api.codiku.com --label onnx-fp32-vps
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from compare_models import _quality, load_cases  # noqa: E402

from jev_api.engine import answers_from_mapping  # noqa: E402
from jev_api.schemas import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, resolve_engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"

Call = Callable[[object, dict[str, Any]], tuple[dict[str, Answer], float]]


def _remote_call(url: str, key: str, model: str, timeout: float) -> Call:
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    def call(state: object, questions: dict[str, Any]) -> tuple[dict[str, Answer], float]:
        body = json.dumps({"state": state, "questions": questions, "model": model}).encode("utf-8")
        request = urllib.request.Request(
            f"{url.rstrip('/')}/v1/systemone",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            data = json.loads(response.read().decode("utf-8"))
        return answers_from_mapping(data["answers"], source=model), (time.perf_counter() - started) * 1000.0

    return call


def _local_call(model: str) -> Call:
    from jev_api.config import Settings
    from jev_api.engine import VerdictEngine
    from jev_api.schemas import SystemOneRequest

    engine = VerdictEngine(Settings(download_on_startup=False))
    if resolve_engine(model) == "gliner-von":
        engine.load_blocking()
        engine.ensure_glinner()

    def call(state: object, questions: dict[str, Any]) -> tuple[dict[str, Answer], float]:
        started = time.perf_counter()
        response = engine.systemone(SystemOneRequest(state=state, questions=questions, model=model))
        return response.answers, (time.perf_counter() - started) * 1000.0

    return call


def _short(answer: Answer) -> str | float:
    if isinstance(answer, ChoiceAnswer):
        return f"{answer.choice} ({max(answer.probabilities.values()):.2f})"
    if isinstance(answer, ScoreAnswer):
        top = max(answer.probabilities, key=answer.probabilities.get) if answer.probabilities else "?"
        return f"{answer.score:.2f} (argmax {top})"
    if isinstance(answer, NoulAnswer):
        return round(answer.noul, 2)
    return str(answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="Result name, e.g. onnx-fp32-vps")
    parser.add_argument("--url", default="", help="Deployed jev-api; in-process when empty")
    parser.add_argument("--key", default=os.environ.get("JEV_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ticket-runs", type=int, default=10)
    parser.add_argument("--skip-jevbench", action="store_true")
    args = parser.parse_args()

    if resolve_engine(args.model) == "jev":
        raise SystemExit("jev is a frozen baseline (benchmarks/baselines.json); bench a new engine instead.")

    call = _remote_call(args.url, args.key, args.model, args.timeout) if args.url else _local_call(args.model)
    baselines = json.loads((BENCH_DIR / "baselines.json").read_text(encoding="utf-8"))
    ticket = json.loads((BENCH_DIR / "support_ticket.json").read_text(encoding="utf-8"))

    started = time.perf_counter()
    answers, _ = call(ticket["state"], ticket["questions"])
    first_call_ms = (time.perf_counter() - started) * 1000.0
    runs = [call(ticket["state"], ticket["questions"]) for _ in range(args.ticket_runs)]
    ticket_ms = sorted(ms for _, ms in runs)
    result: dict[str, Any] = {
        "model": args.model,
        "label": args.label,
        "target": args.url or "in-process",
        "measured": date.today().isoformat(),
        "first_call_ms": round(first_call_ms),
        "support_ticket": {
            "client_p50_ms": round(statistics.median(ticket_ms)),
            "client_max_ms": round(ticket_ms[-1]),
            "answers": {qid: _short(answer) for qid, answer in runs[-1][0].items()},
        },
    }

    if not args.skip_jevbench:
        cases = load_cases(1000, ROOT / "data" / "jevbench")
        passed_ids: list[str] = []
        by_type: dict[str, list[int]] = {}
        latencies: list[float] = []
        for case in cases:
            case_answers, ms = call(case["state"], case["questions"])
            latencies.append(ms)
            quality = _quality(case, case_answers)
            kind = str(case["expected"]["type"])
            by_type.setdefault(kind, [0, 0])
            by_type[kind][0] += quality["passed"]
            by_type[kind][1] += 1
            if quality["ok"]:
                passed_ids.append(case["id"])
        latencies.sort()
        result["jevbench"] = {
            "cases": len(cases),
            "accuracy": round(len(passed_ids) / len(cases), 4),
            "by_type": by_type,
            "client_p50_ms": round(statistics.median(latencies), 1),
            "client_p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 1),
            "passed_ids": passed_ids,
        }

    out = BENCH_DIR / "results" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved {out.relative_to(ROOT)}\n")
    print("JevBench             accuracy  p50 ms")
    for name in ("jev", "von"):
        bench = baselines[name].get("jevbench")
        if bench:
            print(f"  {name:<18} {bench['accuracy']:>7.1%}  {bench['client_p50_ms']:>6.0f}")
        else:
            print(f"  {name:<18}     n/a")
    if "jevbench" in result:
        bench = result["jevbench"]
        print(f"  {args.label:<18} {bench['accuracy']:>7.1%}  {bench['client_p50_ms']:>6.0f}")
    print("\nSupport ticket       p50 ms  answers")
    for name in ("jev", "von"):
        row = baselines[name]["support_ticket"]
        print(f"  {name:<18} {row['client_p50_ms']:>6}  {row['answers']}")
    row = result["support_ticket"]
    print(f"  {args.label:<18} {row['client_p50_ms']:>6}  {row['answers']}")


if __name__ == "__main__":
    main()
