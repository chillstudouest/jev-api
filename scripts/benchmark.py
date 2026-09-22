#!/usr/bin/env python3
"""Benchmark Jev-compatible POST /v1/systemone."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


PAYLOAD = {
    "state": "Charged twice for September and cancelling Friday unless refunded.",
    "model": "von",
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
}


def request_once(
    base_url: str, api_key: str, timeout: float, model: str
) -> tuple[float, dict[str, Any]]:
    payload = {**PAYLOAD, "model": model}
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
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return (time.perf_counter() - started) * 1000.0, payload


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model", default="von", help="von, laya, agent-jev, or semif")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/ready", timeout=args.timeout) as resp:
        print("ready", json.loads(resp.read().decode()))

    for _ in range(args.warmup):
        request_once(args.base_url, args.api_key, args.timeout, args.model)

    latencies: list[float] = []
    errors = 0
    wall_start = time.perf_counter()

    def work(_: int) -> None:
        nonlocal errors
        try:
            ms, _ = request_once(args.base_url, args.api_key, args.timeout, args.model)
            latencies.append(ms)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print("error", exc)

    if args.concurrency <= 1:
        for i in range(args.n):
            work(i)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(work, i) for i in range(args.n)]
            for fut in futs:
                fut.result()

    wall = time.perf_counter() - wall_start
    latencies.sort()
    print(
        json.dumps(
            {
                "model": args.model,
                "n": args.n,
                "errors": errors,
                "concurrency": args.concurrency,
                "wall_s": round(wall, 3),
                "throughput_rps": round((args.n - errors) / max(wall, 1e-9), 2),
                "client_latency_ms": {
                    "p50": round(percentile(latencies, 50), 2),
                    "p95": round(percentile(latencies, 95), 2),
                    "p99": round(percentile(latencies, 99), 2),
                    "mean": round(statistics.fmean(latencies), 2) if latencies else None,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
