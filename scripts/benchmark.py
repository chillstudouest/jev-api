#!/usr/bin/env python3
"""Benchmark jev-api /v1/decide latency and throughput.

Examples:
  python scripts/benchmark.py --base-url http://127.0.0.1:8000 --api-key "$JEV_API_KEY" --n 500
  python scripts/benchmark.py --base-url https://jev-api.codiku.com --api-key "$JEV_API_KEY" --n 200
"""

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
    "type": "workflow",
    "question": "What should happen next?",
    "state": "The customer wants to renovate their bathroom and asks for a quote.",
    "options": [
        {"id": "create_client", "description": "Create a client"},
        {"id": "create_job", "description": "Create a job"},
        {"id": "create_quote", "description": "Create a quote"},
    ],
}


def request_once(base_url: str, api_key: str, timeout: float) -> tuple[float, dict[str, Any]]:
    body = json.dumps(PAYLOAD).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/decide",
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
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, payload


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
    parser = argparse.ArgumentParser(description="Benchmark jev-api decide endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--n", type=int, default=200, help="Timed requests after warmup")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    # Cold /ready probe
    ready_url = f"{args.base_url.rstrip('/')}/ready"
    t0 = time.perf_counter()
    with urllib.request.urlopen(ready_url, timeout=args.timeout) as resp:
        ready = json.loads(resp.read().decode("utf-8"))
    ready_ms = (time.perf_counter() - t0) * 1000.0
    print(f"ready probe: {ready_ms:.1f} ms -> {ready}")

    print(f"warmup ({args.warmup})...")
    for _ in range(args.warmup):
        request_once(args.base_url, args.api_key, args.timeout)

    print(f"running {args.n} requests concurrency={args.concurrency}...")
    latencies: list[float] = []
    model_latencies: list[float] = []
    errors = 0
    wall_start = time.perf_counter()

    def work(_: int) -> None:
        nonlocal errors
        try:
            client_ms, payload = request_once(args.base_url, args.api_key, args.timeout)
            latencies.append(client_ms)
            model_latencies.append(float(payload.get("latency_ms", client_ms)))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors += 1
            print(f"error: {exc}")

    if args.concurrency <= 1:
        for i in range(args.n):
            work(i)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, i) for i in range(args.n)]
            for fut in as_completed(futures):
                fut.result()

    wall = time.perf_counter() - wall_start
    latencies.sort()
    model_latencies.sort()
    report = {
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
        "model_latency_ms": {
            "p50": round(percentile(model_latencies, 50), 2),
            "p95": round(percentile(model_latencies, 95), 2),
            "p99": round(percentile(model_latencies, 99), 2),
            "mean": round(statistics.fmean(model_latencies), 2) if model_latencies else None,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
