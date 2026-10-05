#!/usr/bin/env python3
"""Smoke-check /health, /ready and Jev-compatible /v1/systemone."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def get(url: str, timeout: float = 30.0) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def post(url: str, payload: dict, headers: dict[str, str], timeout: float = 60.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("JEV_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("JEV_API_KEY", ""))
    parser.add_argument("--wait-ready-s", type=int, default=600)
    parser.add_argument("--model", default="gliner-von", help="gliner-von or jev")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = get(f"{base}/health")
    print("health", status, health)
    assert status == 200 and health["status"] == "ok"

    deadline = time.time() + args.wait_ready_s
    while time.time() < deadline:
        status, ready = get(f"{base}/ready")
        print("ready", status, ready)
        if ready.get("ready"):
            break
        if ready.get("status") == "error":
            return 1
        time.sleep(5)
    else:
        return 1

    if not args.api_key:
        print("skip systemone (no API key)", file=sys.stderr)
        return 0

    payload = {
        "state": "Charged twice for September and cancelling Friday unless refunded.",
        "model": args.model,
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
        },
    }
    try:
        status, body = post(
            f"{base}/v1/systemone",
            payload,
            {"Authorization": f"Bearer {args.api_key}"},
        )
    except urllib.error.HTTPError as exc:
        print(exc.code, exc.read().decode(), file=sys.stderr)
        return 1
    print("systemone", status, json.dumps(body, indent=2))
    assert status == 200
    assert "route" in body["answers"] and "urgency" in body["answers"]
    assert body["answers"]["urgency"]["type"] == "noul"
    assert "confidence" not in body["answers"]["urgency"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
