#!/usr/bin/env python3
"""Smoke-check that /health, /ready and /v1/decide respond as expected."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def get(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = get(f"{base}/health")
    print("health", status, health)
    assert status == 200 and health["status"] == "ok"

    deadline = time.time() + args.wait_ready_s
    ready_body: dict = {}
    while time.time() < deadline:
        status, ready_body = get(f"{base}/ready")
        print("ready", status, ready_body)
        if ready_body.get("ready"):
            break
        if ready_body.get("status") == "error":
            print("model failed to load", ready_body, file=sys.stderr)
            return 1
        time.sleep(5)
    else:
        print("timed out waiting for ready", file=sys.stderr)
        return 1

    if not args.api_key:
        print("JEV_API_KEY missing; skipping /v1/decide", file=sys.stderr)
        return 0

    headers = {"Authorization": f"Bearer {args.api_key}"}
    payload = {
        "type": "workflow",
        "question": "What should happen next?",
        "state": "The customer wants to renovate their bathroom and asks for a quote.",
        "options": [
            {"id": "create_client", "description": "Create a client"},
            {"id": "create_job", "description": "Create a job"},
            {"id": "create_quote", "description": "Create a quote"},
        ],
    }
    try:
        status, decide = post(f"{base}/v1/decide", payload, headers)
    except urllib.error.HTTPError as exc:
        print("decide failed", exc.code, exc.read().decode(), file=sys.stderr)
        return 1
    print("decide", status, json.dumps(decide, indent=2))
    assert status == 200
    assert decide["choice"] in decide["scores"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
