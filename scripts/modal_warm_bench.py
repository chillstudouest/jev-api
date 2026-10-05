#!/usr/bin/env python3
"""Warm Modal benchmark: 50 distinct prompts, sequential, single model."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

BASE_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments and refunds",
            "technical": "Bugs and outages",
            "support": "General account help",
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
}

STATES = [
    "Charged twice for September and cancelling Friday unless refunded.",
    "App crashes on login after the iOS 18 update; screenshot attached.",
    "Invoice PDF download returns 500 for workspace admin only.",
    "Need to pause subscription for 2 months while renovating the shop.",
    "Delivery delayed 11 days; customer wants partial refund plus credit.",
    "Webhook signatures failing since midnight UTC; all partners broken.",
    "Staff member left; revoke SSO and rotate API keys today.",
    "Quote Q-2026-0142 accepted but job still shows draft on the map.",
    "Kitchen fit: 3.2m run, 3rd floor no lift, deadline 15 October.",
    "Bathroom suite quote contested — competitor is 12% cheaper HT.",
    "Client says call-out fee was never disclosed on the phone booking.",
    "Scheduler double-booked two plumbers on the same site Tuesday AM.",
    "Catalogue price for KIT-WORKQ looks wrong vs last accepted quote.",
    "Customer uploaded blurry photos; cannot size the splashback.",
    "VAT number missing on PDF; French B2B client refuses to pay.",
    "Enquiry from Instagram DM: urgent leak under kitchen sink tonight.",
    "Convert accepted quote to invoice but line tax rates do not match.",
    "Travel fee disputed — site is 4 km from depot, not 40.",
    "AI suggested SKIP but client has a communal waste room already.",
    "Need overtime LAB-OT hours to hit Friday handover, confirm margin.",
    "Payment link expired; client paid via bank transfer with wrong ref.",
    "Two enquiries same address — merge clients or keep separate jobs?",
    "Scaffold hire quote missing SCAFF line; risk if we send as-is.",
    "Customer wants exclusive tax mode for comparison with rival HT total.",
    "Job note: asbestos survey pending; should we delay quote validation?",
    "Email send failed (Brevo 400); status flipped to sent incorrectly.",
    "Duplicate quote created by mistake; cancel the newer draft only.",
    "Client upgraded from laminate to quartz mid-thread; recalculate pack.",
    "Night alarm: API latency p95 above 8s on /v1/systemone production.",
    "GDPR deletion request for closed job with attached site photos.",
    "Supplier delay on BTH-SHOWER; propose alternate enclosure code.",
    "Card declined for deposit; keep job reserved or release slot?",
    "Map pin wrong postcode; crew went to neighbour's house this morning.",
    "Warranty claim 14 months after install; policy is 12 months labour.",
    "Multi-site facility manager wants one quote covering 6 bathrooms.",
    "Cash customer refuses digital signature; ok to mark accepted manually?",
    "Insurance surveyor needs itemised labour vs materials split on PDF.",
    "Promo: waive CALL-OUT for first-time web enquiries this week only.",
    "Spanish client, invoice in EUR, terms currently English UK copy.",
    "Internal note: margin under 18% if we match competitor — escalate.",
    "Photo shows water damage behind vanity; add tanking line or not?",
    "Scheduler conflict: same van booked for roof and kitchen same day.",
    "Client asked for inclusive prices; switch taxMode and regenerate PDF.",
    "Chargeback opened for invoice INV-2026-0088; freeze related job.",
    "New apprentice on site — do we still bill full KIT-FIT day rate?",
    "Lead from partner portal has no phone; email-only follow-up path.",
    "Skip permit not approved by council; switch to WASTE bags plan.",
    "Customer wants weekend install only; apply LAB-OT and confirm urgency.",
    "Quote expired yesterday; client replied today — revive or new draft?",
    "Dashboard shows ready=false after deploy; poll /ready before traffic.",
]


def post_systemone(
    base_url: str, api_key: str, model: str, state: str, timeout: float
) -> dict[str, Any]:
    payload = {
        "state": state,
        "model": model,
        "questions": BASE_QUESTIONS,
    }
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
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base_url: str, timeout: float, attempts: int = 60) -> None:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/ready", timeout=timeout
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("ready"):
                print(f"# ready={data}", file=sys.stderr)
                return
            print(f"# waiting ready attempt={i+1} body={data}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"# ready error attempt={i+1}: {exc}", file=sys.stderr)
        time.sleep(5)
    raise RuntimeError("Modal endpoint never became ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gliner-von", choices=["gliner-von", "jev"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--csv-out", default="-")
    args = parser.parse_args()

    if args.n > len(STATES):
        raise SystemExit(f"--n {args.n} > available states {len(STATES)}")

    wait_ready(args.base_url, args.timeout)

    print(f"# warmup model={args.model} n={args.warmup}", file=sys.stderr)
    for i in range(args.warmup):
        try:
            body = post_systemone(
                args.base_url, args.api_key, args.model, STATES[i], args.timeout
            )
            print(
                f"# warmup {i+1}/{args.warmup} duration_ms={body.get('duration_ms')} "
                f"inference_ms={body.get('inference_ms')}",
                file=sys.stderr,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"warmup failed HTTP {exc.code}: {detail}") from exc

    rows: list[tuple[int, int, float]] = []
    print(f"# bench model={args.model} n={args.n}", file=sys.stderr)
    for i in range(args.n):
        state = STATES[i]
        try:
            body = post_systemone(
                args.base_url, args.api_key, args.model, state, args.timeout
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"req {i+1} failed HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"req {i+1} failed: {exc}") from exc

        input_tokens = int(body.get("input_tokens") or body.get("usage", {}).get("input_tokens") or 0)
        duration_ms = float(body["duration_ms"])
        rows.append((i + 1, input_tokens, duration_ms))
        print(
            f"# {i+1}/{args.n} tokens={input_tokens} duration_ms={duration_ms} "
            f"inference_ms={body.get('inference_ms')}",
            file=sys.stderr,
        )

    lines = ["index,input_tokens,duration_ms"]
    lines.extend(f"{idx},{tok},{dur}" for idx, tok, dur in rows)
    text = "\n".join(lines) + "\n"
    if args.csv_out == "-":
        sys.stdout.write(text)
    else:
        with open(args.csv_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"# wrote {args.csv_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
