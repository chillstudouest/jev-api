# jev-api

Self-hosted **Jev / TypeSafe System One** compatible HTTP API.

Two local engines, switched per request with the JSON `model` field:

| `model` | Engine | Weights |
| --- | --- | --- |
| **`von`** (default) | [Von OptionMarker 395M](https://github.com/wfzyx/von) | `wfzyx/von-1.0` |
| **`laya`** | Official [Laya](https://github.com/NandhaKishorM/laya) Python CPU runtime | `convaiinnovations/laya` |

> Protocol compatibility ≠ model identity. Same `/v1/systemone` shapes as Jev.

```text
Client  →  POST /v1/systemone  →  jev-api
              model=von   →  Von 395M
              model=laya  →  Laya 421M
```

Prod: `https://jev-api.codiku.com`

## Why not laya-mlx?

[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) is a native **Apple Silicon / MLX** port (macOS 14+, M-series). It does not run on a Debian CPU VPS.

This service uses the official `laya` package (PyTorch + Transformers, `device=cpu`). Same System One contract (`choice` / `score` / `noul`), works on the existing Coolify image.

A later win for RAM/CPU/latency on the VPS is the **Rust/Candle** runtime ([`laya` crate](https://github.com/aovestdipaperino/laya-rust)), not another Python stack. That is not wired in yet so the current Von path stays untouched.

## Endpoints

| Method | Path | Auth |
| --- | --- | --- |
| `POST` | `/v1/systemone` | Bearer `JEV_API_KEY` |
| `POST` | `/api/v1/systemone` | Bearer (alias) |
| `GET` | `/health` | no |
| `GET` | `/ready` | no |
| `GET` | `/v1/model` | Bearer |

Accepted `model` values:

- **Von:** `von` (default), plus aliases `von-latest`, `von-option-marker`, `jev-latest`, `jev-preview`, `jev-1.13.0`
- **Laya:** `laya`, plus aliases `laya-latest`, `laya-1.0`

`/ready` stays Von-based so existing deploys keep working. Laya downloads on the first `model=laya` request (or at startup if `PRELOAD_LAYA=true`).

## Weights

- Von **`wfzyx/von-1.0`** (~1.5 GB) — loaded at startup into `MODEL_CACHE_DIR`
- Laya **`convaiinnovations/laya`** (~0.8 GB) — lazy-loaded unless `PRELOAD_LAYA=true`

Persist `/data/models` across restarts. Loading both at once needs headroom (Coolify is set to 8 GiB). Prefer `model=von` only if RAM is tight.

## Local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
cp .env.example .env   # set JEV_API_KEY
python -m uvicorn jev_api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Wait for `/ready` → `true` (first Von download ~1.5 GB).

## Example

```bash
# Von (default)
curl -s https://jev-api.codiku.com/v1/systemone \
  -H "Authorization: Bearer $JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Charged twice for September and cancelling Friday unless refunded.",
    "model": "von",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Bugs and outages"
        }
      },
      "urgency": {
        "type": "noul",
        "instructions": "Does this need a reply today?",
        "criteria": { "true": "Time-sensitive", "false": "Can wait" }
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is this?",
        "criteria": ["Low", "Medium", "High", "Critical"]
      }
    }
  }'

# Laya (same payload, only model changes)
curl ... -d '{ "model": "laya", "state": "...", "questions": { ... } }'
```

## Benchmark

```bash
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model von --n 20
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model laya --n 20
```

First Laya call may download weights and take minutes; later calls are in-process.

## Env

| Variable | Default | Notes |
| --- | --- | --- |
| `JEV_API_KEY` | — | Required |
| `VON_BACKEND` | `von` | Von 395M |
| `HF_REPO` | `wfzyx/von-1.0` | Von checkpoint |
| `LAYA_HF_REPO` | `convaiinnovations/laya` | Official Laya |
| `LAYA_SUBFOLDER` | — | e.g. `typed-decisions` or `multilingual` |
| `PRELOAD_LAYA` | `false` | Load Laya at startup |
| `HF_TOKEN` | — | Optional |
| `MODEL_CACHE_DIR` | `/data/models` | Persist volume |
| `JEV_DEVICE` | `cpu` | |

## Resources

Von ~**395M / ~1.5 GB**. Laya English ~**421M / ~0.8 GB**. Coolify memory limit **8 GiB**, 1 Uvicorn worker. Leave `PRELOAD_LAYA=false` unless you need both hot.

## Tests

```bash
pytest -q
```
