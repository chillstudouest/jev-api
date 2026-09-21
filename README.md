# jev-api

Self-hosted **Jev / TypeSafe System One** compatible HTTP API.

**Engine:** [Von OptionMarker 395M](https://github.com/wfzyx/von) (`wfzyx/von-1.0`, public on Hugging Face).

> Protocol compatibility ≠ model identity. Same `/v1/systemone` shapes as Jev; predictions come from Von.

```text
Client  →  POST /v1/systemone  →  jev-api (auth + validation)  →  Von 395M  →  Jev-shaped answers
```

Prod: `https://jev-api.codiku.com`

## Endpoints

| Method | Path | Auth |
| --- | --- | --- |
| `POST` | `/v1/systemone` | Bearer `JEV_API_KEY` |
| `POST` | `/api/v1/systemone` | Bearer (alias) |
| `GET` | `/health` | no |
| `GET` | `/ready` | no |
| `GET` | `/v1/model` | Bearer |

Accepted `model`: `jev-latest`, `jev-preview`, `jev-1.13.0` (default), plus `von-*` aliases. All run Von OptionMarker.

## Weights (~1.5 GB)

Public checkpoint **`wfzyx/von-1.0`** — downloaded automatically at first start into `MODEL_CACHE_DIR` (no private HF access). Persist `/data/models` across restarts.

## Local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
cp .env.example .env   # set JEV_API_KEY
python -m uvicorn jev_api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Wait for `/ready` → `true` (first download ~1.5 GB).

## Example

```bash
curl -s https://jev-api.codiku.com/v1/systemone \
  -H "Authorization: Bearer $JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Charged twice for September and cancelling Friday unless refunded.",
    "model": "jev-1.13.0",
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
```

## Env

| Variable | Default | Notes |
| --- | --- | --- |
| `JEV_API_KEY` | — | Required |
| `VON_BACKEND` | `option-marker` | Von 395M |
| `HF_REPO` | `wfzyx/von-1.0` | Public |
| `HF_TOKEN` | — | Optional |
| `MODEL_CACHE_DIR` | `/data/models` | Persist volume |
| `JEV_DEVICE` | `cpu` | |

## Resources

~**395M params / ~1.5 GB** weights. Coolify memory limit **6 GiB**, 1 Uvicorn worker.

## Tests

```bash
pytest -q
```
