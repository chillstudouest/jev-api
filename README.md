# jev-api

Self-hosted **Jev / TypeSafe System One** compatible HTTP API.

Engines, switched per request with the JSON `model` field:

| `model` | Engine | Weights |
| --- | --- | --- |
| **`gliner-von`** (default) | `noul` → [Von OptionMarker 395M](https://github.com/wfzyx/von), `choice` / `score` → [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (ONNX on CPU), in one request | `wfzyx/von-1.0` + [`nishparadox/gliner2.5-decide-onnx`](https://huggingface.co/nishparadox/gliner2.5-decide-onnx) (fp32) |
| **`jev`** | Official TypeSafe Jev (proxy) | `api.typesafe.ai` (`TYPESAFE_API_KEY`) |

> Protocol compatibility ≠ model identity. Same `/v1/systemone` shapes as Jev.

```text
Client  →  POST /v1/systemone  →  jev-api
              model=gliner-von →  noul: Von · choice/score: Gliner
              model=jev        →  TypeSafe official Jev
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

Accepted `model` values:

- **Gliner + Von:** `gliner-von` (default), plus aliases `gliner-von-latest`, `jev-latest`, `jev-preview`, `jev-1.13.0`
- **Official Jev:** `jev`, plus aliases `jev-official`, `typesafe`, `typesafe-jev` — needs `TYPESAFE_API_KEY`

Any other value (including the retired `von` and `glinner`) returns `400 Unknown model`.

`/ready` reports Von. Gliner loads at startup too (`PRELOAD_GLINNER=true`, the default).

## Weights

- Von **`wfzyx/von-1.0`** (~1.5 GB) — loaded at startup into `MODEL_CACHE_DIR`
- Gliner **GLiNER2.5-Decide** (DeBERTa-v3-large) — every question is a head of the same forward pass. Loaded at startup (`PRELOAD_GLINNER=true`).
  - `GLINNER_BACKEND=onnx` (default): onnxruntime on the pinned [ONNX export](https://huggingface.co/nishparadox/gliner2.5-decide-onnx). `GLINNER_ONNX_VARIANT=fp32` (1.75 GB, default) gives the same answers as torch; `int8` (643 MB) is smaller but lossy.
  - `GLINNER_BACKEND=torch`: the `gliner2` library on `fastino/GLiNER2.5-Decide`, needed for CUDA/MPS.

  Backends on public JevBench (120 decisions) plus a 3- and 7-question request, M-series CPU limited to 4 threads, 2026-09-26:

  | Backend | Accuracy | Same answer as torch | p50 1 q | p50 3 q | p50 7 q | Peak RSS | Load |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | torch | 80.0 % | — | 275 ms | 286 ms | 489 ms | 2.6–3.7 GB | 17 s |
  | onnx fp32 | 80.0 % | 120/120 | 97 ms | 162 ms | 321 ms | 1.9 GB | 4 s |
  | onnx int8 | 77.5 % | 117/120 | 100 ms | 155 ms | 305 ms | 1.3 GB | 3 s |

  int8 is no faster than fp32 on Apple Silicon; it may be on the x86 VPS (not measured).

  Public JevBench (120 decisions, CPU, M-series, 2026-09-26):

  | Engine | Accuracy | choice | noul | score | p50 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | TypeSafe Jev (`jev-latest`, remote) | 99.2 % | 72/72 | 35/36 | 12/12 | 284 ms |
  | `gliner-von` (2026-09-27) | 84.2 % | 63/72 | 26/36 | 12/12 | 81 ms |
  | GLiNER2.5-Decide (onnx fp32, 2026-09-27) | 80.0 % | 63/72 | 21/36 | 12/12 | 73 ms |
  | Von | 79.2 % | 57/72 | 26/36 | 12/12 | 103 ms |
  | GLiClass base (removed) | 64.2 % | 55/72 | 18/36 | 4/12 | 71 ms |

  `noul` splits into `fact` (Von 12/12, Decide 10/12), `policy` (7/12, 5/12) and `adequacy` (7/12, 6/12): the last two need applying a rule or checking an answer, which neither encoder does. Decide is near chance on them; label wording (`true`/`false`, `yes`/`no`, descriptions as labels) makes no difference. The [Core ML port](https://huggingface.co/FluidInference/gliner2-5-decide-coreml) is Apple Silicon only and does not run on the VPS.

Persist `/data/models` across restarts. Von and Gliner (onnx fp32) together take ~3.5 GB on the 8 GiB VPS.

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
# gliner-von (default)
curl -s https://jev-api.codiku.com/v1/systemone \
  -H "Authorization: Bearer $JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Charged twice for September and cancelling Friday unless refunded.",
    "model": "gliner-von",
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

# Official Jev (same payload, only model changes)
curl ... -d '{ "model": "jev", "state": "...", "questions": { ... } }'
```

## Benchmark

```bash
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model gliner-von --n 20

# Sequential in-process compare on public JevBench decisions (--models jev,gliner-von by default)
PYTHONPATH=src python scripts/compare_models.py --models gliner-von --limit 120 --repeats 1 --warmup 1 --out /tmp/compare-120.json
```

First startup downloads the weights and may take minutes; later calls are in-process.

## Env

| Variable | Default | Notes |
| --- | --- | --- |
| `JEV_API_KEY` | — | Required |
| `VON_BACKEND` | `von` | Von 395M |
| `HF_REPO` | `wfzyx/von-1.0` | Von checkpoint |
| `GLINNER_BACKEND` | `onnx` | `onnx` (CPU) or `torch` |
| `GLINNER_ONNX_REPO` | `nishparadox/gliner2.5-decide-onnx` | ONNX export |
| `GLINNER_ONNX_REVISION` | pinned commit | ONNX export revision |
| `GLINNER_ONNX_VARIANT` | `fp32` | `fp32` or `int8` |
| `GLINNER_THREADS` | — | onnxruntime intra-op threads |
| `GLINNER_HF_REPO` | `fastino/GLiNER2.5-Decide` | torch checkpoint |
| `PRELOAD_GLINNER` | `true` | Load Gliner at startup (needed by `gliner-von`) |
| `TYPESAFE_API_KEY` | — | Needed for `model=jev` |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | Official Jev endpoint |
| `TYPESAFE_MODEL` | `jev-1.13.0` | Model sent upstream |
| `TYPESAFE_TIMEOUT` | `60` | Seconds |
| `HF_TOKEN` | — | Optional |
| `MODEL_CACHE_DIR` | `/data/models` | Persist volume |
| `JEV_DEVICE` | `cpu` | |

## Resources

Von ~**395M / ~1.5 GB**. Gliner onnx fp32 ~**1.9 GB** RSS (int8 ~1.3 GB). Coolify memory limit **8 GiB**, 1 Uvicorn worker.

## Benchmarks

Jev and Von are frozen baselines in `benchmarks/baselines.json` and are never re-run. Bench only new engines, in-process or against the deployed API:

```bash
MODEL_CACHE_DIR=data/models PYTHONPATH=src python scripts/bench_new_model.py --model gliner-von --label gliner-von-mac
PYTHONPATH=src python scripts/bench_new_model.py --model gliner-von --url https://jev-api.codiku.com --key "$JEV_API_KEY" --label gliner-von-vps
```

Each run covers the 120 public JevBench decisions and `benchmarks/support_ticket.json`, is saved to `benchmarks/results/<label>.json`, and prints next to the baselines.

## Tests

```bash
pytest -q
```
