# jev-api

Self-hosted **Jev / TypeSafe System One** compatible HTTP API.

Four engines, switched per request with the JSON `model` field:

| `model` | Engine | Weights |
| --- | --- | --- |
| **`von`** (default) | [Von OptionMarker 395M](https://github.com/wfzyx/von) | `wfzyx/von-1.0` |
| **`laya`** | Official [Laya](https://github.com/NandhaKishorM/laya) Python CPU runtime | `convaiinnovations/laya` |
| **`semif`** | [SemIf](https://github.com/TheoLeeCJ/SemIf) option-logit readout | `Qwen/Qwen3.5-4B` (GPU BF16) or Q4 GGUF (CPU) |
| **`autojev`** | Remote [AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) via `autojev-serve` | HTTP only — not loaded here |

> Protocol compatibility ≠ model identity. Same `/v1/systemone` shapes as Jev.

```text
Client  →  POST /v1/systemone  →  jev-api
              model=von        →  Von 395M
              model=laya       →  Laya 421M
              model=semif      →  SemIf Qwen3.5-4B
              model=autojev    →  AUTOJEV_BASE_URL /v1/systemone
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
- **SemIf:** `semif`, plus aliases `semif-latest`, `semif-phase1`, `openjev`
- **AutoJev:** `autojev`, plus aliases `autojev-27b`, `autojev-latest` — requires `AUTOJEV_BASE_URL`

`/ready` stays Von-based so existing deploys keep working. Extra engines download on first use (`PRELOAD_LAYA` / `PRELOAD_SEMIF` to load at startup). AutoJev is never loaded in-process.

## Weights

- Von **`wfzyx/von-1.0`** (~1.5 GB) — loaded at startup into `MODEL_CACHE_DIR`
- Laya **`convaiinnovations/laya`** (~0.8 GB) — lazy-loaded unless `PRELOAD_LAYA=true`
- SemIf **`Qwen/Qwen3.5-4B`** — GPU/MPS uses official SemIf torch (BF16 ~8 GB). CPU uses llama.cpp + `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~3 GB). Lazy-loaded unless `PRELOAD_SEMIF=true`. Install `pip install -e ".[semif]"` for the CPU path.
- AutoJev **`denis-pplx/autojev-27b`** — official BF16 is ~49 GiB plus runtime overhead. It does **not** fit the Modal T4 (16 GB) and is not loaded in this container. Point `AUTOJEV_BASE_URL` at a separate `autojev-serve` on an ~80 GB GPU if you want to benchmark it.

Persist `/data/models` across restarts. Do not keep several engines resident on an 8 GiB VPS; switch one at a time or unload between benchmarks. SemIf BF16 will not fit next to Von on that box.

## Local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
# optional CPU SemIf (llama.cpp):
# pip install -e ".[semif]"
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

# Laya / SemIf / AutoJev (same payload, only model changes)
curl ... -d '{ "model": "laya", "state": "...", "questions": { ... } }'
curl ... -d '{ "model": "semif", "state": "...", "questions": { ... } }'
curl ... -d '{ "model": "autojev", "state": "...", "questions": { ... } }'
```

## Benchmark

```bash
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model von --n 20
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model laya --n 20
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model semif --n 5
# AutoJev only after AUTOJEV_BASE_URL points at autojev-serve:
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model autojev --n 5 --timeout 180
PYTHONPATH=src python scripts/compare_models.py --models autojev --limit 20 --repeats 1 --warmup 1 --out /tmp/compare-autojev.json

# Sequential in-process compare on 100 public JevBench decisions (not full JevBench score)
PYTHONPATH=src python scripts/compare_models.py --limit 100 --repeats 1 --warmup 1 --out /tmp/compare-100.json
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
| `SEMIF_HF_REPO` | `Qwen/Qwen3.5-4B` | SemIf reference tokenizer / torch model |
| `SEMIF_REVISION` | pinned Qwen3.5-4B commit | Required by official SemIf |
| `SEMIF_BACKEND` | `auto` | `torch` (CUDA/MPS) or `llamacpp` (CPU GGUF) |
| `SEMIF_MODE` | `auto` | `direct` per question, `shared` prefix reuse |
| `SEMIF_GGUF_REPO` / `SEMIF_GGUF_FILE` | bartowski Q4_K_M | CPU weights |
| `PRELOAD_SEMIF` | `false` | Load SemIf at startup |
| `AUTOJEV_BASE_URL` | — | `autojev-serve` origin; required for `model=autojev` |
| `AUTOJEV_API_KEY` | — | Optional upstream Bearer key |
| `AUTOJEV_TIMEOUT` | `120` | Seconds for the upstream System One call |
| `HF_TOKEN` | — | Optional |
| `MODEL_CACHE_DIR` | `/data/models` | Persist volume |
| `JEV_DEVICE` | `cpu` | |

## Resources

Von ~**395M / ~1.5 GB**. Laya ~**421M / ~0.8 GB**. SemIf Q4 GGUF ~**3 GB** (BF16 ~8 GB, GPU). AutoJev-27B ~**49 GiB BF16** — remote only, not the Modal T4. Coolify memory limit **8 GiB**, 1 Uvicorn worker. Leave extra engines lazy.

## AutoJev / T4 (EN)

[AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) is a Qwen3.8-27B decision model (published 84.60% vs Jev 82.79%). Official serve needs ~49 GiB of BF16 weights plus overhead — an 80GB-class GPU, not the Modal **T4 (16 GB)**. A 27B forward pass will also be slower than Von 395M / Laya 421M / SemIf 4B on that card. This API therefore only **proxies** `model=autojev` to `AUTOJEV_BASE_URL`. Without that URL, the request returns 503.

## AutoJev / T4 (FR)

[AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) est un modèle de décision Qwen3.8-27B (84,60 % publié vs Jev 82,79 %). Le serveur officiel demande ~49 Gio de poids BF16 plus l’overhead — GPU classe 80 Go, pas la **T4 Modal (16 Go)**. Un passage avant 27B sera aussi plus lent que Von 395M / Laya 421M / SemIf 4B sur cette carte. L’API **proxifie** seulement `model=autojev` vers `AUTOJEV_BASE_URL`. Sans cette URL, la requête renvoie 503.

## Tests

```bash
pytest -q
```
