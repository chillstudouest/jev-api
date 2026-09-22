# jev-api

Self-hosted **Jev / TypeSafe System One** compatible HTTP API.

Four local engines, switched per request with the JSON `model` field:

| `model` | Engine | Weights |
| --- | --- | --- |
| **`von`** (default) | [Von OptionMarker 395M](https://github.com/wfzyx/von) | `wfzyx/von-1.0` |
| **`laya`** | Official [Laya](https://github.com/NandhaKishorM/laya) Python CPU runtime | `convaiinnovations/laya` |
| **`agent-jev`** | [AgentJev-0.6B](https://huggingface.co/aimeigaoshou/agent-jev) | `aimeigaoshou/agent-jev` + `Qwen/Qwen3-0.6B` |
| **`semif`** | [SemIf](https://github.com/TheoLeeCJ/SemIf) option-logit readout | `Qwen/Qwen3.5-4B` (GPU BF16) or Q4 GGUF (CPU) |

> Protocol compatibility ≠ model identity. Same `/v1/systemone` shapes as Jev.

```text
Client  →  POST /v1/systemone  →  jev-api
              model=von        →  Von 395M
              model=laya       →  Laya 421M
              model=agent-jev  →  AgentJev 0.6B
              model=semif      →  SemIf Qwen3.5-4B
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
- **AgentJev:** `agent-jev`, plus aliases `agentjev`, `agent-jev-0.6b`
- **SemIf:** `semif`, plus aliases `semif-latest`, `semif-phase1`, `openjev`

`/ready` stays Von-based so existing deploys keep working. Extra engines download on first use (`PRELOAD_LAYA` / `PRELOAD_AGENTJEV` / `PRELOAD_SEMIF` to load at startup).

## Weights

- Von **`wfzyx/von-1.0`** (~1.5 GB) — loaded at startup into `MODEL_CACHE_DIR`
- Laya **`convaiinnovations/laya`** (~0.8 GB) — lazy-loaded unless `PRELOAD_LAYA=true`
- AgentJev **`aimeigaoshou/agent-jev`** (~1.2 GB) plus Qwen3-0.6B skeleton — lazy-loaded unless `PRELOAD_AGENTJEV=true`
- SemIf **`Qwen/Qwen3.5-4B`** — GPU/MPS uses official SemIf torch (BF16 ~8 GB). CPU uses llama.cpp + `Qwen_Qwen3.5-4B-Q4_K_M.gguf` (~3 GB). Lazy-loaded unless `PRELOAD_SEMIF=true`. Install `pip install -e ".[semif]"` for the CPU path.

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

# Laya / AgentJev / SemIf (same payload, only model changes)
curl ... -d '{ "model": "laya", "state": "...", "questions": { ... } }'
curl ... -d '{ "model": "semif", "state": "...", "questions": { ... } }'
```

## Benchmark

```bash
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model von --n 20
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model laya --n 20
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model agent-jev --n 20
python scripts/benchmark.py --api-key "$JEV_API_KEY" --model semif --n 5

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
| `AGENTJEV_HF_REPO` | `aimeigaoshou/agent-jev` | AgentJev weights |
| `AGENTJEV_BACKBONE` | `Qwen/Qwen3-0.6B` | Tokenizer / skeleton |
| `PRELOAD_AGENTJEV` | `false` | Load AgentJev at startup |
| `SEMIF_HF_REPO` | `Qwen/Qwen3.5-4B` | SemIf reference tokenizer / torch model |
| `SEMIF_REVISION` | pinned Qwen3.5-4B commit | Required by official SemIf |
| `SEMIF_BACKEND` | `auto` | `torch` (CUDA/MPS) or `llamacpp` (CPU GGUF) |
| `SEMIF_MODE` | `auto` | `direct` per question, `shared` prefix reuse |
| `SEMIF_GGUF_REPO` / `SEMIF_GGUF_FILE` | bartowski Q4_K_M | CPU weights |
| `PRELOAD_SEMIF` | `false` | Load SemIf at startup |
| `HF_TOKEN` | — | Optional |
| `MODEL_CACHE_DIR` | `/data/models` | Persist volume |
| `JEV_DEVICE` | `cpu` | |

## Resources

Von ~**395M / ~1.5 GB**. Laya ~**421M / ~0.8 GB**. AgentJev ~**598M / ~1.2 GB**. SemIf Q4 GGUF ~**3 GB** (BF16 ~8 GB, GPU). Coolify memory limit **8 GiB**, 1 Uvicorn worker. Leave extra engines lazy.

## Tests

```bash
pytest -q
```
