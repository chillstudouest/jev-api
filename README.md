# jev-api

Production HTTP microservice for **openJev Verdict 2.0** decision inference.

Other apps send a typed question + options + state and receive calibrated choice probabilities and a correctness-head confidence score — exactly as produced by the official Verdict 2.0 checkpoint.

- Upstream: [Heman10x-NGU/openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0)
- Checkpoint: `heman10x/openJev-verdict-2.0` (Hugging Face)
- Public URL (prod): `https://jev-api.codiku.com`

## Architecture

| Layer | Choice | Why |
| --- | --- | --- |
| API | FastAPI + Uvicorn (1 worker) | Thin HTTP surface, async-friendly, low overhead |
| Inference | **PyTorch + Transformers (CPU)** | Official Verdict 2.0 runtime; bit-faithful to `verdict2/train.py` `infer()` |
| Backbone config/tokenizer | Vendored `vendor/modernbert/` | Offline, reproducible builds — no `answerdotai/ModernBERT-base` fetch at runtime |
| Weights | `snapshot_download` at **first container start** | Image builds without secrets/weights; `/ready` stays false until load completes |
| Concurrency | Single process + inference lock | Avoids N× model RAM copies from multi-worker Uvicorn |

ONNX was considered. There is no official lossless ONNX export of the Verdict 2.0 marker-pointer + correctness heads in the upstream repo that we verified as parity-safe, so the default path stays PyTorch.

### Model internals (Verdict 2.0)

Sequence (from upstream `verdict2/data.py` `build_item`):

```text
[CLS] {qtype} question: {instructions} [SEP] [MASK]{opt0} [MASK]{opt1} ... [SEP] {state} [SEP]
```

- `head_max_len=192`, `max_len=512`, per-option body capped at **48** tokens
- Marker hidden states → scorer MLP → logits
- `apply_temperature(logits, qtype, K, temperature[3×8])` → softmax → renormalize over real option width
- Correctness head: `sigmoid(CorrectnessHead(features(probs)))` → `confidence`
- qtypes: `choice=0`, `score=1`, `noul=2` — API alias `workflow` → `choice`

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | no | Liveness |
| `GET` | `/ready` | no | Model loaded and ready |
| `GET` | `/v1/model` | Bearer | Model / runtime metadata |
| `POST` | `/v1/decide` | Bearer | Run a decision |

## Quickstart (local)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

cp .env.example .env
# set JEV_API_KEY and HF_TOKEN (private checkpoint)

mkdir -p data/models
export $(grep -v '^#' .env | xargs)
python -m uvicorn jev_api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Wait until `/ready` returns `"ready": true` (first start downloads the checkpoint).

### curl

```bash
curl -s https://jev-api.codiku.com/health

curl -s https://jev-api.codiku.com/ready

curl -s -X POST https://jev-api.codiku.com/v1/decide \
  -H "Authorization: Bearer $JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "workflow",
    "question": "What should happen next?",
    "state": "The customer wants to renovate their bathroom and asks for a quote.",
    "options": [
      {"id": "create_client", "description": "Create a client"},
      {"id": "create_job", "description": "Create a job"},
      {"id": "create_quote", "description": "Create a quote"}
    ]
  }'
```

Example response:

```json
{
  "choice": "create_quote",
  "scores": {
    "create_client": 0.03,
    "create_job": 0.18,
    "create_quote": 0.79
  },
  "confidence": 0.94,
  "latency_ms": 24.1,
  "qtype": "choice",
  "expected_level": 1.76,
  "label_index": 2,
  "option_keys": ["create_client", "create_job", "create_quote"],
  "model": "openJev-verdict-2.0"
}
```

`scores` and `confidence` are the real model outputs (temperature-scaled softmax + correctness head). Nothing is fabricated.

### TypeScript client

```ts
type Option = { id: string; description: string };

type DecideResponse = {
  choice: string;
  scores: Record<string, number>;
  confidence: number;
  latency_ms: number;
  qtype: "choice" | "score" | "noul";
};

export async function decide(params: {
  baseUrl: string;
  apiKey: string;
  type?: string;
  question: string;
  state: string;
  options: Option[];
}): Promise<DecideResponse> {
  const res = await fetch(`${params.baseUrl.replace(/\/$/, "")}/v1/decide`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${params.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: params.type ?? "workflow",
      question: params.question,
      state: params.state,
      options: params.options,
    }),
  });
  if (res.status === 401) throw new Error("Invalid or missing API key");
  if (!res.ok) throw new Error(`decide failed: ${res.status} ${await res.text()}`);
  return res.json() as Promise<DecideResponse>;
}
```

## Environment variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `JEV_API_KEY` | yes (prod) | — | Bearer token for `/v1/*` |
| `HF_TOKEN` | yes (first download) | — | HF token for private `heman10x/openJev-verdict-2.0` |
| `HF_REPO` | no | `heman10x/openJev-verdict-2.0` | Checkpoint repo |
| `HF_REVISION` | no | latest | Optional pin |
| `MODEL_CACHE_DIR` | no | `/data/models` | Persistent download cache |
| `BACKBONE_DIR` | no | `./vendor/modernbert` | Local ModernBERT config + tokenizer |
| `LOCAL_CHECKPOINT_DIR` | no | — | Skip HF; use a local unpacked dir |
| `CHECKPOINT_FILENAME` | no | `model.pt` | Checkpoint file name |
| `JEV_DEVICE` | no | `cpu` | Torch device |
| `DOWNLOAD_ON_STARTUP` | no | `true` | Background load on boot |
| `HOST` / `PORT` | no | `0.0.0.0` / `8000` | Bind address |

**Never commit secrets.** Use Coolify / Bitwarden env injection.

## Docker

```bash
docker build -t jev-api .
docker run --rm -p 8000:8000 \
  -e JEV_API_KEY=dev-key \
  -e HF_TOKEN=$HF_TOKEN \
  -v jev-cache:/data/models \
  jev-api
```

Or `docker compose up --build`.

- Multi-stage, **CPU-only** PyTorch wheels (no CUDA)
- Model weights are **not** baked into the image
- Healthcheck hits `/health`; readiness is `/ready`

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Coverage includes `/health`, `/ready`, auth, payload validation, multi-option inference fixtures, score coherence, and preprocessing layout checks against the vendored tokenizer.

## Benchmark

```bash
python scripts/benchmark.py \
  --base-url https://jev-api.codiku.com \
  --api-key "$JEV_API_KEY" \
  --n 500 \
  --concurrency 1
```

Smoke:

```bash
python scripts/smoke_check.py --base-url https://jev-api.codiku.com --api-key "$JEV_API_KEY"
```

### Performance notes

- Keep **`--workers 1`**. Each extra Uvicorn worker loads another ~150M-param copy (~600MB+ RAM).
- First request after cold start includes model load (minutes on first download); subsequent requests stay warm.
- Expected CPU latency on a modest VPS: roughly **20–40 ms** model time per decision (upstream reports ~20–25 ms), plus network.
- Prefer a volume on `MODEL_CACHE_DIR` so restarts do not re-download.

## Deployment (Coolify)

`coolify.config.json` describes the Dockerfile app (port **8000**, health `/health`, ~3 GiB RAM / 2 CPU limits). Hermes wires domain `jev-api.codiku.com`, HTTPS, `JEV_API_KEY`, `HF_TOKEN`, and the model cache volume.

## Project layout

```text
jev-api/
├── Dockerfile
├── compose.yaml
├── coolify.config.json
├── pyproject.toml
├── vendor/modernbert/          # config + tokenizer (no weights)
├── src/jev_api/
│   ├── main.py                 # FastAPI routes
│   ├── engine.py               # download + load-once + infer
│   ├── model.py                # VerdictModel (upstream-faithful)
│   ├── preprocess.py           # build_item / collate
│   ├── auth.py
│   ├── config.py
│   └── schemas.py
├── tests/
└── scripts/
```

## License

API code: Apache-2.0 aligned with upstream openJev. Upstream model license applies to the checkpoint weights.
