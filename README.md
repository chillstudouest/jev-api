# jev-api

Self-hosted **Jev / TypeSafe System One** compatible HTTP API, powered by **openJev-verdict-2.0**.

> Protocol compatibility ≠ model identity. Clients can point at this host with the same request/response shapes as `POST /v1/systemone`. Predictions come from OpenJev Verdict 2.0, not from TypeSafe Jev.

```text
Client  →  POST /v1/systemone  →  Jev adapter  →  openJev-verdict-2.0  →  Jev-shaped answers
```

- Upstream engine: [openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0)
- Checkpoint: `heman10x/openJev-verdict-2.0`
- Contract reference: [jev-agent.com/api-reference](https://jev-agent.com/api-reference)
- Prod URL: `https://jev-api.codiku.com`

## Endpoints

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| `POST` | `/v1/systemone` | Bearer | **Primary** — Jev-compatible |
| `POST` | `/api/v1/systemone` | Bearer | Alias (jev-agent path layout) |
| `GET` | `/health` | no | Liveness |
| `GET` | `/ready` | no | Model loaded |
| `GET` | `/v1/model` | Bearer | Runtime metadata (non-Jev debug) |

Accepted `model` values (all map to OpenJev locally): `jev-latest`, `jev-preview`, `jev-1.13.0` (default when omitted).

## Local install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
cp .env.example .env   # set JEV_API_KEY and HF_TOKEN
python -m uvicorn jev_api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Health

```bash
curl -s https://jev-api.codiku.com/health
curl -s https://jev-api.codiku.com/ready
```

## System One (multi-question)

```bash
curl -s -X POST https://jev-api.codiku.com/v1/systemone \
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

### Choice

Returns `choice`, `probabilities{option}`, `confidence`.

### Score

Returns weighted-mean `score` on indices `0..n-1` (not 0–1), plus `legend`, `probabilities` keyed `"0","1",…`, and `confidence`.

### Noul

Returns only `{"type":"noul","noul": <p_true>}` — **no** `confidence` field.

## TypeScript client

```ts
type SystemOneResponse = {
  model: string;
  answers: Record<string, Record<string, unknown>>;
  usage: { input_tokens: number; output_tokens: number };
};

export async function systemOne(params: {
  baseUrl: string; // e.g. https://jev-api.codiku.com
  apiKey: string;
  state: string;
  questions: Record<string, unknown>;
  model?: string;
}): Promise<SystemOneResponse> {
  const res = await fetch(`${params.baseUrl.replace(/\/$/, "")}/v1/systemone`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${params.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      state: params.state,
      model: params.model ?? "jev-1.13.0",
      questions: params.questions,
    }),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<SystemOneResponse>;
}
```

Swap `https://api.typesafe.ai` → `https://jev-api.codiku.com` (same path `/v1/systemone`, same body). Predictions will differ because the engine is OpenJev.

## Auth & errors (Jev-shaped)

| Case | HTTP | `detail` |
| --- | --- | --- |
| Missing `Authorization` | 403 | `authentication_error` / Must supply an API key |
| Bad key | 401 | `authentication_error` / Cannot authenticate… |
| Unknown type / model | 400 | `api_usage_error` |
| Context too long | 400 | `max_tokens_exceeded` |
| Schema / missing fields | 422 | FastAPI field list |

Env: `JEV_API_KEY` (never commit). Also `HF_TOKEN`, `HF_REPO`, `MODEL_CACHE_DIR`, `BACKBONE_DIR`, `JEV_DEVICE=cpu`.

## Architecture

| Layer | Role |
| --- | --- |
| HTTP | FastAPI + Uvicorn (1 worker) |
| Contract | `schemas.py` — Jev System One |
| Adapter | `adapter.py` — qdef ↔ answers |
| Engine | PyTorch + Transformers CPU, load-once, batch all questions in one forward |

## Tests

```bash
pytest -q
```

Golden fixtures live in `tests/fixtures/jev/`.

## Benchmark

```bash
python scripts/benchmark.py --base-url https://jev-api.codiku.com --api-key "$JEV_API_KEY" --n 100
```

Keep `--workers 1` so RAM stays at one model copy (~150M params).

## Docker / Coolify

See `Dockerfile`, `compose.yaml`, `coolify.config.json` (port 8000, health `/health`). Domain: `jev-api.codiku.com`.
