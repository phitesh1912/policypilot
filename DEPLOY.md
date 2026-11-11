# Deploying PolicyPilot

The app is one container: FastAPI serves the API and the built React UI. State is an embedded
Qdrant directory that is rebuilt from `data/policies/` at startup, so the container is effectively
stateless and safe to redeploy at will.

## 1. Get a free Gemini key

1. Go to <https://aistudio.google.com> and sign in with a Google account.
2. **Get API key → Create API key**. Copy it; it is shown once.
3. The free tier is enough for the demo and the eval. Models used by default:
   `gemini-2.5-flash` (chat) and `gemini-embedding-001` (embeddings, 768 dims).

Never commit the key. Locally it goes in `.env` (git-ignored); on Render it goes in the dashboard.

## 2. Environment variables

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` \| `gemini` \| `openai`. `fake` needs no key and is what CI uses. |
| `LLM_MODEL` | `gemini-2.5-flash` | Chat model name for the chosen provider |
| `EMBED_MODEL` | `gemini-embedding-001` | Embedding model name |
| `GEMINI_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `ADMIN_TOKEN` | — | Bearer token for `POST /ingest`. If unset, ingest returns 503. |
| `QDRANT_URL` / `QDRANT_API_KEY` | — | Set both to use Qdrant Cloud instead of the embedded store |
| `QDRANT_PATH` | `data/qdrant` | Embedded store location (`/data/qdrant` in the image) |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP` | `350` / `50` | Chunker settings |
| `RETRIEVAL_K` | `10` | Candidates per retriever before fusion |
| `RRF_K` | `60` | RRF constant |
| `REFUSAL_THRESHOLD` | `0.02` | Minimum top fused score to call the model |
| `CONTEXT_BUDGET_TOKENS` | `1800` | Max retrieved tokens packed into the prompt |
| `CACHE_CAPACITY` / `CACHE_SIMILARITY_THRESHOLD` | `256` / `0.95` | Semantic cache |
| `PRICE_INPUT_PER_1M` / `PRICE_OUTPUT_PER_1M` / `PRICE_EMBED_PER_1M` | `0.30` / `2.50` / `0.15` | USD per 1M tokens for the cost estimate; set to your provider's current price sheet |
| `PORT` | `8000` | Render sets this automatically |

## 3. Render (recommended, free tier)

`render.yaml` is a Blueprint, so the whole service is declared in the repo.

1. Push the repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo. Render reads `render.yaml` and proposes one web
   service named `policypilot` (Docker runtime, free plan, Singapore region — change `region` if
   you prefer).
3. When prompted for `GEMINI_API_KEY`, paste the key from step 1. `ADMIN_TOKEN` is generated
   for you; copy it from the service's Environment tab if you want to call `/ingest`.
4. Click **Apply**. The first build takes ~3–4 minutes (Node build stage + Python stage).
5. Open the service URL. `/health` should return `"provider": "gemini"` and the chunk count.

Notes for the free tier:
- The instance sleeps after 15 minutes idle; the first request after that takes ~30 s while it
  boots and re-ingests. Mention this in the demo or keep it warm with a free uptime pinger.
- The filesystem is ephemeral; that's fine because the index is rebuilt at startup.
- `autoDeploy: true` means every push to `main` redeploys.

To redeploy with a different model, change `LLM_MODEL` in the Environment tab; no rebuild needed.

## 4. Docker, anywhere

```bash
docker build -t policypilot .
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=gemini -e GEMINI_API_KEY=... -e ADMIN_TOKEN=change-me \
  policypilot
```

The image is multi-stage (`node:22-alpine` builds the UI, `python:3.12-slim` runs the app), runs as
a non-root user, bakes the tiktoken tables in at build time so startup needs no outbound download
beyond the LLM API, and exposes a `HEALTHCHECK` on `/health`.

## 5. Azure Container Apps (optional)

```bash
az group create -n policypilot-rg -l southeastasia
az acr create -n policypilotacr -g policypilot-rg --sku Basic --admin-enabled true
az acr build -r policypilotacr -t policypilot:latest .
az containerapp env create -n policypilot-env -g policypilot-rg -l southeastasia
az containerapp create -n policypilot -g policypilot-rg --environment policypilot-env \
  --image policypilotacr.azurecr.io/policypilot:latest \
  --registry-server policypilotacr.azurecr.io \
  --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 1 \
  --secrets gemini-key=<KEY> admin-token=<TOKEN> \
  --env-vars LLM_PROVIDER=gemini GEMINI_API_KEY=secretref:gemini-key ADMIN_TOKEN=secretref:admin-token
```

`--min-replicas 0` gives scale-to-zero like Render's free tier. Add a liveness probe on `/health`
in the portal or via `--yaml` if you want restarts on failure.

## 6. Running the eval against the deployed provider

The eval runs locally but uses the same provider configuration:

```bash
LLM_PROVIDER=gemini GEMINI_API_KEY=... python -m eval.run --provider gemini
```

It writes `eval/results/results_gemini.md`. Copy that table into the README's Gemini section —
that is the only sanctioned way numbers get there.

## 7. Local development

```bash
pip install -e ".[dev]"
uvicorn policypilot.api:app --reload          # API on :8000, serves ./static if it exists
cd frontend && npm ci && npm run dev          # UI on :5173 with /ask proxied to :8000
```

Set `LLM_PROVIDER=fake` (the default) to work without any key; everything, including streaming
and refusals, works offline.
