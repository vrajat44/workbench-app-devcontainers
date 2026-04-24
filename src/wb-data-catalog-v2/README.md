# WB Data Catalog

React + FastAPI app: browse all BigQuery datasets/tables in a GCP project, preview capped rows, run **technical (C2a)** and **semantic (C2b)** profiling on demand, view profiles from GCS, and get **LLM-suggested charts** (Gemini).

UI uses lightweight **Verily Pre–inspired** tokens and RDS-shaped primitives in `frontend/src/components/rds.tsx` — swap in `@verily-src/rds-*` when your npm registry is configured.

## Environment variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Billing / ADC project for BigQuery jobs and Vertex AI |
| `DATA_PROJECT_ID` | Project whose datasets are listed (defaults to `GCP_PROJECT_ID`) |
| `PROFILE_GCS_BUCKET` | Bucket name (no `gs://`) where `profiling/{project}/{dataset}/{table}/tech_profile.json` and `semantic_profile.json` are stored |
| `GEMINI_MODEL` | Optional override (e.g. `gemini-2.5-flash`) |
| `FRONTEND_DIST` | Optional path to built SPA (default: `backend/static` in Docker image) |

## Local development

**Backend** (from `backend/`):

```bash
cd backend
pip install -r requirements.txt
export GCP_PROJECT_ID=your-billing-project
export DATA_PROJECT_ID=your-data-project
export PROFILE_GCS_BUCKET=your-bucket
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

**Frontend** (Vite proxies `/api` to port 8080):

```bash
cd frontend
npm install
npm run dev
```

Build SPA into `backend/static` for single-process serving:

```bash
cd frontend && npm run build && mkdir -p ../backend/static && rm -rf ../backend/static/* && cp -r dist/* ../backend/static/
cd ../backend && FRONTEND_DIST=./static uvicorn main:app --host 0.0.0.0 --port 8080
```

## Docker / Compute Engine

**Workbench / local compose:** `docker-compose.yaml` follows [workbench-app-devcontainers](https://github.com/vrajat44/workbench-app-devcontainers/blob/master/README.md) (same pattern as `src/example/docker-compose.yaml`): `container_name: application-server`, external **`app-network`**, and FUSE flags for gcsfuse. Before `docker compose up` locally, create the network once:

```bash
docker network create app-network
```

Then:

```bash
docker compose build
export GCP_PROJECT_ID=...
export DATA_PROJECT_ID=...   # optional; defaults to billing project
docker compose up
```

Workbench creates `app-network` in its environment; you do not manage that in the cloud UI.

On **Compute Engine**, use a service account with:

- BigQuery: `bigquery.jobs.create`, read metadata and table data for preview/profiling
- Storage: read/write objects on `PROFILE_GCS_BUCKET`
- Vertex AI: Gemini access in your region

Reserve a static external IP, allow TCP **8080** in firewall rules, then open `http://<EXTERNAL_IP>:8080`.

## API summary

- `GET /api/catalog` — all datasets + tables + profiling flags from GCS index
- `GET /api/projects/{p}/datasets/{d}/tables/{t}/preview` — capped preview
- `POST .../profile/technical` / `POST .../profile/semantic` — start profiling (async)
- `GET .../profile/status` — `{ technical, semantic }` states
- `GET .../profile/technical` / `.../semantic` — JSON profiles
- `POST /api/charts/suggest` — body `{ technical, semantic? }` → suggested charts

## Repo layout

- `backend/` — FastAPI, BQ preview/discovery, profiling runner, chart advisor, vendored `profiler/` package from WB Data Profiler
- `frontend/` — Vite + React + Recharts
- `Dockerfile` / `docker-compose.yaml` — production-style container
