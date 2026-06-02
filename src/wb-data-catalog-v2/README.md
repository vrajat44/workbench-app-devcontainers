# WB Data Catalog (Experimental)

AI-powered metadata discovery, profiling, and cohort building for BigQuery datasets. Automatically generates semantic profiles, FHIR concept bindings, and cross-table relationships, then lets you build cohorts and query your data through natural language.

**[Install on Workbench](INSTALL_GUIDE.md)** | **[Usage Guide](USAGE_GUIDE.md)**

## Features

| Feature | Description |
|---------|-------------|
| **Data Catalog** | Browse all BigQuery datasets and tables with progressive loading |
| **Technical Profiling** | Column stats, null rates, distinct counts, top values, patterns (pure BigQuery) |
| **Semantic Profiling** | AI-generated business names, definitions, sensitivity labels, entity classification, cohort dimensions |
| **FHIR Concept Bindings** | Fixed concept binding (column IS a concept) vs code system binding (column CONTAINS codes) |
| **Structural Links** | Typed join paths with cardinality and confidence between tables |
| **Cohort Builder** | Three modes: table filters, terminology-based, and natural language |
| **Chat Agent** | Q&A mode (metadata questions) and Agent mode (SQL generation + execution) |
| **Terminology Registry** | Cross-table standardized codes (LOINC, SNOMED, ICD-10, RxNorm, CPT) |
| **Chart Advisor** | AI-suggested visualizations from profiled data |
| **Graphic Walker** | Drag-and-drop visual analytics (Tableau-style) |

## Architecture

| Layer | Stack |
|-------|-------|
| **Frontend** | React 18 + Vite + TypeScript + Recharts |
| **Backend** | FastAPI + Python 3.11+, BigQuery SDK, GCS, Vertex AI (Gemini) |
| **Profiling** | `backend/verily_profiler/` — technical stats + semantic LLM profiling |
| **Chat** | `backend/verily_chat/` — LangGraph agent with BigQuery tools |
| **Storage** | GCS bucket `metadata-json-{project}` for profiles, catalog context, terminology |
| **Deploy** | Docker, Workbench custom app (devcontainers pattern) |

## Environment variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Billing project for BigQuery jobs and Vertex AI |
| `DATA_PROJECT_ID` | Project whose datasets are listed (defaults to `GCP_PROJECT_ID`) |
| `GEMINI_MODEL` | Model override (default: auto-detect, typically `gemini-2.5-flash`) |
| `CHAT_MODEL` | Chat model override (optional) |
| `CORS_ORIGINS` | Comma-separated allowed origins (default: `http://localhost:5173`) |
| `LOG_FORMAT` | Set to `json` for structured JSON logging |

## Local development

### One-time setup

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd frontend && npm install
```

### Running the app

**Terminal 1 — Backend** (port 8080):

```bash
cd backend && source .venv/bin/activate
GCP_PROJECT_ID=<your-project> uvicorn main:app --host 127.0.0.1 --port 8080
```

**Terminal 2 — Frontend** (port 5173):

```bash
cd frontend && npm run dev
```

Open http://localhost:5173. Vite proxies `/api/*` to the backend on port 8080.

### Production build (single process)

```bash
cd frontend && npm run build:static
cd ../backend && source .venv/bin/activate && FRONTEND_DIST=./static uvicorn main:app --host 0.0.0.0 --port 8080
```

## Docker

```bash
docker network create app-network   # once
docker compose build && docker compose up
```

Requires: BigQuery (jobs + metadata + data), GCS (read/write on profile bucket), Vertex AI access.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Liveness check |
| `GET /api/health/deep` | Readiness check (validates GCS access) |
| `GET /api/config` | Current configuration |
| `PUT /api/config` | Update settings (billing project, data project, model) |
| `GET /api/models` | Available Gemini models and locations |
| `GET /api/datasets` | List dataset names |
| `GET /api/datasets/{id}/tables` | Tables in a dataset with profiling status |
| `GET /api/.../preview` | Row preview (50 rows) |
| `POST /api/.../profile/technical` | Run technical profiling |
| `POST /api/.../profile/semantic` | Run semantic profiling |
| `POST /api/bulk-profile` | Bulk profile multiple tables |
| `POST /api/chat` | Chat message (Q&A or agent mode) |
| `GET /api/terminology` | Cross-table terminology registry |
| `GET /api/cohorts/dimensions` | Cohort filter dimensions |
| `POST /api/cohorts/from-terminology` | Build cohort from terminology filters |
| `POST /api/cohorts/from-natural-language` | Build cohort from natural language |
| `DELETE /api/profiles/reset` | Delete all profiling artifacts |

## Repo layout

```
backend/
  main.py                 FastAPI app, all API endpoints
  profiling_runner.py     Async profiling job management
  bulk_profiler.py        Concurrent batch profiling + structural links
  chat_handler.py         Chat session management + schema linking
  bq_preview.py           BigQuery table preview
  chart_advisor.py        LLM chart suggestions
  gw_computation.py       Graphic Walker query translation
  verily_profiler/        Profiling engine (technical + semantic)
  verily_chat/            Chat Q&A + LangGraph agent
frontend/
  src/pages/              CatalogPage, TablePage, ChatPage, CohortsPage, etc.
  src/components/         UI components, help system, notifications
  src/hooks/              Data fetching hooks with caching + abort
tests/
  run_profile_field_tests.py    Profile field validation tests
  run_regression_tests.py       End-to-end regression tests
Dockerfile                Multi-stage build (Node + Python)
docker-compose.yaml       Workbench custom app pattern
start.sh                  Container entrypoint
```
