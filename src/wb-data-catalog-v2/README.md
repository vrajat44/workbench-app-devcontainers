# WB Data Catalog v2

Self-service data catalog for BigQuery datasets on Verily Workbench. Browse datasets, profile tables with AI-generated metadata, explore data visually, and query with natural language.

**For usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).**

## What it does

- **Data Catalog** — browse all BigQuery datasets/tables in a GCP project with progressive on-demand loading
- **Technical Profiling** — column stats, null rates, distinct counts, top values, patterns (pure BigQuery, no AI)
- **Semantic Profiling** — AI-generated business names, definitions, HIPAA sensitivity labels, terminology bindings, join paths, cohort dimensions (Gemini LLM)
- **Key Insights** — AI-suggested chart visualizations from profile data
- **Interactive Explorer** — drag-and-drop visual analytics (Graphic Walker)
- **Terminology Registry** — cross-table standardized codes (LOINC, SNOMED, ICD-10, RxNorm, CPT)
- **Cohort Builder** — filter and count subjects across tables using dimensions, terminology, or natural language
- **Data AMA Agent** — chat interface with Q&A mode (metadata questions) and Agent mode (SQL generation + execution)

## Architecture

| Layer | Stack |
|-------|-------|
| **Backend** | FastAPI + Python, BigQuery, GCS, Vertex AI (Gemini) |
| **Frontend** | Vite + React + TypeScript + Recharts |
| **Profiling** | `backend/verily_profiler/` — technical stats + semantic LLM profiling |
| **Chat** | `backend/verily_chat/` — LangGraph agent with BigQuery tools |
| **Deploy** | Docker, Workbench app-devcontainers pattern |

## Environment variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Billing project for BigQuery jobs and Vertex AI. Defaults to `wb-shrewd-papaya-8403` if not set. |
| `DATA_PROJECT_ID` | Project whose datasets are listed (defaults to `GCP_PROJECT_ID`) |
| `GEMINI_MODEL` | Optional model override (default: auto-detect, typically `gemini-3.5-flash`) |
| `CHAT_MODEL` | Optional chat model override |
| `CORS_ORIGINS` | Comma-separated allowed origins (default: `http://localhost:5173`) |
| `LOG_FORMAT` | Set to `json` for structured JSON logging |
| `FRONTEND_DIST` | Path to built SPA (default: `backend/static` in Docker image) |

## Local development

### One-time setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd frontend
npm install
```

### Running the app

**Terminal 1 — Backend:**

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8080
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Open http://localhost:5173/. The Vite dev server proxies `/api/*` to the backend on port 8080.

### Production-style single-process

```bash
cd frontend && npm run build && mkdir -p ../backend/static && rm -rf ../backend/static/* && cp -r dist/* ../backend/static/
cd ../backend && source .venv/bin/activate && FRONTEND_DIST=./static uvicorn main:app --host 0.0.0.0 --port 8080
```

## Docker

```bash
docker network create app-network   # once
docker compose build
docker compose up
```

Requires a service account with BigQuery (jobs + metadata + data), GCS (read/write on profile bucket), and Vertex AI access.

## API summary

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Liveness check |
| `GET /api/health/deep` | Readiness check (validates GCS access) |
| `GET /api/config` | Current configuration |
| `GET /api/models` | Available Gemini models |
| `GET /api/datasets` | List dataset names (fast) |
| `GET /api/datasets/{id}/tables` | Tables in one dataset with profiling status |
| `GET /api/.../preview` | Capped row preview (50 rows) |
| `POST /api/.../profile/technical` | Start technical profiling (async) |
| `POST /api/.../profile/semantic` | Start semantic profiling (async) |
| `POST /api/bulk-profile` | Bulk profile multiple tables |
| `POST /api/chat` | Chat message (metadata Q&A or agent mode) |
| `GET /api/chat/context-info` | What context is available for chat |
| `GET /api/terminology` | Cross-table terminology registry |
| `GET /api/cohorts/dimensions` | Cohort filter dimensions from profiles |
| `DELETE /api/profiles/reset` | Delete all profiling artifacts |

## Repo layout

```
backend/
  main.py               FastAPI app, all API endpoints
  profiling_runner.py    Async profiling job management
  bulk_profiler.py       Concurrent batch profiling
  chat_handler.py        Chat session management + schema linking
  bq_preview.py          BigQuery table preview
  chart_advisor.py       LLM chart suggestions
  gw_computation.py      Graphic Walker query translation
  verily_profiler/       Profiling engine (technical + semantic)
  verily_chat/           Chat Q&A + LangGraph agent
frontend/
  src/pages/             CatalogPage, TablePage, ChatPage, SettingsPage, etc.
  src/components/        UI components, help system, notifications
  src/hooks/             Data fetching hooks with caching + abort
tests/
  run_structural_tests.py   Structural compliance tests for profiles
  run_p0_tests.py           Entity classification + cross-table tests
Dockerfile              Multi-stage build (Node + Python)
docker-compose.yaml     Workbench app-devcontainers pattern
USAGE_GUIDE.md          End-user usage guide
```
