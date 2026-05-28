# Installing WB Data Catalog v2 on Workbench

## Prerequisites

- A Verily Workbench workspace with OWNER or WRITER access
- At least one BigQuery dataset in the workspace (or a referenced data project)

## Steps

### Step 1: Open the Apps tab

In your Workbench workspace, go to the **Apps** tab and click **+ New app instance**.

![Step 1 — Apps tab](docs/install-step0-apps-tab.png)

### Step 2: Select Custom app type

Scroll down the app type list and select **Custom** at the bottom. Click **Next**.

![Step 2 — Select Custom](docs/install-step1-select-custom.png)

### Step 3: Configure the container

Fill in the following fields:

| Field | Value |
|-------|-------|
| **App name** | `WB Data Catalog v2` |
| **Description** | (optional) Self-service data catalog with AI profiling |
| **Repository URL** | `https://github.com/vrajat44/workbench-app-devcontainers.git` |
| **Repository branch** | `wb-data-catalog-v2` |
| **Repository folder path** | `src/wb-data-catalog-v2` |

![Step 3 — App details and container setup](docs/install-step2-app-details.png)

Click **Next**.

### Step 4: Compute options

Use the defaults or adjust based on your needs:

| Setting | Recommended |
|---------|------------|
| **Machine type** | `e2-standard-4` (4 vCPU, 16 GB) or higher |
| **Disk size** | 50 GB (default is fine) |
| **Region** | Same region as your BigQuery datasets (e.g., `us-central1`) |

Click **Next**.

### Step 5: Review and create

Review the details and click **Create**. The app will take 2-5 minutes to build and start.

### Step 6: Open the app

Once the status shows **Running**, click **Open** to launch the Data Catalog. The app auto-detects your workspace's GCP project and starts loading datasets.

## First-time setup

On first launch:

1. The app auto-detects your workspace project. If needed, go to **Settings** in the sidebar to change the billing or data project.
2. Click a dataset to expand it and see its tables.
3. Select tables and click **Profile entire project** to generate metadata.
4. See [USAGE_GUIDE.md](USAGE_GUIDE.md) for full usage instructions.

---

## Local Installation (Developer Setup)

Run the app locally for development, testing, or when the Workbench platform is unavailable.

### Prerequisites

- **Python 3.11+** and **Node.js 20+**
- **GCP credentials** — run `gcloud auth application-default login` (ADC) so the backend can access BigQuery, GCS, and Vertex AI
- **Workbench CLI** (optional) — install `wb` for workspace listing and auto-detection. Run `wb auth login` to authenticate. Without it, you can still use the app by entering a GCP project ID manually in Settings.

### 1. Clone and install dependencies

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend (separate terminal)
cd frontend
npm install
```

### 2. Start the app

**Terminal 1 — Backend (port 8080):**

```bash
cd backend
source .venv/bin/activate
python main.py
```

This starts uvicorn with hot-reload enabled. The backend auto-detects your GCP project from the Workbench CLI or falls back to the default billing project.

**Terminal 2 — Frontend (port 5173):**

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies all `/api/*` requests to the backend on port 8080.

### 3. Configure via the UI

On first load, the Settings panel appears. Choose one of:

- **Workbench workspace** — select a workspace from the dropdown. Requires `wb auth login`.
- **Custom GCP project** — enter a project ID directly (no `wb` CLI needed).

The app needs:
- A **billing project** — pays for BigQuery jobs, Vertex AI calls, and owns the GCS profile bucket
- A **data project** — whose BigQuery datasets are browsed and profiled (can be the same as billing)

### 4. Optional environment variables

Override defaults by exporting before starting the backend:

```bash
export GCP_PROJECT_ID=my-billing-project
export DATA_PROJECT_ID=my-data-project
export GEMINI_MODEL=gemini-3.5-flash
export GEMINI_LOCATION=us-central1
```

### 5. Production-style single-process (optional)

Build the frontend and serve everything from the backend on port 8080:

```bash
cd frontend && npm run build
cp -r dist/* ../backend/static/
cd ../backend && source .venv/bin/activate
FRONTEND_DIST=./static uvicorn main:app --host 0.0.0.0 --port 8080
```

### Local troubleshooting

| Issue | Fix |
|-------|-----|
| `python main.py` exits silently | Ensure you're in the `backend/` directory with the venv activated. Check that `uvicorn` is installed (`pip install uvicorn`). |
| "wb CLI unavailable" in Settings | Install the Workbench CLI or switch to **Custom GCP project** mode. |
| "Workbench session expired" | Run `wb auth login` in your terminal, then re-select the workspace. |
| "No BigQuery datasets found" | Check that the selected project actually has BQ datasets. Verify with `bq ls --project_id=<project>`. |
| Backend can't reach GCP | Run `gcloud auth application-default login` to refresh ADC credentials. |
| Port 8080 already in use | Kill the existing process: `lsof -i :8080` then `kill <PID>`, or use a different port: `uvicorn main:app --port 9090` (update `vite.config.ts` proxy target to match). |

### Claude Code quick-start prompt

Copy and paste this into Claude Code to have it install and run the app for you:

```
Install and run the WB Data Catalog v2 app locally from this directory. Steps:
1. Set up the backend Python venv in backend/ and install requirements.txt
2. Install frontend npm dependencies in frontend/
3. Verify GCP credentials are available (gcloud auth application-default print-access-token)
4. Start the backend (python main.py) and frontend (npm run dev) 
5. Confirm both servers are healthy (curl localhost:8080/api/health, check localhost:5173 is serving)
6. Report the URLs to open in the browser

If gcloud credentials are missing, tell me to run: gcloud auth application-default login
If wb CLI auth is expired, tell me to run: wb auth login
Don't modify any source files — just install deps and start servers.
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App shows "Loading datasets..." forever | The GCP project may not have BQ datasets, or permissions are missing. Check Settings. |
| Profiling fails | Ensure the workspace service account has BigQuery Data Viewer and Storage Object Admin on the profile bucket. |
| Chat shows "No context" | Profile at least one table first. The chat agent needs profiling data to answer questions. |
| Can't access Vertex AI / Gemini | Ensure the Vertex AI API is enabled in your billing project and the service account has `aiplatform.user` role. |
