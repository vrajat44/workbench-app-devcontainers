"""
WB Data Catalog — FastAPI backend: BQ discovery, preview, profiling, GCS profiles, charts, static SPA.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api_models import (
    CatalogResponse,
    ChartSuggestion,
    ChartsSuggestResponse,
    JobStartResponse,
    ProfileStatusResponse,
    TableSummary,
)
from bq_discovery import discover_bq_datasets, discover_bq_tables
from bq_preview import MAX_PREVIEW_ROWS, preview_table
from chart_advisor import suggest_charts
from gcs_paths import parse_fq_table, sem_object_path, tech_object_path
from gcs_reader import read_json_if_exists, scan_profile_availability
from profiling_runner import (
    job_state,
    load_table_info,
    profile_status_from_gcs_and_jobs,
    run_semantic_profile_async,
    run_technical_profile_async,
)


def _profiling_for_catalog_row(fq: str, prof: dict[str, bool]) -> dict[str, str]:
    """Merge GCS profile presence with in-memory job running flags."""
    flags = job_state.running_flags(fq)
    tech = "running" if flags["technical"] else ("available" if prof["technical"] else "none")
    sem = "running" if flags["semantic"] else ("available" if prof["semantic"] else "none")
    return {"technical": tech, "semantic": sem}
from profiler.prompt_engine import detect_available_model

# ── Settings ──────────────────────────────────────────────────────────────────
# PROFILE_BUCKET is derived from the billing project: metadata-json-{project-id}
# (matches the convention used by WB_Data_Profiler and existing Workbench buckets).
# GCP_PROJECT_ID, DATA_PROJECT_ID, GEMINI_MODEL are configurable at runtime
# via the UI (PUT /api/settings).  Env vars seed the initial values.

PROFILE_BUCKET = ""

BILLING_PROJECT: str = ""
DATA_PROJECT: str = ""
GEMINI_MODEL: Optional[str] = None
FRONTEND_DIST = Path(
    os.environ.get("FRONTEND_DIST", str(Path(__file__).resolve().parent / "static")),
)


def _derive_bucket(project_id: str) -> str:
    return f"metadata-json-{project_id}" if project_id else ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BILLING_PROJECT, DATA_PROJECT, GEMINI_MODEL, PROFILE_BUCKET
    BILLING_PROJECT = os.environ.get("GCP_PROJECT_ID") or os.environ.get("BILLING_PROJECT_ID", "")
    DATA_PROJECT = (os.environ.get("DATA_PROJECT_ID") or "").strip() or BILLING_PROJECT
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or None
    PROFILE_BUCKET = _derive_bucket(BILLING_PROJECT)
    if not BILLING_PROJECT:
        print("INFO: GCP_PROJECT_ID not set — configure via UI Settings")
    yield


app = FastAPI(title="WB Data Catalog", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _fq(project_id: str, dataset_id: str, table_id: str) -> str:
    return f"{project_id}.{dataset_id}.{table_id}"


@app.get("/api/health")
def health():
    return {"status": "ok", "data_project": DATA_PROJECT, "profile_bucket": PROFILE_BUCKET}


@app.get("/api/config")
def api_config():
    return {
        "billing_project": BILLING_PROJECT,
        "data_project": DATA_PROJECT,
        "profile_bucket": PROFILE_BUCKET,
        "gemini_model": GEMINI_MODEL,
        "configured": bool(BILLING_PROJECT),
    }


def _check_bucket_exists(bucket_name: str, project_id: str) -> dict[str, Any]:
    """Verify the GCS bucket exists. Returns status dict."""
    from google.cloud import storage

    client = storage.Client(project=project_id)
    bucket_ref = client.bucket(bucket_name)
    try:
        bucket_ref.reload(client=client)
        return {"bucket": bucket_name, "action": "exists"}
    except Exception as e:
        return {"bucket": bucket_name, "action": "error", "error": str(e)}


@app.put("/api/settings")
def api_update_settings(body: dict[str, Any]):
    """
    Update runtime settings from the UI.
    Accepts: { billing_project?, data_project?, gemini_model? }
    Bucket is auto-derived from billing project: metadata-json-{project-id}
    """
    global BILLING_PROJECT, DATA_PROJECT, GEMINI_MODEL, PROFILE_BUCKET
    if "billing_project" in body:
        BILLING_PROJECT = str(body["billing_project"]).strip()
        PROFILE_BUCKET = _derive_bucket(BILLING_PROJECT)
    if "data_project" in body:
        DATA_PROJECT = str(body["data_project"]).strip() or BILLING_PROJECT
    elif not DATA_PROJECT and BILLING_PROJECT:
        DATA_PROJECT = BILLING_PROJECT
    if "gemini_model" in body:
        val = str(body["gemini_model"]).strip()
        GEMINI_MODEL = val if val else None

    bucket_status: dict[str, Any] = {}
    if BILLING_PROJECT and PROFILE_BUCKET:
        bucket_status = _check_bucket_exists(PROFILE_BUCKET, BILLING_PROJECT)
        print(f"Bucket check: {bucket_status}")

    cfg = api_config()
    cfg["bucket_status"] = bucket_status
    return cfg


@app.get("/api/datasets")
def api_datasets():
    if not DATA_PROJECT:
        raise HTTPException(503, "DATA_PROJECT_ID not configured")
    return {"project_id": DATA_PROJECT, "datasets": discover_bq_datasets(DATA_PROJECT, BILLING_PROJECT)}


@app.get("/api/datasets/{dataset_id}/tables")
def api_dataset_tables(dataset_id: str):
    if not DATA_PROJECT:
        raise HTTPException(503, "DATA_PROJECT_ID not configured")
    tables = discover_bq_tables(DATA_PROJECT, dataset_id, BILLING_PROJECT)
    profile_index: dict[str, dict[str, bool]] = {}
    if PROFILE_BUCKET:
        try:
            profile_index = scan_profile_availability(PROFILE_BUCKET, DATA_PROJECT, BILLING_PROJECT)
        except Exception as e:
            print(f"Profile scan failed: {e}")
    out = []
    for t in tables:
        fq = t.fq_name
        prof = profile_index.get(fq, {"technical": False, "semantic": False})
        ts = getattr(t, "creation_time", None)
        out.append(
            TableSummary(
                fq_table=fq,
                project_id=t.project_id,
                dataset_id=t.dataset_id,
                table_id=t.table_id,
                row_count=t.row_count,
                size_bytes=t.size_bytes,
                table_type=t.table_type,
                column_count=len(t.columns),
                creation_time=str(ts) if ts else None,
                profiling=_profiling_for_catalog_row(fq, prof),
                business_name=prof.get("business_name"),
                table_definition=prof.get("table_definition"),
            ).model_dump()
        )
    return {"dataset_id": dataset_id, "tables": out}


@app.get("/api/catalog")
def api_catalog():
    """All datasets with table summaries and profiling flags."""
    if not DATA_PROJECT:
        raise HTTPException(503, "DATA_PROJECT_ID not configured")
    profile_index: dict[str, dict[str, bool]] = {}
    if PROFILE_BUCKET:
        try:
            profile_index = scan_profile_availability(PROFILE_BUCKET, DATA_PROJECT, BILLING_PROJECT)
        except Exception as e:
            print(f"Profile scan failed: {e}")
    datasets = discover_bq_datasets(DATA_PROJECT, BILLING_PROJECT)
    result: list[dict[str, Any]] = []
    for ds in datasets:
        tables = discover_bq_tables(DATA_PROJECT, ds, BILLING_PROJECT)
        rows = []
        for t in tables:
            fq = t.fq_name
            prof = profile_index.get(fq, {"technical": False, "semantic": False})
            ts = getattr(t, "creation_time", None)
            rows.append(
                TableSummary(
                    fq_table=fq,
                    project_id=t.project_id,
                    dataset_id=t.dataset_id,
                    table_id=t.table_id,
                    row_count=t.row_count,
                    size_bytes=t.size_bytes,
                    table_type=t.table_type,
                    column_count=len(t.columns),
                    creation_time=str(ts) if ts else None,
                    profiling=_profiling_for_catalog_row(fq, prof),
                    business_name=prof.get("business_name"),
                    table_definition=prof.get("table_definition"),
                ).model_dump()
            )
        result.append({"dataset_id": ds, "tables": rows})
    return CatalogResponse(
        project_id=DATA_PROJECT,
        profile_bucket=PROFILE_BUCKET or "",
        datasets=result,
    ).model_dump()


@app.get("/api/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}/preview")
def api_preview(project_id: str, dataset_id: str, table_id: str, limit: int = MAX_PREVIEW_ROWS):
    info = load_table_info(_fq(project_id, dataset_id, table_id), BILLING_PROJECT, project_id)
    if not info:
        raise HTTPException(404, "Table not found")
    return preview_table(info, BILLING_PROJECT, limit=limit)


@app.get("/api/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}/profile/status")
def api_profile_status(project_id: str, dataset_id: str, table_id: str):
    fq = _fq(project_id, dataset_id, table_id)
    if not PROFILE_BUCKET:
        raise HTTPException(503, "PROFILE_GCS_BUCKET not configured")
    st = profile_status_from_gcs_and_jobs(fq, PROFILE_BUCKET, BILLING_PROJECT)
    return ProfileStatusResponse(**st).model_dump()


@app.get("/api/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}/profile/technical")
def api_get_technical(project_id: str, dataset_id: str, table_id: str):
    fq = _fq(project_id, dataset_id, table_id)
    if not PROFILE_BUCKET:
        raise HTTPException(503, "PROFILE_GCS_BUCKET not configured")
    p, d, t = parse_fq_table(fq)
    data = read_json_if_exists(PROFILE_BUCKET, tech_object_path(p, d, t), BILLING_PROJECT)
    if not data:
        raise HTTPException(404, "Technical profile not found")
    return data


@app.get("/api/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}/profile/semantic")
def api_get_semantic(project_id: str, dataset_id: str, table_id: str):
    fq = _fq(project_id, dataset_id, table_id)
    if not PROFILE_BUCKET:
        raise HTTPException(503, "PROFILE_GCS_BUCKET not configured")
    p, d, t = parse_fq_table(fq)
    data = read_json_if_exists(PROFILE_BUCKET, sem_object_path(p, d, t), BILLING_PROJECT)
    if not data:
        raise HTTPException(404, "Semantic profile not found")
    return data


@app.post("/api/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}/profile/technical")
async def api_run_technical(
    project_id: str,
    dataset_id: str,
    table_id: str,
    background_tasks: BackgroundTasks,
):
    fq = _fq(project_id, dataset_id, table_id)
    if not PROFILE_BUCKET:
        raise HTTPException(503, "PROFILE_GCS_BUCKET not configured")
    jid, started = job_state.try_start(fq, "technical")
    if not started:
        return JobStartResponse(job_id=jid, status="running").model_dump()

    async def _job():
        await run_technical_profile_async(
            fq_table=fq,
            bucket=PROFILE_BUCKET,
            billing_project=BILLING_PROJECT,
            data_project=DATA_PROJECT,
            job_id=jid,
        )

    background_tasks.add_task(_job)
    return JobStartResponse(job_id=jid, status="running").model_dump()


@app.post("/api/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}/profile/semantic")
async def api_run_semantic(
    project_id: str,
    dataset_id: str,
    table_id: str,
    background_tasks: BackgroundTasks,
):
    fq = _fq(project_id, dataset_id, table_id)
    if not PROFILE_BUCKET:
        raise HTTPException(503, "PROFILE_GCS_BUCKET not configured")
    p, d, t = parse_fq_table(fq)
    if not read_json_if_exists(PROFILE_BUCKET, tech_object_path(p, d, t), BILLING_PROJECT):
        raise HTTPException(409, "Run technical profiling first")
    jid, started = job_state.try_start(fq, "semantic")
    if not started:
        return JobStartResponse(job_id=jid, status="running").model_dump()

    async def _job():
        await run_semantic_profile_async(
            fq_table=fq,
            bucket=PROFILE_BUCKET,
            billing_project=BILLING_PROJECT,
            data_project=DATA_PROJECT,
            model_name=GEMINI_MODEL,
            job_id=jid,
        )

    background_tasks.add_task(_job)
    return JobStartResponse(job_id=jid, status="running").model_dump()


@app.post("/api/charts/suggest")
def api_charts_suggest(body: dict[str, Any]):
    technical = body.get("technical") or {}
    semantic = body.get("semantic")
    if not technical.get("columns"):
        raise HTTPException(400, "technical profile with columns is required")
    model = GEMINI_MODEL or detect_available_model(BILLING_PROJECT)
    charts_raw = suggest_charts(technical, semantic, model, BILLING_PROJECT)
    charts = [ChartSuggestion.model_validate(c) for c in charts_raw]
    return ChartsSuggestResponse(charts=charts).model_dump()


@app.post("/api/gw/compute/{project_id}/{dataset_id}/{table_id}")
def api_gw_compute(project_id: str, dataset_id: str, table_id: str, body: dict[str, Any]):
    """Execute a Graphic Walker computation query against BigQuery."""
    from gw_computation import execute_workflow

    fq = _fq(project_id, dataset_id, table_id)
    try:
        rows = execute_workflow(fq, body, billing_project=BILLING_PROJECT)
        return rows
    except Exception as e:
        raise HTTPException(400, f"Computation failed: {e}")


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    j = job_state.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


# ── Static frontend (production build) ───────────────────────────────────────

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(404)
