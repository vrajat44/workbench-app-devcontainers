"""
WB Data Catalog v2 — FastAPI backend.

Single-project scope: profiles the GCP project provided via settings or
defaults to the current workspace project.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api_models import (
    CatalogResponse,
    ChartSuggestion,
    ChartsSuggestResponse,
    JobStartResponse,
    ProfileStatusResponse,
    TableSummary,
)
from bq_preview import MAX_PREVIEW_ROWS, preview_table
from chart_advisor import suggest_charts
from profiling_runner import (
    job_state,
    load_table_info,
    profile_status_from_gcs_and_jobs,
    run_semantic_profile_async,
    run_technical_profile_async,
)
from verily_profiler import discover_datasets, discover_tables, get_table_api_metadata, scan_profile_availability
from verily_profiler.storage import parse_fq_table, tech_object_path, sem_object_path, read_json_if_exists
from verily_profiler.llm import detect_available_model


import time as _time

_catalog_cache: dict[str, tuple[float, Any]] = {}
_CATALOG_CACHE_TTL = 30  # seconds


def _profiling_for_catalog_row(fq: str, prof: dict[str, bool]) -> dict[str, str]:
    """Merge GCS profile presence with in-memory job running flags."""
    flags = job_state.running_flags(fq)
    tech = "running" if flags["technical"] else ("available" if prof["technical"] else "none")
    sem = "running" if flags["semantic"] else ("available" if prof["semantic"] else "none")
    return {"technical": tech, "semantic": sem}


# ── Settings ──────────────────────────────────────────────────────────────────

PROFILE_BUCKET = ""
BILLING_PROJECT: str = ""
DATA_PROJECT: str = ""
GEMINI_MODEL: Optional[str] = None      # for profiling (tech + semantic)
CHAT_MODEL: Optional[str] = None         # for chat (None = use verily-chat default 3.1-pro)
FRONTEND_DIST = Path(
    os.environ.get("FRONTEND_DIST", str(Path(__file__).resolve().parent / "static")),
)


def _derive_bucket(project_id: str) -> str:
    return f"metadata-json-{project_id}" if project_id else ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BILLING_PROJECT, DATA_PROJECT, GEMINI_MODEL, CHAT_MODEL, PROFILE_BUCKET
    BILLING_PROJECT = os.environ.get("GCP_PROJECT_ID") or os.environ.get("BILLING_PROJECT_ID", "")
    DATA_PROJECT = (os.environ.get("DATA_PROJECT_ID") or "").strip() or BILLING_PROJECT
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or None
    CHAT_MODEL = os.environ.get("CHAT_MODEL") or None
    PROFILE_BUCKET = _derive_bucket(BILLING_PROJECT)
    if not BILLING_PROJECT:
        print("INFO: GCP_PROJECT_ID not set — configure via UI Settings")
    elif PROFILE_BUCKET:
        result = _ensure_bucket(PROFILE_BUCKET, BILLING_PROJECT)
        print(f"Bucket: {result}")
        if DATA_PROJECT:
            _ensure_catalog_context_exists()
    yield


def _ensure_catalog_context_exists():
    """Generate catalog context .md in background if missing but profiles exist."""
    import threading
    from verily_profiler.storage import read_catalog_context, regenerate_catalog_context, scan_profile_availability

    def _work():
        try:
            existing = read_catalog_context(PROFILE_BUCKET, DATA_PROJECT, billing_project_id=BILLING_PROJECT)
            if existing:
                print(f"Catalog context exists ({len(existing)} chars)")
                return
            avail = scan_profile_availability(PROFILE_BUCKET, DATA_PROJECT, billing_project_id=BILLING_PROJECT)
            profiled = [fq for fq, info in avail.items() if info.get("technical") or info.get("semantic")]
            if not profiled:
                return
            print(f"Generating catalog context for {len(profiled)} existing profiles...")
            regenerate_catalog_context(PROFILE_BUCKET, DATA_PROJECT, billing_project_id=BILLING_PROJECT)
        except Exception as e:
            print(f"Startup context generation failed: {e}")

    threading.Thread(target=_work, daemon=True).start()


app = FastAPI(title="WB Data Catalog v2", lifespan=lifespan)
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


def _ensure_bucket(bucket_name: str, project_id: str) -> dict[str, Any]:
    """Check if the profile bucket exists; create it if it doesn't."""
    from google.cloud import storage

    client = storage.Client(project=project_id)
    bucket_ref = client.bucket(bucket_name)
    try:
        bucket_ref.reload(client=client)
        return {"bucket": bucket_name, "action": "exists"}
    except Exception:
        try:
            bucket_ref.storage_class = "STANDARD"
            client.create_bucket(bucket_ref, project=project_id, location="us")
            print(f"Created bucket: {bucket_name}")
            return {"bucket": bucket_name, "action": "created"}
        except Exception as e:
            return {"bucket": bucket_name, "action": "error", "error": str(e)}


@app.put("/api/settings")
def api_update_settings(body: dict[str, Any]):
    """
    Update runtime settings from the UI.
    Accepts: { billing_project?, data_project?, gemini_model? }
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
        bucket_status = _ensure_bucket(PROFILE_BUCKET, BILLING_PROJECT)
        print(f"Bucket check: {bucket_status}")

    cfg = api_config()
    cfg["bucket_status"] = bucket_status
    return cfg


# ── Discovery & catalog ───────────────────────────────────────────────────────

@app.get("/api/datasets")
def api_datasets():
    if not DATA_PROJECT:
        raise HTTPException(503, "No data project configured")
    return {"project_id": DATA_PROJECT, "datasets": discover_datasets(DATA_PROJECT, billing_project=BILLING_PROJECT)}


@app.get("/api/datasets/{dataset_id}/tables")
def api_dataset_tables(dataset_id: str):
    if not DATA_PROJECT:
        raise HTTPException(503, "No data project configured")
    tables = discover_tables(DATA_PROJECT, dataset_id, billing_project=BILLING_PROJECT)
    profile_index: dict[str, dict[str, bool]] = {}
    if PROFILE_BUCKET:
        try:
            profile_index = scan_profile_availability(PROFILE_BUCKET, DATA_PROJECT, billing_project_id=BILLING_PROJECT)
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not DATA_PROJECT:
        raise HTTPException(503, "No data project configured")

    cache_key = f"{DATA_PROJECT}:{PROFILE_BUCKET}"
    now = _time.time()
    cached_entry = _catalog_cache.get(cache_key)
    if cached_entry is not None:
        ts, cached_resp = cached_entry
        if now - ts < _CATALOG_CACHE_TTL:
            return cached_resp

    # Run GCS profile scan and BQ dataset discovery in parallel
    profile_index: dict[str, dict[str, bool]] = {}
    datasets: list[str] = []

    def _scan_profiles():
        if not PROFILE_BUCKET:
            return {}
        try:
            return scan_profile_availability(PROFILE_BUCKET, DATA_PROJECT, billing_project_id=BILLING_PROJECT)
        except Exception as e:
            print(f"Profile scan failed: {e}")
            return {}

    def _list_datasets():
        return discover_datasets(DATA_PROJECT, billing_project=BILLING_PROJECT)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_profiles = ex.submit(_scan_profiles)
        f_datasets = ex.submit(_list_datasets)

        profile_index = f_profiles.result()

        try:
            datasets = f_datasets.result()
        except Exception as e:
            print(f"Dataset discovery failed: {e}")

    # Discover tables across all datasets in parallel
    ds_tables: dict[str, list] = {}

    def _discover_ds(ds_id: str):
        return ds_id, discover_tables(DATA_PROJECT, ds_id, billing_project=BILLING_PROJECT)

    with ThreadPoolExecutor(max_workers=min(8, max(len(datasets), 1))) as ex:
        futures = {ex.submit(_discover_ds, ds): ds for ds in datasets}
        for f in as_completed(futures):
            try:
                ds_id, tables = f.result()
                ds_tables[ds_id] = tables
            except Exception as e:
                ds_id = futures[f]
                print(f"Table discovery failed for {ds_id}: {e}")
                ds_tables[ds_id] = []

    result: list[dict[str, Any]] = []
    for ds in datasets:
        tables = ds_tables.get(ds, [])
        rows = []
        for t in tables:
            fq = t.fq_name
            prof = profile_index.get(fq, {"technical": False, "semantic": False})
            ts = getattr(t, "creation_time", None)

            row_count = t.row_count
            size_bytes = t.size_bytes
            col_count = len(t.columns)

            if (row_count is None or size_bytes is None) and prof.get("technical"):
                try:
                    p_, d_, t_ = parse_fq_table(fq)
                    tech_data = read_json_if_exists(PROFILE_BUCKET, tech_object_path(p_, d_, t_), BILLING_PROJECT)
                    if tech_data:
                        if row_count is None and tech_data.get("row_count") is not None:
                            row_count = tech_data["row_count"]
                        if size_bytes is None and tech_data.get("size_bytes") is not None:
                            size_bytes = tech_data["size_bytes"]
                        if col_count == 0 and tech_data.get("columns"):
                            col_count = len(tech_data["columns"])
                    if size_bytes is None:
                        _, api_bytes = get_table_api_metadata(
                            p_, d_, t_, billing_project=BILLING_PROJECT or DATA_PROJECT
                        )
                        if api_bytes is not None:
                            size_bytes = api_bytes
                except Exception:
                    pass

            rows.append(
                TableSummary(
                    fq_table=fq,
                    project_id=t.project_id,
                    dataset_id=t.dataset_id,
                    table_id=t.table_id,
                    row_count=row_count,
                    size_bytes=size_bytes,
                    table_type=t.table_type,
                    column_count=col_count,
                    creation_time=str(ts) if ts else None,
                    profiling=_profiling_for_catalog_row(fq, prof),
                    business_name=prof.get("business_name"),
                    table_definition=prof.get("table_definition"),
                ).model_dump()
            )
        result.append({"dataset_id": ds, "tables": rows})
    response = CatalogResponse(
        project_id=DATA_PROJECT,
        profile_bucket=PROFILE_BUCKET or "",
        datasets=result,
    ).model_dump()
    _catalog_cache[cache_key] = (_time.time(), response)
    return response


# ── Table-level endpoints ─────────────────────────────────────────────────────

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
            data_project=project_id,
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
            data_project=project_id,
            model_name=GEMINI_MODEL,
            job_id=jid,
        )

    background_tasks.add_task(_job)
    return JobStartResponse(job_id=jid, status="running").model_dump()


# ── Charts / Explore ──────────────────────────────────────────────────────────

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


# ── Bulk profiling ───────────────────────────────────────────────────────────

from bulk_profiler import bulk_manager


@app.post("/api/bulk-profile")
def api_bulk_profile(body: dict[str, Any]):
    """
    Start bulk profiling.
    Body: { tables: ["fq1", ...], mode: "technical" | "semantic" | "both", force?: boolean }
    """
    tables = body.get("tables", [])
    mode = body.get("mode", "both")
    force = bool(body.get("force", False))
    if not tables:
        raise HTTPException(400, "tables list is required")
    if mode not in ("technical", "semantic", "both"):
        mode = "both"
    if not DATA_PROJECT or not PROFILE_BUCKET:
        raise HTTPException(503, "Project or bucket not configured")

    batch_id = bulk_manager.start_batch(
        tables=tables,
        mode=mode,
        bucket=PROFILE_BUCKET,
        billing_project=BILLING_PROJECT,
        data_project=DATA_PROJECT,
        model_name=GEMINI_MODEL,
        force=force,
    )
    return {"batch_id": batch_id, "total": len(tables), "mode": mode, "force": force}


@app.get("/api/bulk-profile/{batch_id}")
def api_bulk_status(batch_id: str):
    """Get bulk profiling batch status."""
    batch = bulk_manager.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    return batch.summary()


# ── Catalog context regeneration ─────────────────────────────────────────────

@app.post("/api/catalog-context/regenerate")
def api_regenerate_context():
    """Manually regenerate the catalog context .md file."""
    if not DATA_PROJECT or not PROFILE_BUCKET:
        raise HTTPException(503, "Project or bucket not configured")
    from verily_profiler.storage import regenerate_catalog_context
    from chat_handler import invalidate_context_cache
    try:
        path = regenerate_catalog_context(PROFILE_BUCKET, DATA_PROJECT, billing_project_id=BILLING_PROJECT)
        invalidate_context_cache(DATA_PROJECT, PROFILE_BUCKET)
        return {"status": "ok", "path": path}
    except Exception as e:
        raise HTTPException(500, f"Regeneration failed: {e}")


# ── Chat ─────────────────────────────────────────────────────────────────────

from chat_handler import chat_store, handle_chat_message


@app.post("/api/chat")
async def api_chat(body: dict[str, Any]):
    """
    Send a chat message.
    Body: { message, mode?, fq_table?, session_id? }
    Returns: { session_id, message: ChatMessage }
    """
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(400, "message is required")
    if not DATA_PROJECT:
        raise HTTPException(503, "No data project configured")

    mode = body.get("mode", "metadata")
    if mode not in ("metadata", "agent"):
        mode = "metadata"

    fq_table = body.get("fq_table") or None
    session_id = body.get("session_id") or None

    try:
        result = await handle_chat_message(
            message=message,
            mode=mode,
            fq_table=fq_table,
            session_id=session_id,
            data_project=DATA_PROJECT,
            billing_project=BILLING_PROJECT,
            bucket=PROFILE_BUCKET,
            model=CHAT_MODEL,
        )
        return result
    except Exception as e:
        import traceback
        print(f"Chat error:\n{traceback.format_exc()}")
        raise HTTPException(500, f"Chat failed: {e}")


@app.post("/api/chat/clear")
def api_chat_clear(body: dict[str, Any]):
    """Clear a chat session. Body: { session_id }"""
    sid = body.get("session_id", "")
    if not sid:
        raise HTTPException(400, "session_id is required")
    chat_store.clear(sid)
    return {"status": "cleared", "session_id": sid}


@app.get("/api/chat/history/{session_id}")
def api_chat_history(session_id: str):
    """Get conversation history for a session."""
    sess = chat_store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": sess.session_id,
        "mode": sess.mode,
        "messages": [m.to_json_dict() for m in sess.messages],
    }


# ── Static frontend (production build) ───────────────────────────────────────

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(404)
