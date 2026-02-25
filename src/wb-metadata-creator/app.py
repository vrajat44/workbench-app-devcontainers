"""
WB Metadata Creator — Gradio UI
Main application entry point for the FHIR Metadata Creator.

Implements a multi-step workflow:
  Tab 1: Setup — Study configuration + BQ source selection
  Tab 2: Profile & Define — Data profiling + LLM definitions → editable table
  Tab 3: Validate Inputs — LLM-as-a-Judge validation of metadata table (before JSON)
  Tab 4: Add to Cortex — Build FHIR JSON + register listing
  Tab 5: Deliver to Workbench — GCS bucket discovery + save to data collection
  Tab 6: Validate JSONs (optional) — Post-generation 8-check validation

Run:
    python app.py --project=YOUR_GCP_PROJECT_ID
    python app.py --project=YOUR_PROJECT --data-project YOUR_DATA_PROJECT
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import gradio as gr

from fhir_generator import (
    BQTableInfo,
    StudyConfig,
    ColumnMetadata,
    TableMetadata,
    REVIEW_COLUMNS,
    discover_bq_datasets,
    discover_bq_tables,
    discover_gcs_buckets,
    discover_gcs_folders,
    map_bq_type_to_fhir,
    profile_bq_table,
    generate_column_definitions,
    build_structure_definition,
    build_value_set,
    build_terminology_bundle,
    build_terminology_bundle_with_concept_maps,
    generate_concept_maps,
    generate_data_profile,
    build_measure_definition,
    TableProfile,
)
from fhir_validator import (
    apply_fix,
    format_check_status_display,
    format_input_check_status,
    format_input_validation_report,
    format_validation_report,
    run_input_validation,
    run_validation_pipeline,
)


# ── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_JSON_DIR = str(
    Path(__file__).parent.parent.parent
    / "product_mgmnt"
    / "Metadata"
    / "Metadata JSON for Demo"
    / "JSON Metadata"
)


def parse_args():
    parser = argparse.ArgumentParser(description="WB Metadata Creator")
    parser.add_argument("--project", type=str, default=os.environ.get("GCP_PROJECT_ID"),
                        help="GCP project ID for Vertex AI / LLM calls and BQ job billing")
    parser.add_argument("--data-project", type=str, nargs="+",
                        default=(os.environ.get("DATA_PROJECT_ID", "").split(",")
                                 if os.environ.get("DATA_PROJECT_ID") else None),
                        help="GCP project ID(s) where BigQuery data lives")
    parser.add_argument("--json-dir", type=str,
                        default=os.environ.get("METADATA_JSON_DIR", _DEFAULT_JSON_DIR),
                        help="Path to existing FHIR metadata (for cross-reference)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GRADIO_PORT", "7860")),
                        help="Port to run Gradio on")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--output-bucket", type=str,
                        default=os.environ.get("OUTPUT_GCS_BUCKET", ""))
    return parser.parse_args()


# ── App State ─────────────────────────────────────────────────────────────────

_config = None
_existing_metadata: list[dict] = []

# Source selection
_discovered_tables: list[BQTableInfo] = []
_selected_tables: list[BQTableInfo] = []

# Profiling state (per table)
_table_profiles: dict[str, TableProfile] = {}

# ── TYPED METADATA STORE (single source of truth) ────────────────────────────
# Replaces the old fragile triple:
#   _review_data (DataFrame)  — only 8 UI columns, lost fhir_type/fhir_mapping
#   _table_meta (dict)        — separate from column data
#   _column_enrichments (dict) — side store for LLM fields not in the DataFrame
#
# Now every column's metadata lives in ONE ColumnMetadata object inside a
# TableMetadata container. The Gradio DataFrame is just a view.
_table_metadata: dict[str, TableMetadata] = {}  # table_fq_name → TableMetadata

# Input validation
_input_validation_report = None

# Concept mapping state (generated in Tab 2 via LLM, used in Tab 4 assembly)
_concept_maps: dict[str, list[dict]] = {}  # table_name → list of ConceptMap dicts

# Generation state — final JSONs
_generated_jsons: dict[str, dict] = {}
_generated_value_sets: dict[str, list[dict]] = {}
_generated_terminology_bundles: dict[str, dict] = {}
_generated_data_profiles: dict[str, dict] = {}
_generated_measure: Optional[dict] = None  # Shared Measure definition

# JSON Validation
_json_validation_report = None


# ── Review table columns (imported from models via fhir_generator) ────────────
# REVIEW_COLUMNS is now defined in models.py and imported above.
# Alias for backward compatibility with any local references:
_REVIEW_COLUMNS = REVIEW_COLUMNS


# ── Initialization ────────────────────────────────────────────────────────────

def initialize(args):
    """Load existing metadata for cross-reference."""
    global _config, _existing_metadata
    _config = args

    if args.json_dir and os.path.exists(args.json_dir):
        print(f"📂 Loading existing metadata from: {args.json_dir}")
        _existing_metadata = _load_existing_fhir_jsons(args.json_dir)
        print(f"✓ Loaded {len(_existing_metadata)} existing FHIR profiles for cross-reference")
    elif args.json_dir and args.json_dir.startswith("gs://"):
        print(f"📂 Loading existing metadata from GCS: {args.json_dir}")
        _existing_metadata = _load_existing_fhir_jsons_gcs(args.json_dir)
        print(f"✓ Loaded {len(_existing_metadata)} existing FHIR profiles from GCS")
    else:
        print("ℹ No existing metadata path — cross-reference disabled")


def _load_existing_fhir_jsons(json_dir: str) -> list[dict]:
    profiles = []
    for json_file in sorted(Path(json_dir).rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("resourceType") == "StructureDefinition":
                profiles.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return profiles


def _load_existing_fhir_jsons_gcs(gcs_uri: str) -> list[dict]:
    try:
        from google.cloud import storage
        path = gcs_uri[5:]
        parts = path.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(parts[0])
        blobs = bucket.list_blobs(prefix=parts[1] if len(parts) > 1 else "")
        profiles = []
        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue
            try:
                data = json.loads(blob.download_as_text())
                if data.get("resourceType") == "StructureDefinition":
                    profiles.append(data)
            except Exception:
                continue
        return profiles
    except Exception as e:
        print(f"⚠ Could not load from GCS: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SETUP HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def discover_datasets(data_project_id: str):
    if not (data_project_id or "").strip():
        return gr.Dropdown(choices=[], value=[], multiselect=True)
    billing = _config.project if _config else None
    datasets = discover_bq_datasets(data_project_id.strip(), billing)
    return gr.Dropdown(choices=datasets, value=[], multiselect=True)


def discover_tables(data_project_id: str, dataset_ids: list[str]) -> tuple:
    global _discovered_tables
    if not (data_project_id or "").strip() or not dataset_ids:
        return gr.Dropdown(choices=[], value=[], multiselect=True), "No datasets selected."
    billing = _config.project if _config else None
    _discovered_tables = []
    for ds_id in dataset_ids:
        tables = discover_bq_tables(data_project_id.strip(), ds_id, billing)
        _discovered_tables.extend(tables)
    choices = [t.fq_name for t in _discovered_tables]
    summary = f"Found {len(_discovered_tables)} tables across {len(dataset_ids)} dataset(s):\n"
    for t in _discovered_tables:
        cols = len(t.columns)
        rows = f"{t.row_count:,}" if t.row_count else "?"
        summary += f"\n  📊 {t.fq_name} ({cols} columns, {rows} rows)"
    return gr.Dropdown(choices=choices, value=choices, multiselect=True), summary


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PROFILE & DEFINE HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_study_config(study_name, compliance_zone, retention_years, schema_stability, domain_contact, confidentiality):
    """Build StudyConfig from form values."""
    return StudyConfig(
        study_name=(study_name or "").strip(),
        compliance_zone=compliance_zone or "HIPAA-covered",
        retention_years=int(retention_years or 7),
        schema_stability=(schema_stability or "Stable").lower(),
        domain_contact=(domain_contact or "").strip(),
        confidentiality=confidentiality or "R",
    )


def run_profile_and_define(
    selected_table_names: list[str],
    study_name: str, compliance_zone: str, retention_years, schema_stability: str,
    domain_contact: str, confidentiality: str, data_dict_text: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple:
    """
    Combined Profile + Define: profiles BQ data, runs LLM for descriptions/sensitivity,
    and generates ConceptMaps — all in one step so the user can review everything in Tab 2.

    Uses the typed TableMetadata / ColumnMetadata model as single source of truth.
    The Gradio DataFrame is just a *view* of ColumnMetadata objects.

    Returns:
        (status_md, concept_map_md, first_df, table_dropdown, title, description, purpose, primary_key, coded_values_md, save_status, profiling_md)
    """
    global _table_profiles, _selected_tables, _concept_maps, _table_metadata

    _empty_df = pd.DataFrame(columns=_REVIEW_COLUMNS)
    _empty = ("", "", _empty_df, gr.Dropdown(choices=[]), "", "", "", "", "", "")

    if not selected_table_names:
        return ("❌ No tables selected.",) + _empty[1:]

    _selected_tables = [t for t in _discovered_tables if t.fq_name in selected_table_names]
    if not _selected_tables:
        return ("❌ Tables not found.",) + _empty[1:]

    billing = _config.project if _config else None
    project_id = _config.project if _config else None
    total = len(_selected_tables)
    study_config = _build_study_config(study_name, compliance_zone, retention_years,
                                        schema_stability, domain_contact, confidentiality)
    data_dict = (data_dict_text or "").strip()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    start = time.time()

    # ── Phase 1: BQ Profiling ────────────────────────────────────────────────
    progress(0, desc=f"Phase 1/3: Profiling {total} table(s)...")

    _table_profiles = {}
    with ThreadPoolExecutor(max_workers=min(4, total)) as ex:
        futures = {ex.submit(profile_bq_table, t, billing): t for t in _selected_tables}
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            done += 1
            progress(done / (total * 3), desc=f"[{done}/{total}] Profiled {t.table_id}")
            try:
                _table_profiles[t.fq_name] = f.result()
            except Exception as e:
                print(f"⚠ Profile failed for {t.fq_name}: {e}")
                _table_profiles[t.fq_name] = TableProfile(table_name=t.fq_name)

    # ── Build initial TableMetadata objects from BQ schema + profiling ────────
    _table_metadata = {}
    for t in _selected_tables:
        prof = _table_profiles.get(t.fq_name, TableProfile(table_name=t.fq_name))
        cols = []
        for col in t.columns:
            cp = prof.columns.get(col.column_name)
            cols.append(ColumnMetadata(
                column_name=col.column_name,
                bq_type=col.data_type,
                fhir_type=map_bq_type_to_fhir(col.data_type),
                short_label="",
                description=col.description or "",
                required=(col.is_nullable != "YES"),
                sensitivity="",
                measurement_method="",
                coded=bool(cp and 1 < cp.distinct_count <= 50),
                fhir_mapping="",
            ))
        _table_metadata[t.fq_name] = TableMetadata(
            table_fq_name=t.fq_name,
            columns=cols,
            bq_table_info=t,       # Store BQ schema reference
            profile=prof,          # Store profiling reference
        )

    profile_elapsed = time.time() - start

    # ── Phase 2: LLM Column Definitions ──────────────────────────────────────
    progress(total / (total * 3), desc=f"Phase 2/3: Generating definitions for {total} table(s)...")

    results = {}
    with ThreadPoolExecutor(max_workers=min(4, total)) as ex:
        futures = {
            ex.submit(
                generate_column_definitions,
                t,
                _table_profiles.get(t.fq_name, TableProfile(table_name=t.fq_name)),
                study_config,
                project_id,
                data_dict,
            ): t
            for t in _selected_tables
        }
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            done += 1
            progress((total + done) / (total * 3), desc=f"[{done}/{total}] Defined {t.table_id}")
            try:
                results[t.fq_name] = f.result()
            except Exception as e:
                print(f"⚠ Definitions failed for {t.fq_name}: {e}")
                results[t.fq_name] = None

    # ── Merge LLM definitions into typed ColumnMetadata objects ──
    # All validation (description quality, measurement vocab, sensitivity) is
    # handled inside ColumnMetadata.apply_llm_output — no split state.

    for t in _selected_tables:
        llm_result = results.get(t.fq_name)
        if not llm_result:
            continue

        tm = _table_metadata[t.fq_name]

        # Table-level LLM metadata
        tm.apply_llm_table_meta(llm_result.get("table", {}))

        # Column-level LLM metadata — each ColumnMetadata validates & stores everything
        llm_cols = {c["column_name"]: c for c in llm_result.get("columns", [])}
        for col_meta in tm.columns:
            llm = llm_cols.get(col_meta.column_name, {})
            if llm:
                bq_desc = next(
                    (c.description for c in t.columns if c.column_name == col_meta.column_name),
                    "",
                )
                col_meta.apply_llm_output(llm, bq_description=bq_desc or "")

    # ── Auto-populate fixed values and enrich PK comments ──────────────────
    for t in _selected_tables:
        tm = _table_metadata.get(t.fq_name)
        if tm:
            tm.auto_populate_fixed_values()  # Set fixed_value from profiling
            tm.enrich_pk_comments()           # Enrich PK column comments

    # ── Phase 3: ConceptMap Generation (LLM) ─────────────────────────────────
    _concept_maps = {}
    cm_total = 0
    if project_id:
        progress(2 * total / (total * 3), desc=f"Phase 3/3: Generating concept mappings...")
        for t in _selected_tables:
            prof = _table_profiles.get(t.fq_name)
            tm = _table_metadata.get(t.fq_name)
            if not prof or not tm:
                continue

            coded_cols_for_cm = []
            for col_meta in tm.columns:
                if col_meta.coded:
                    coded_vals = prof.columns[col_meta.column_name].top_values if col_meta.column_name in prof.columns else []
                    if coded_vals:
                        coded_cols_for_cm.append({
                            "column_name": col_meta.column_name,
                            "coded_values": coded_vals,
                            "description": col_meta.description,
                        })

            if coded_cols_for_cm:
                try:
                    cms = generate_concept_maps(
                        study_config.study_name, coded_cols_for_cm, project_id
                    )
                    _concept_maps[t.fq_name] = cms
                    cm_total += len(cms)
                    # Also store in TableMetadata for consolidated access
                    tm_for_cm = _table_metadata.get(t.fq_name)
                    if tm_for_cm:
                        tm_for_cm.concept_maps = cms
                except Exception as e:
                    print(f"⚠ ConceptMap generation skipped for {t.fq_name}: {e}")

    elapsed = time.time() - start
    table_choices = sorted(_table_metadata.keys())

    # Count sensitivity tags
    sens_count = sum(
        1 for tm in _table_metadata.values()
        for col in tm.columns
        if col.sensitivity and col.sensitivity != "NONE"
    )

    status_parts = [
        f"### ✅ Profile & Define complete — {elapsed:.1f}s\n",
        f"- 📊 Profiled {len(table_choices)} table(s) ({profile_elapsed:.1f}s)",
        f"- 🤖 Definitions generated for {sum(1 for r in results.values() if r)} / {total} table(s)",
        f"- 🔒 {sens_count} sensitivity tags assigned",
        f"- 🔗 {cm_total} concept mapping(s) generated",
        f"\n**Review** the metadata below, then proceed to **Tab 3** (validate) or **Tab 4** (generate JSONs).",
    ]
    status = "\n".join(status_parts)

    # Build concept map display
    concept_map_display = _build_concept_map_display(table_choices[0]) if table_choices else ""
    profiling_display = _build_profiling_results_display(table_choices[0]) if table_choices else ""

    first_name = table_choices[0] if table_choices else None
    first_tm = _table_metadata.get(first_name)
    first_df = first_tm.to_review_dataframe() if first_tm else _empty_df
    coded_display = _build_coded_values_display(first_name) if first_name else ""

    return (
        status,
        concept_map_display,
        first_df,
        gr.Dropdown(choices=table_choices, value=first_name),
        first_tm.title if first_tm else "",
        first_tm.description if first_tm else "",
        first_tm.purpose if first_tm else "",
        first_tm.primary_key if first_tm else "",
        coded_display,
        "",
        profiling_display,
    )


def on_table_selected(table_name: str) -> tuple:
    """When user picks a different table from the dropdown."""
    _empty_df = pd.DataFrame(columns=_REVIEW_COLUMNS)
    tm = _table_metadata.get(table_name)
    if not table_name or not tm:
        return _empty_df, "", "", "", "", "", "", ""
    df = tm.to_review_dataframe()
    coded = _build_coded_values_display(table_name)
    cm_display = _build_concept_map_display(table_name)
    prof_display = _build_profiling_results_display(table_name)
    return (
        df,
        tm.title,
        tm.description,
        tm.purpose,
        tm.primary_key,
        coded,
        cm_display,
        prof_display,
    )


def save_review_edits(table_name: str, df: pd.DataFrame,
                       title: str, description: str, purpose: str, primary_key: str) -> str:
    """
    Save edits from the Gradio review table back to the typed model.

    Writes DataFrame edits → ColumnMetadata objects, and text fields → TableMetadata.
    fhir_type and fhir_mapping are preserved (they live in ColumnMetadata, not the DF).
    """
    if not table_name:
        return "❌ No table selected."

    tm = _table_metadata.get(table_name)
    if not tm:
        return "❌ Table not found in metadata store."

    # Update table-level fields
    tm.title = (title or "").strip()
    tm.description = (description or "").strip()
    tm.purpose = (purpose or "").strip()
    tm.primary_key = (primary_key or "").strip()

    # Update column-level fields from the DataFrame (preserves fhir_type/fhir_mapping!)
    tm.update_from_review_dataframe(df)

    return f"✅ Saved edits for **{table_name}** ({len(tm.columns)} columns)."


def _build_coded_values_display(table_name: str) -> str:
    """Build markdown showing coded column values for a table."""
    tm = _table_metadata.get(table_name)
    prof = tm.profile if tm else _table_profiles.get(table_name)
    if not prof:
        return ""
    lines = ["#### Coded Column Values (from data profiling)\n"]
    has_any = False
    for cn, cp in prof.columns.items():
        if cp.top_values:
            has_any = True
            vals = ", ".join(f"`{v}`" for v in cp.top_values[:15])
            if len(cp.top_values) > 15:
                vals += f" ... +{len(cp.top_values)-15} more"
            lines.append(f"- **{cn}** ({cp.distinct_count} values): {vals}")
    if not has_any:
        lines.append("*No coded columns detected (all columns have >50 distinct values)*")
    return "\n".join(lines)


def _build_profiling_results_display(table_name: str) -> str:
    """Build markdown showing detailed profiling results for a table (min/max, etc.)."""
    tm = _table_metadata.get(table_name)
    prof = tm.profile if tm else _table_profiles.get(table_name)
    if not prof:
        return "*No profiling data available.*"

    from bq_profiler import _NUMERIC_BQ_TYPES
    table_info = tm.bq_table_info if tm else next((t for t in _selected_tables if t.fq_name == table_name), None)
    if not table_info:
        return "*No profiling data available.*"

    lines = [
        f"#### 📊 Data Profile — {prof.total_rows:,} rows\n",
        "| Column | Null % | Distinct | Type | Stats |",
        "|--------|-------:|--------:|------|-------|",
    ]

    for col in table_info.columns:
        cp = prof.columns.get(col.column_name)
        if not cp:
            continue

        bq_upper = col.data_type.upper().split("<")[0].strip()
        is_numeric = bq_upper in _NUMERIC_BQ_TYPES

        stats_parts = []
        if is_numeric:
            if cp.min_value is not None:
                stats_parts.append(f"min={cp.min_value:g}")
            if cp.max_value is not None:
                stats_parts.append(f"max={cp.max_value:g}")
            if cp.median is not None:
                stats_parts.append(f"med={cp.median:g}")
            if cp.stddev is not None:
                stats_parts.append(f"σ={cp.stddev:g}")
        else:
            if cp.min_length is not None:
                stats_parts.append(f"len={cp.min_length}-{cp.max_length}")
            if cp.avg_length is not None:
                stats_parts.append(f"avg_len={cp.avg_length}")
            if cp.top_values:
                top3 = ", ".join(f"`{v}`" for v in cp.top_values[:3])
                stats_parts.append(f"top: {top3}")

        stats = ", ".join(stats_parts) if stats_parts else "—"
        kind = "numeric" if is_numeric else "string"
        if cp.top_values and 1 < cp.distinct_count <= 50:
            kind = "coded"

        lines.append(
            f"| {col.column_name} | {cp.null_percent}% | {cp.distinct_count} | {kind} | {stats} |"
        )

    return "\n".join(lines)


def _build_concept_map_display(table_name: str) -> str:
    """Build markdown showing concept mappings for a table (generated in Tab 2)."""
    cms = _concept_maps.get(table_name, [])
    if not cms:
        return "*No concept mappings generated for this table.*"

    lines = ["#### 🔗 Concept Mappings (LLM-generated — review before generating JSONs)\n"]
    for cm in cms:
        col = cm.get("_column_name", cm.get("title", "?"))
        confidence = cm.get("_confidence", "?")
        target_name = cm.get("_target_system_name", "")
        badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")

        # Extract group info
        groups = cm.get("group", [])
        if groups:
            target_sys = groups[0].get("target", "")
            elements = groups[0].get("element", [])
            lines.append(f"**{col}** → {target_name or target_sys} {badge} *{confidence} confidence*")
            for el in elements[:8]:  # Show top 8 mappings
                src_code = el.get("code", "?")
                src_disp = el.get("display", src_code)
                targets = el.get("target", [])
                if targets:
                    tgt = targets[0]
                    tgt_code = tgt.get("code", "?")
                    tgt_disp = tgt.get("display", tgt_code)
                    equiv = tgt.get("equivalence", "?")
                    lines.append(f"  - `{src_code}` ({src_disp}) → `{tgt_code}` ({tgt_disp}) — *{equiv}*")
            if len(elements) > 8:
                lines.append(f"  - ... +{len(elements)-8} more mappings")
            lines.append("")
        else:
            lines.append(f"**{col}** — no mapping groups found\n")

    return "\n".join(lines)


# ── Helper: get review DataFrame for a table ─────────────────────────────────

def _get_review_df(table_name: str) -> pd.DataFrame:
    """Return the review DataFrame for a table, or an empty one."""
    _empty_df = pd.DataFrame(columns=_REVIEW_COLUMNS)
    tm = _table_metadata.get(table_name)
    return tm.to_review_dataframe() if tm else _empty_df


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VALIDATE INPUTS (metadata table, before JSON generation)
# ══════════════════════════════════════════════════════════════════════════════

def run_input_validation_handler(
    progress=gr.Progress(track_tqdm=False),
) -> tuple:
    """Run LLM-as-a-Judge validation on the metadata table inputs."""
    global _input_validation_report

    if not _table_metadata:
        return "❌ No metadata to validate. Complete Tab 2 first.", ""

    project_id = _config.project if _config else None

    # Build the maps for validation — derive from typed model
    table_meta_map = {}
    columns_df_records_map = {}
    for tbl_name, tm in _table_metadata.items():
        table_meta_map[tbl_name] = tm.to_table_meta_dict()
        columns_df_records_map[tbl_name] = [col.to_review_row() for col in tm.columns]

    total_checks = 5
    progress(0, desc="Validating metadata inputs...")

    def _progress_cb(check_num, check_name, status):
        if status == "running":
            progress(check_num / total_checks, desc=f"Running check {check_num}: {check_name}...")
        else:
            progress(check_num / total_checks, desc=f"Check {check_num}: {check_name} — {status}")

    _input_validation_report = run_input_validation(
        table_meta_map=table_meta_map,
        columns_df_records_map=columns_df_records_map,
        project_id=project_id,
        progress_callback=_progress_cb,
    )

    check_display = format_input_check_status(_input_validation_report)
    full_report = format_input_validation_report(_input_validation_report)
    return check_display, full_report


def _apply_suggestion_to_column(col: ColumnMetadata, check_name: str, suggested: str) -> None:
    """Apply a single validation suggestion to a ColumnMetadata object."""
    if "Description" in check_name:
        col.description = suggested
    elif "Sensitivity" in check_name or "Security" in check_name:
        col.sensitivity = suggested.strip().upper()
    elif "Type" in check_name and "Coded" in check_name:
        if suggested.lower() in ("yes", "no"):
            col.coded = suggested.lower() == "yes"
    elif "Measurement" in check_name:
        col.measurement_method = suggested
    else:
        col.description = suggested


def apply_input_suggestion(issue_index: int, review_selector_value: str) -> tuple:
    """Apply a suggested fix from input validation to the typed model."""
    _empty_df = pd.DataFrame(columns=_REVIEW_COLUMNS)
    if not _input_validation_report:
        return "No validation report.", _empty_df

    issues = _input_validation_report.input_issues
    idx = int(issue_index)
    if idx < 0 or idx >= len(issues):
        return f"❌ Invalid issue index: {idx}", _get_review_df(review_selector_value)

    issue = issues[idx]

    if issue.applied:
        return f"⏭ Issue #{idx} already applied.", _get_review_df(review_selector_value)

    if issue.skipped:
        return f"⏭ Issue #{idx} was skipped.", _get_review_df(review_selector_value)

    col_name = issue.column_name
    suggested = issue.suggested_value

    if not col_name or not suggested:
        issue.skipped = True
        return f"⏭ Issue #{idx}: No specific column/value to apply — review manually.", _get_review_df(review_selector_value)

    # Find which table this column belongs to and apply directly to ColumnMetadata
    applied = False
    target_table = None
    for tbl_name, tm in _table_metadata.items():
        col = tm.get_column(col_name)
        if col:
            target_table = tbl_name
            _apply_suggestion_to_column(col, issue.check_name, suggested)
            applied = True
            break

    if applied:
        issue.applied = True
        return f"✅ Applied suggestion for **{col_name}** in {target_table}.", _get_review_df(review_selector_value)
    else:
        issue.skipped = True
        return f"⚠ Could not find column **{col_name}** in any table.", _get_review_df(review_selector_value)


def skip_input_issue(issue_index: int) -> str:
    """Skip an input validation issue."""
    if not _input_validation_report:
        return "No validation report."
    issues = _input_validation_report.input_issues
    idx = int(issue_index)
    if idx < 0 or idx >= len(issues):
        return f"Invalid issue index: {idx}"
    issues[idx].skipped = True
    return f"⏭ Skipped issue #{idx}"


def apply_all_input_fixes(review_selector_value: str) -> tuple:
    """Apply all pending suggested fixes from input validation in one batch."""
    if not _input_validation_report:
        return "❌ No validation report.", _get_review_df(review_selector_value)

    issues = _input_validation_report.input_issues
    applied = 0
    skipped = 0

    for idx, issue in enumerate(issues):
        if issue.applied or issue.skipped:
            continue

        col_name = issue.column_name
        suggested = issue.suggested_value

        if not col_name or not suggested:
            issue.skipped = True
            skipped += 1
            continue

        # Find the column in the typed model and apply directly
        found = False
        for tbl_name, tm in _table_metadata.items():
            col = tm.get_column(col_name)
            if col:
                _apply_suggestion_to_column(col, issue.check_name, suggested)
                found = True
                break

        if found:
            issue.applied = True
            applied += 1
        else:
            issue.skipped = True
            skipped += 1

    return f"✅ Applied **{applied}** fix(es), skipped **{skipped}**. Review changes in Tab 2.", _get_review_df(review_selector_value)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GENERATE JSON HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_jsons(
    study_name: str, compliance_zone: str, retention_years, schema_stability: str,
    domain_contact: str, confidentiality: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple:
    """Build FHIR JSONs programmatically from the reviewed metadata table."""
    global _generated_jsons, _generated_value_sets, _generated_terminology_bundles, _generated_data_profiles, _generated_measure

    if not _table_metadata:
        return "❌ No reviewed metadata. Go to Tab 2 first.", "", gr.Dropdown(choices=[])

    try:
        return _generate_jsons_inner(study_name, compliance_zone, retention_years,
                                      schema_stability, domain_contact, confidentiality, progress)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ generate_jsons error:\n{tb}")
        return f"❌ **Error generating JSONs:**\n```\n{tb}\n```", "", gr.Dropdown(choices=[])


def _generate_jsons_inner(
    study_name: str, compliance_zone: str, retention_years, schema_stability: str,
    domain_contact: str, confidentiality: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple:
    """Inner implementation — errors bubble up to generate_jsons for display."""
    global _generated_jsons, _generated_value_sets, _generated_terminology_bundles, _generated_data_profiles, _generated_measure

    study_config = _build_study_config(study_name, compliance_zone, retention_years,
                                        schema_stability, domain_contact, confidentiality)

    _generated_jsons = {}
    _generated_value_sets = {}
    _generated_terminology_bundles = {}
    _generated_data_profiles = {}

    # Generate the shared Measure definition (once for all tables)
    _generated_measure = build_measure_definition()

    total = len(_table_metadata)
    progress(0, desc=f"Building JSONs for {total} table(s)...")

    for i, (tbl_name, tm) in enumerate(_table_metadata.items()):
        progress((i + 1) / total, desc=f"[{i+1}/{total}] {tbl_name}")

        table_info = tm.bq_table_info
        if not table_info:
            continue

        # ── Get typed metadata directly — no DataFrame round-trip, no re-enrichment ──
        table_meta = tm.to_table_meta_dict()
        columns_meta = tm.to_builder_columns()
        # columns_meta already includes FHIR Type and FHIR Mapping from ColumnMetadata!

        # Build StructureDefinition (no LLM!)
        sd = build_structure_definition(table_info, study_config, table_meta, columns_meta)
        _generated_jsons[tbl_name] = sd

        # Build ValueSets for coded columns (no LLM!)
        prof = tm.profile
        value_sets = []
        profile_id = sd.get("id", "")
        for col_meta in tm.columns:
            if col_meta.coded:
                coded_vals = prof.columns[col_meta.column_name].top_values if prof and col_meta.column_name in prof.columns else []
                if coded_vals:
                    vs = build_value_set(profile_id, col_meta.column_name, coded_vals, study_config.study_name)
                    value_sets.append(vs)
        _generated_value_sets[tbl_name] = value_sets

        # Build Terminology Bundle (CodeSystem + ValueSet + ConceptMap for each coded column)
        coded_cols_for_bundle = []
        for col_meta in tm.columns:
            if col_meta.coded:
                coded_vals = prof.columns[col_meta.column_name].top_values if prof and col_meta.column_name in prof.columns else []
                if coded_vals:
                    coded_cols_for_bundle.append({
                        "column_name": col_meta.column_name,
                        "coded_values": coded_vals,
                        "description": col_meta.description,
                    })
        if coded_cols_for_bundle:
            # Use pre-generated ConceptMaps from Tab 2 (or stored in TableMetadata)
            concept_maps = tm.concept_maps

            term_bundle = build_terminology_bundle_with_concept_maps(
                profile_id, study_config.study_name, coded_cols_for_bundle, concept_maps
            )
            if term_bundle:
                _generated_terminology_bundles[tbl_name] = term_bundle

        # Build DataProfile (no LLM!) — pass profiling data for per-column metrics
        dp = generate_data_profile(
            table_info, profile_id,
            table_profile=prof,
            columns_meta=columns_meta,
            sd_title=tm.title,
        )
        if dp:
            _generated_data_profiles[tbl_name] = dp

    # Summary
    summary_parts = [f"### ✅ Generated FHIR JSONs for {len(_generated_jsons)} table(s)\n"]
    total_vs = 0
    total_cm = 0
    for name, sd in _generated_jsons.items():
        col_count = len([e for e in sd.get("differential", {}).get("element", []) if "." in e.get("path", "")])
        vs_count = len(_generated_value_sets.get(name, []))
        total_vs += vs_count
        dp_flag = "✓" if name in _generated_data_profiles else "—"
        tb = _generated_terminology_bundles.get(name)
        tb_flag = "—"
        cm_count = 0
        if tb:
            cm_count = sum(1 for e in tb.get("entry", []) if e.get("resource", {}).get("resourceType") == "ConceptMap")
            total_cm += cm_count
            tb_flag = f"✓ ({len(tb.get('entry', []))} resources, {cm_count} ConceptMaps)"

        # Count sensitivity-tagged columns
        sens_count = sum(1 for e in sd.get("differential", {}).get("element", [])
                         if any("security-label-ds4p" in ext.get("url", "") for ext in e.get("extension", [])))

        summary_parts.append(
            f"- **{sd.get('id', name)}**: {col_count} columns, "
            f"{sens_count} sensitivity tags, "
            f"TermBundle: {tb_flag}, DataProfile: {dp_flag}"
        )

    total_files = len(_generated_jsons) + total_vs + len(_generated_terminology_bundles) + len(_generated_data_profiles) + (1 if _generated_measure else 0)
    summary_parts.append(f"\n**Total: {total_files} files** ({total_cm} ConceptMaps via LLM, rest built programmatically)")
    if _generated_measure:
        summary_parts.append(f"- 📐 Shared Measure definition: `{_generated_measure.get('id', '?')}`")
    summary = "\n".join(summary_parts)

    first_name = next(iter(_generated_jsons), None)
    first_json = json.dumps(_generated_jsons[first_name], indent=2) if first_name else ""
    table_choices = sorted(_generated_jsons.keys())

    return (
        summary,
        first_json,
        gr.Dropdown(choices=table_choices, value=first_name),
    )


def show_generated_json(table_name: str) -> str:
    """Show generated JSON for a table."""
    if not table_name or table_name not in _generated_jsons:
        return ""
    return json.dumps(_generated_jsons[table_name], indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: JSON VALIDATION HANDLERS (optional, post-generation)
# ══════════════════════════════════════════════════════════════════════════════

def run_json_validation(progress=gr.Progress(track_tqdm=False)) -> tuple:
    """Run the full 8-check LLM-as-a-Judge validation on generated JSONs."""
    global _json_validation_report
    if not _generated_jsons:
        return "❌ No metadata generated yet. Generate JSONs in 'Add to Cortex' tab first.", ""

    project_id = _config.project if _config else None
    fhir_jsons = list(_generated_jsons.values())
    bq_tables = [t for t in _discovered_tables if t.fq_name in _generated_jsons]

    total_checks = 8
    progress(0, desc="Validating generated JSONs...")

    def _progress_cb(check_num, check_name, status):
        if status == "running":
            progress(check_num / total_checks, desc=f"Running check {check_num}: {check_name}...")
        else:
            progress(check_num / total_checks, desc=f"Check {check_num}: {check_name} — {status}")

    _json_validation_report = run_validation_pipeline(
        fhir_jsons=fhir_jsons,
        bq_tables=bq_tables,
        project_id=project_id,
        progress_callback=_progress_cb,
    )
    return format_check_status_display(_json_validation_report), format_validation_report(_json_validation_report)


def apply_json_fix(issue_index: int) -> tuple:
    """Apply a fix from JSON validation."""
    if not _json_validation_report:
        return "No validation report.", ""
    all_issues = _json_validation_report.all_issues
    idx = int(issue_index)
    if idx < 0 or idx >= len(all_issues):
        return f"Invalid issue index: {idx}", ""

    issue = all_issues[idx]
    target_json = target_name = None
    for name, obj in _generated_jsons.items():
        bq_name = ""
        for ext in obj.get("extension", []):
            if "BigQueryTableSchemaMetadata" in ext.get("url", ""):
                for sub in ext.get("extension", []):
                    if sub.get("url") == "table-name":
                        bq_name = sub.get("valueString", "")
        if issue.table_name in name or issue.table_name in bq_name or issue.table_name == obj.get("id", ""):
            target_json, target_name = obj, name
            break
    if not target_json:
        for name, obj in _generated_jsons.items():
            target_json, target_name = obj, name
            break
    if not target_json:
        return f"Could not find JSON for {issue.table_name}", ""

    project_id = _config.project if _config else None
    fixed = apply_fix(issue, target_json, project_id)
    if fixed:
        _generated_jsons[target_name] = fixed
        issue.applied = True
        updated_json = json.dumps(fixed, indent=2)
        return f"✅ Fix applied for {issue.table_name}.{issue.column_name or ''}", updated_json
    return f"❌ Fix failed for {issue.table_name}.{issue.column_name or ''}", ""


def skip_json_issue(issue_index: int) -> str:
    """Skip a JSON validation issue."""
    if not _json_validation_report:
        return "No validation report."
    all_issues = _json_validation_report.all_issues
    idx = int(issue_index)
    if idx < 0 or idx >= len(all_issues):
        return f"Invalid issue index: {idx}"
    all_issues[idx].skipped = True
    return f"⏭ Skipped issue #{idx}"


def apply_all_json_fixes(progress=gr.Progress(track_tqdm=False)) -> tuple:
    """Apply all pending LLM-generated fixes for JSON validation issues in one batch."""
    if not _json_validation_report:
        return "❌ No validation report.", ""

    all_issues = _json_validation_report.all_issues
    pending = [i for i in range(len(all_issues)) if not all_issues[i].applied and not all_issues[i].skipped]

    if not pending:
        return "ℹ️ No pending issues to fix.", ""

    project_id = _config.project if _config else None
    applied = 0
    failed = 0
    last_json = ""

    for count, idx in enumerate(pending):
        progress((count + 1) / len(pending), desc=f"Fixing issue {count + 1}/{len(pending)}...")
        issue = all_issues[idx]

        # Find the target JSON
        target_json = target_name = None
        for name, obj in _generated_jsons.items():
            bq_name = ""
            for ext in obj.get("extension", []):
                if "BigQueryTableSchemaMetadata" in ext.get("url", ""):
                    for sub in ext.get("extension", []):
                        if sub.get("url") == "table-name":
                            bq_name = sub.get("valueString", "")
            if issue.table_name in name or issue.table_name in bq_name or issue.table_name == obj.get("id", ""):
                target_json, target_name = obj, name
                break
        if not target_json:
            for name, obj in _generated_jsons.items():
                target_json, target_name = obj, name
                break
        if not target_json:
            issue.skipped = True
            failed += 1
            continue

        fixed = apply_fix(issue, target_json, project_id)
        if fixed:
            _generated_jsons[target_name] = fixed
            issue.applied = True
            applied += 1
            last_json = json.dumps(fixed, indent=2)
        else:
            failed += 1

    status = f"✅ Applied **{applied}** fix(es)"
    if failed:
        status += f", **{failed}** could not be applied"
    return status, last_json


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CORTEX REGISTRATION HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def register_to_cortex_handler(name: str, listing_type: str, visibility: str) -> str:
    """Fake Cortex registration — shows a success message using listing info."""
    if not _generated_jsons:
        return "❌ Generate FHIR JSONs first before registering to Cortex."

    display_name = (name or "").strip() or "Untitled Listing"
    lt = (listing_type or "Study").replace(" (WIP)", "")
    vis = visibility or "Internal"

    import time
    time.sleep(1.5)  # Simulate registration latency

    vis_text = (
        "**externally discoverable**"
        if vis == "External"
        else "visible to **your team**"
    )
    return (
        f"### ✅ {lt} *{display_name}* added to Cortex!\n\n"
        f"This {lt.lower()} is now registered and {vis_text} in the "
        "[Verily Cortex Data Catalog](https://cortex.verily.com).\n\n"
        f"**Listing type:** {lt}  \n"
        f"**Visibility:** {vis}  \n"
        f"**Tables:** {len(_generated_jsons)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DELIVER TO WORKBENCH HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def discover_buckets_handler() -> gr.Dropdown:
    """List GCS buckets in the configured project."""
    project_id = _config.project if _config else None
    if not project_id:
        return gr.Dropdown(choices=["(no project configured)"], value=None)
    buckets = discover_gcs_buckets(project_id)
    if not buckets:
        return gr.Dropdown(choices=["(no buckets found)"], value=None)
    return gr.Dropdown(choices=buckets, value=buckets[0] if buckets else None)


def discover_folders_handler(bucket_name: str) -> gr.Dropdown:
    """List top-level folders in the selected GCS bucket."""
    if not bucket_name or bucket_name.startswith("("):
        return gr.Dropdown(choices=[], value=None)
    folders = discover_gcs_folders(bucket_name)
    # Add option to save at root
    choices = ["(root)"] + folders
    return gr.Dropdown(choices=choices, value="(root)")


def save_to_gcs(bucket_name: str, folder: str, custom_prefix: str) -> str:
    if not _generated_jsons:
        return "❌ No metadata to save. Generate JSONs in the 'Add to Cortex' tab first."
    if not (bucket_name or "").strip() or bucket_name.startswith("("):
        return "❌ Select a GCS bucket first."

    bucket_clean = bucket_name.strip().replace("gs://", "")

    # Build prefix from folder selection + custom prefix
    parts = []
    if folder and folder != "(root)":
        parts.append(folder.rstrip("/"))
    if (custom_prefix or "").strip():
        parts.append(custom_prefix.strip().strip("/"))
    prefix = "/".join(parts)
    if prefix:
        prefix += "/"

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_clean)
        saved = []
        for name, obj in _generated_jsons.items():
            sd_id = obj.get("id", name.replace(".", "_"))
            bn = f"{prefix}{sd_id}.json"
            bucket.blob(bn).upload_from_string(json.dumps(obj, indent=2), content_type="application/json")
            saved.append(bn)
        for name, vs_list in _generated_value_sets.items():
            for vs in vs_list:
                bn = f"{prefix}Terminology/{vs.get('id', 'unknown')}.json"
                bucket.blob(bn).upload_from_string(json.dumps(vs, indent=2), content_type="application/json")
                saved.append(bn)
        for name, tb in _generated_terminology_bundles.items():
            bn = f"{prefix}Terminology/{tb.get('id', 'unknown')}.json"
            bucket.blob(bn).upload_from_string(json.dumps(tb, indent=2), content_type="application/json")
            saved.append(bn)
        for name, dp in _generated_data_profiles.items():
            bn = f"{prefix}{dp.get('id', name.replace('.', '_'))}.json"
            bucket.blob(bn).upload_from_string(json.dumps(dp, indent=2), content_type="application/json")
            saved.append(bn)
        if _generated_measure:
            bn = f"{prefix}{_generated_measure.get('id', 'data-profile-tabular')}.json"
            bucket.blob(bn).upload_from_string(json.dumps(_generated_measure, indent=2), content_type="application/json")
            saved.append(bn)
        return (
            f"### ✅ Saved {len(saved)} files to `gs://{bucket_clean}/{prefix}`\n\n"
            + "\n".join(f"- 📄 `{f}`" for f in saved)
        )
    except Exception as e:
        return f"❌ GCS save failed: {e}"


def download_as_zip() -> Optional[str]:
    if not _generated_jsons:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, obj in _generated_jsons.items():
            zf.writestr(f"{obj.get('id', name.replace('.', '_'))}.json", json.dumps(obj, indent=2))
        for name, vs_list in _generated_value_sets.items():
            for vs in vs_list:
                zf.writestr(f"Terminology/{vs.get('id', 'unknown')}.json", json.dumps(vs, indent=2))
        for name, tb in _generated_terminology_bundles.items():
            zf.writestr(f"Terminology/{tb.get('id', 'unknown')}.json", json.dumps(tb, indent=2))
        for name, dp in _generated_data_profiles.items():
            zf.writestr(f"{dp.get('id', name.replace('.', '_'))}.json", json.dumps(dp, indent=2))
        if _generated_measure:
            zf.writestr(f"{_generated_measure.get('id', 'data-profile-tabular')}.json",
                         json.dumps(_generated_measure, indent=2))
    return tmp.name


def get_export_file_list() -> str:
    if not _generated_jsons:
        return "*No files generated yet. Generate JSONs in the 'Add to Cortex' tab first.*"
    lines = ["### Files to Export\n"]
    lines.append("**StructureDefinitions:**")
    for name, obj in sorted(_generated_jsons.items()):
        lines.append(f"- 📄 `{obj.get('id', '?')}.json` — {obj.get('title', '?')}")
    all_vs = []
    for vs_list in _generated_value_sets.values():
        all_vs.extend(vs_list)
    if all_vs:
        lines.append("\n**ValueSets:**")
        for vs in all_vs:
            lines.append(f"- 📋 `Terminology/{vs.get('id', '?')}.json` — {vs.get('title', '?')}")
    if _generated_terminology_bundles:
        lines.append("\n**Terminology Bundles:**")
        for name, tb in sorted(_generated_terminology_bundles.items()):
            entries = tb.get("entry", [])
            cs_count = sum(1 for e in entries if e.get("resource", {}).get("resourceType") == "CodeSystem")
            vs_count = sum(1 for e in entries if e.get("resource", {}).get("resourceType") == "ValueSet")
            cm_count = sum(1 for e in entries if e.get("resource", {}).get("resourceType") == "ConceptMap")
            detail = f"{cs_count} CodeSystems, {vs_count} ValueSets"
            if cm_count:
                detail += f", {cm_count} ConceptMaps"
            lines.append(f"- 📦 `Terminology/{tb.get('id', '?')}.json` — {detail}")
    if _generated_data_profiles:
        lines.append("\n**DataProfiles:**")
        for name, dp in sorted(_generated_data_profiles.items()):
            lines.append(f"- 📊 `{dp.get('id', '?')}.json`")
    if _generated_measure:
        lines.append("\n**Measure Definition:**")
        lines.append(f"- 📐 `{_generated_measure.get('id', '?')}.json` — shared profiling metrics template")
    total = len(_generated_jsons) + len(all_vs) + len(_generated_terminology_bundles) + len(_generated_data_profiles) + (1 if _generated_measure else 0)
    lines.append(f"\n**Total: {total} files**")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="Catalog Listing Creator") as app:
        gr.Markdown("# 📋 Catalog Listing Creator\n**Setup → Profile & Define → Validate → Add to Cortex → Deliver to Workbench**")
        mode_text = (
            "🟢 **Full Mode** (BigQuery + Gemini)"
            if _config and _config.project
            else "🟡 **Limited Mode** (no GCP project — pass `--project` to enable)"
        )
        gr.Markdown(mode_text)

        # ══════════════════════════════════════════════════════════════
        # TAB 1: SETUP
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("1. Setup"):
            # ── Section 1: Listing Creation ──
            gr.Markdown("### Listing Creation")
            with gr.Row():
                with gr.Column(scale=1):
                    listing_visibility = gr.Radio(
                        choices=["Internal", "External"],
                        value="Internal",
                        label="Visibility",
                        info="Internal listings are only visible to your team. External listings are discoverable outside your organization.",
                    )
                with gr.Column(scale=2):
                    listing_type = gr.Radio(
                        choices=["Study", "Data Product", "AI Model (WIP)", "AI Agent (WIP)"],
                        value="Study",
                        label="Listing Type",
                        info="Select the type of listing to create. AI Model and AI Agent are coming soon.",
                        interactive=True,
                    )

            # Enforce WIP items snap back to previous valid choice
            def _enforce_listing_type(val):
                if val and "(WIP)" in val:
                    gr.Warning(f"{val} is not yet available. Please select Study or Data Product.")
                    return gr.Radio(value="Study")
                return gr.Radio(value=val)
            listing_type.change(fn=_enforce_listing_type, inputs=[listing_type], outputs=[listing_type])

            gr.Markdown("---")

            # ── Section 2: Configuration ──
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Study / Data Product Configuration")
                    study_name = gr.Textbox(label="Name", placeholder="e.g., BHS, PRESCO, CDC_Natality")
                    business_intent = gr.Textbox(label="Business Intent", placeholder="e.g., Support maternal morbidity research across CDC natality data", lines=2)
                    domain_selector = gr.Dropdown(
                        choices=["Lightpath", "Base", "Lifelong", "Verily-Org", "Exchange", "Analysis"],
                        value="Base",
                        label="Domain",
                        interactive=True,
                        allow_custom_value=False,
                    )
                    compliance_zone = gr.Dropdown(choices=["HIPAA-covered", "De-identified", "Public"], value="HIPAA-covered", label="Compliance Zone")
                    retention_years = gr.Number(value=7, label="Retention Policy (years)", minimum=1)
                    schema_stability = gr.Dropdown(choices=["Stable", "Draft", "Experimental"], value="Stable", label="Schema Stability")
                    domain_contact = gr.Textbox(label="Domain Contact", placeholder="e.g., domain:bhs-research")
                    confidentiality = gr.Dropdown(choices=[("Restricted", "R"), ("Normal", "N")], value="R", label="Confidentiality Level")

                with gr.Column(scale=2):
                    gr.Markdown("### BigQuery Source Selection")
                    data_project_input = gr.Textbox(
                        label="Data Project ID", placeholder="GCP project where BigQuery data lives",
                        value=(_config.data_project[0] if _config and _config.data_project and _config.data_project[0] else ""),
                    )
                    discover_ds_btn = gr.Button("🔍 Discover Datasets", variant="secondary")
                    dataset_selector = gr.Dropdown(choices=[], multiselect=True, label="Select Datasets")
                    discover_tbl_btn = gr.Button("🔍 Discover Tables", variant="secondary")
                    table_selector = gr.Dropdown(choices=[], multiselect=True, label="Select Tables")
                    discovery_status = gr.Markdown("*Click 'Discover Datasets' to start*")

            with gr.Accordion("📖 Data Dictionary (optional enrichment)", open=False):
                data_dict_input = gr.Textbox(label="Paste Data Dictionary", placeholder="Paste data dictionary (markdown, CSV, or plain text)...", lines=10)
                data_dict_upload = gr.File(label="Or Upload Data Dictionary File", file_types=[".md", ".csv", ".txt", ".pdf"])

            discover_ds_btn.click(fn=discover_datasets, inputs=[data_project_input], outputs=[dataset_selector])
            discover_tbl_btn.click(fn=discover_tables, inputs=[data_project_input, dataset_selector], outputs=[table_selector, discovery_status])

            def handle_file_upload(file):
                if file is None:
                    return ""
                try:
                    filepath = file if isinstance(file, str) else file.name
                    with open(filepath, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception:
                    return "(Could not read file)"
            data_dict_upload.change(fn=handle_file_upload, inputs=[data_dict_upload], outputs=[data_dict_input])

        # ══════════════════════════════════════════════════════════════
        # TAB 2: PROFILE & DEFINE
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("2. Profile & Define"):
            gr.Markdown(
                "### Profile & Define\n"
                "Profiles BigQuery tables, generates AI descriptions, sensitivity tags, "
                "and concept mappings — all in one step."
            )
            profile_define_btn = gr.Button("🚀 Profile & Define", variant="primary", size="lg")
            profile_define_status = gr.Markdown("*Select tables in Tab 1, then click Profile & Define*")

            gr.Markdown("---")
            gr.Markdown("### Review & Edit Metadata")
            review_table_selector = gr.Dropdown(choices=[], label="Select Table")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### Table-Level Metadata")
                    with gr.Row():
                        tbl_title = gr.Textbox(label="Title", scale=2)
                        tbl_primary_key = gr.Textbox(label="Primary Key", scale=1)
                    tbl_description = gr.Textbox(label="Description", lines=2)
                    tbl_purpose = gr.Textbox(label="Purpose (granularity)", placeholder="One record per ...")
                with gr.Column(scale=1):
                    coded_values_display = gr.Markdown("", label="Coded Values")

            gr.Markdown("#### Column Metadata *(click a cell to edit)*")
            columns_dataframe = gr.Dataframe(
                headers=_REVIEW_COLUMNS,
                interactive=True,
                wrap=True,
                column_widths=["130px", "90px", "140px", "320px", "65px", "100px", "120px", "60px"],
            )
            with gr.Row():
                save_review_btn = gr.Button("💾 Save Edits", variant="primary", size="sm")
                review_save_status = gr.Markdown("")

            gr.Markdown("---")
            profiling_results_display = gr.Markdown("", label="Profiling Results")
            gr.Markdown("---")
            concept_map_display = gr.Markdown("", label="Concept Mappings")

            # ── Tab 2 Event handlers ──
            profile_define_btn.click(
                fn=run_profile_and_define,
                inputs=[table_selector, study_name, compliance_zone, retention_years, schema_stability, domain_contact, confidentiality, data_dict_input],
                outputs=[profile_define_status, concept_map_display, columns_dataframe, review_table_selector,
                         tbl_title, tbl_description, tbl_purpose, tbl_primary_key, coded_values_display, review_save_status, profiling_results_display],
            )
            review_table_selector.change(
                fn=on_table_selected,
                inputs=[review_table_selector],
                outputs=[columns_dataframe, tbl_title, tbl_description, tbl_purpose, tbl_primary_key, coded_values_display, concept_map_display, profiling_results_display],
            )
            save_review_btn.click(
                fn=save_review_edits,
                inputs=[review_table_selector, columns_dataframe, tbl_title, tbl_description, tbl_purpose, tbl_primary_key],
                outputs=[review_save_status],
            )

        # ══════════════════════════════════════════════════════════════
        # TAB 3: VALIDATE INPUTS
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("3. Validate Inputs"):
            gr.Markdown("### Validate Metadata Inputs (LLM-as-a-Judge)")
            gr.Markdown(
                "Runs 5 quality checks on your metadata **before** generating JSONs.\n"
                "This catches issues like weak descriptions, incorrect sensitivity labels, "
                "wrong FHIR types, inconsistent measurement methods, and cross-column problems.\n\n"
                "**Save your edits in Tab 2 first**, then run validation."
            )
            input_validate_btn = gr.Button("🔍 Validate Metadata Inputs", variant="primary", size="lg")
            input_val_check_status = gr.Markdown("*Click to validate*")

            gr.Markdown("---")
            gr.Markdown("### Full Report")
            input_val_report = gr.Textbox(label="Validation Report", interactive=False, lines=20)

            gr.Markdown("---")
            gr.Markdown("### Issue Resolution")
            gr.Markdown("Enter the issue # from the report above to apply a suggestion or skip it.")
            with gr.Row():
                input_issue_index = gr.Number(label="Issue #", value=0, minimum=0, precision=0)
                apply_input_fix_btn = gr.Button("✅ Apply Suggestion", variant="primary", size="sm")
                skip_input_fix_btn = gr.Button("⏭ Skip", variant="secondary", size="sm")
                apply_all_input_btn = gr.Button("✅ Apply All Fixes", variant="primary", size="sm")
            input_fix_status = gr.Markdown("")
            gr.Markdown("*After applying fixes, go back to Tab 2 to review the updated metadata.*")

            # ── Tab 3 Event handlers ──
            input_validate_btn.click(
                fn=run_input_validation_handler,
                outputs=[input_val_check_status, input_val_report],
            )
            apply_input_fix_btn.click(
                fn=apply_input_suggestion,
                inputs=[input_issue_index, review_table_selector],
                outputs=[input_fix_status, columns_dataframe],
            )
            skip_input_fix_btn.click(
                fn=skip_input_issue,
                inputs=[input_issue_index],
                outputs=[input_fix_status],
            )
            apply_all_input_btn.click(
                fn=apply_all_input_fixes,
                inputs=[review_table_selector],
                outputs=[input_fix_status, columns_dataframe],
            )

        # ══════════════════════════════════════════════════════════════
        # TAB 4: ADD TO CORTEX
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("4. Add to Cortex"):
            gr.Markdown("### Generate FHIR JSONs")
            gr.Markdown(
                "Builds StructureDefinitions, ValueSets, and DataProfiles **programmatically** "
                "from your reviewed metadata — **no LLM needed**."
            )
            generate_btn = gr.Button("📄 Generate FHIR JSONs", variant="primary", size="lg")
            gen_status = gr.Markdown("*Review & validate metadata in Tabs 2-3, then click Generate*")

            gr.Markdown("---")
            gr.Markdown("### JSON Preview")
            json_table_selector = gr.Dropdown(choices=[], label="Select Table")
            json_preview = gr.Code(label="StructureDefinition JSON", language="json", interactive=False, lines=25)

            gr.Markdown("---")
            gr.Markdown("### Register to Cortex")
            gr.Markdown("Once JSONs are generated, register this listing in the Cortex catalog to make it discoverable.")
            register_cortex_btn = gr.Button("🚀 Register to Cortex", variant="primary", size="lg")
            cortex_register_status = gr.Markdown("*Generate JSONs first, then register.*")

            generate_btn.click(
                fn=generate_jsons,
                inputs=[study_name, compliance_zone, retention_years, schema_stability, domain_contact, confidentiality],
                outputs=[gen_status, json_preview, json_table_selector],
            )
            json_table_selector.change(fn=show_generated_json, inputs=[json_table_selector], outputs=[json_preview])
            register_cortex_btn.click(
                fn=register_to_cortex_handler,
                inputs=[study_name, listing_type, listing_visibility],
                outputs=[cortex_register_status],
            )

        # ══════════════════════════════════════════════════════════════
        # TAB 5: DELIVER TO WORKBENCH
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("5. Deliver to Workbench"):
            gr.Markdown("### Deliver Generated Metadata")
            file_list_display = gr.Markdown("*No files generated yet*")
            refresh_files_btn = gr.Button("🔄 Refresh File List", size="sm")
            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 📦 Deliver to Workbench Data Collection")
                    gr.Markdown("Discover and browse buckets in your GCP project, then deliver the metadata to a Workbench data collection.")
                    with gr.Row():
                        discover_buckets_btn = gr.Button("🔍 Discover Buckets", variant="secondary", size="sm")
                        gcs_bucket_selector = gr.Dropdown(choices=[], label="Select Bucket", scale=2)
                    with gr.Row():
                        discover_folders_btn = gr.Button("📂 Browse Folders", variant="secondary", size="sm")
                        gcs_folder_selector = gr.Dropdown(choices=[], label="Select Folder", scale=2)
                    gcs_custom_prefix = gr.Textbox(label="Additional Sub-folder (optional)", placeholder="e.g., v2/ or 2026-02/")
                    save_gcs_btn = gr.Button("📦 Add to Data Collection", variant="primary", size="lg")
                    gcs_status = gr.Markdown("")

                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Download as ZIP")
                    gr.Markdown("Download all generated files as a single ZIP archive.")
                    download_btn = gr.Button("📥 Download ZIP", variant="primary", size="lg")
                    download_file = gr.File(label="Download", visible=True)

            refresh_files_btn.click(fn=get_export_file_list, outputs=[file_list_display])
            discover_buckets_btn.click(fn=discover_buckets_handler, outputs=[gcs_bucket_selector])
            discover_folders_btn.click(fn=discover_folders_handler, inputs=[gcs_bucket_selector], outputs=[gcs_folder_selector])
            save_gcs_btn.click(
                fn=save_to_gcs,
                inputs=[gcs_bucket_selector, gcs_folder_selector, gcs_custom_prefix],
                outputs=[gcs_status],
            )
            download_btn.click(fn=download_as_zip, outputs=[download_file])

        # ══════════════════════════════════════════════════════════════
        # TAB 6: VALIDATE JSONs (optional)
        # ══════════════════════════════════════════════════════════════
        with gr.Tab("6. Validate JSONs (optional)"):
            gr.Markdown("### Validate Generated JSONs (LLM-as-a-Judge)")
            gr.Markdown(
                "**Optional step** — runs 8 deep-dive checks on the generated FHIR JSONs:\n"
                "1. Column Coverage · 2. Data Type Accuracy · 3. VFIG Mapping Accuracy\n"
                "4. Security Label Accuracy · 5. Measurement Method Accuracy\n"
                "6. Cross-File Consistency · 7. L3 Metadata Completeness · 8. ValueSet Binding Completeness\n\n"
                "This is useful for final quality assurance before sharing the metadata."
            )
            json_validate_btn = gr.Button("▶ Run JSON Validation", variant="secondary", size="lg")
            json_val_check_status = gr.Markdown("*Generate JSONs in 'Add to Cortex' tab, then run validation*")

            gr.Markdown("---")
            gr.Markdown("### Full Report")
            json_val_report = gr.Textbox(label="Validation Report", interactive=False, lines=20)

            gr.Markdown("---")
            gr.Markdown("### Issue Resolution")
            with gr.Row():
                json_issue_index = gr.Number(label="Issue #", value=0, minimum=0, precision=0)
                apply_json_fix_btn = gr.Button("✅ Apply Fix", variant="primary", size="sm")
                skip_json_fix_btn = gr.Button("⏭ Skip", variant="secondary", size="sm")
                apply_all_json_btn = gr.Button("✅ Apply All Fixes", variant="primary", size="sm")
            json_fix_status = gr.Markdown("")

            gr.Markdown("### Updated JSON Preview")
            json_fix_preview = gr.Code(label="Updated JSON", language="json", interactive=False, lines=15)

            json_validate_btn.click(
                fn=run_json_validation,
                outputs=[json_val_check_status, json_val_report],
            )
            apply_json_fix_btn.click(
                fn=apply_json_fix,
                inputs=[json_issue_index],
                outputs=[json_fix_status, json_fix_preview],
            )
            skip_json_fix_btn.click(
                fn=skip_json_issue,
                inputs=[json_issue_index],
                outputs=[json_fix_status],
            )
            apply_all_json_btn.click(
                fn=apply_all_json_fixes,
                outputs=[json_fix_status, json_fix_preview],
            )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    initialize(args)
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
