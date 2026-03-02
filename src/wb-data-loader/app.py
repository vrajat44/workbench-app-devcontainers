"""
WB Data Loader — Gradio UI  (PyAirbyte edition)

Ingest data from 600+ sources into BigQuery using Airbyte's pre-built connectors.

Run:
    python app.py                                     # Preview-only mode
    python app.py --project=YOUR_GCP_PROJECT_ID       # Full mode with BigQuery
    python app.py --port=8080                          # Custom port

In a Workbench JupyterLab terminal:
    pip install -r requirements.txt
    python app.py --project=$(wb workspace describe --format=json | jq -r '.gcpProjectId')
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from typing import Any, Optional

import gradio as gr
import pandas as pd

from airbyte_connector import (
    ConnectorInfo,
    PreviewResult,
    StreamInfo,
    SyncResult,
    check_source,
    create_source,
    discover_streams,
    get_config_template,
    get_curated_display_names,
    preview_stream,
    resolve_connector_name,
    sync_to_bigquery,
)


# ── Configuration ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="WB Data Loader (PyAirbyte)")
    parser.add_argument(
        "--project", type=str,
        default=os.environ.get("GCP_PROJECT_ID"),
        help="GCP project ID for BigQuery destination and billing",
    )
    parser.add_argument(
        "--bq-dataset", type=str,
        default=os.environ.get("BQ_DATASET", ""),
        help="Default BigQuery dataset for loading data",
    )
    parser.add_argument(
        "--credentials-path", type=str,
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        help="Path to GCP service account JSON (optional, uses ADC by default)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("GRADIO_PORT", "7860")),
        help="Port to run Gradio on",
    )
    parser.add_argument(
        "--share", action="store_true", default=False,
        help="Create a public Gradio share link",
    )
    return parser.parse_args()


# ── App State ─────────────────────────────────────────────────────────────────

_config = None
_current_connector: Optional[str] = None
_current_source_config: dict[str, Any] = {}
_discovered_streams: list[StreamInfo] = []
_sync_history: list[dict] = []


def initialize(args):
    """Initialize app state."""
    global _config
    _config = args
    if args.project:
        print(f"🔧 GCP Project: {args.project}")
        if args.bq_dataset:
            print(f"📊 Default BQ Dataset: {args.bq_dataset}")
    else:
        print("ℹ️  No --project specified. BigQuery sync will not be available.")
    print("✅ WB Data Loader (PyAirbyte edition) ready")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SOURCE SETUP — handlers
# ══════════════════════════════════════════════════════════════════════════════

def on_connector_select(connector_display: str) -> tuple:
    """
    When a connector is selected, show its config template.
    Returns: (config_form_html, raw_json_visible, status)
    """
    connector_name = resolve_connector_name(connector_display)
    template = get_config_template(connector_name)

    if template:
        # Build a human-readable description of required fields
        lines = [f"### Configure: `{connector_name}`\n"]
        for f in template:
            req = " *(required)*" if f.get("required") else ""
            lines.append(f"- **{f['label']}**{req}")
            if f.get("placeholder"):
                lines.append(f"  - Example: `{f['placeholder']}`")
        form_md = "\n".join(lines)

        # Build a template JSON
        template_json = {}
        for f in template:
            template_json[f["key"]] = f.get("placeholder", "")
        json_str = json.dumps(template_json, indent=2)

        return (
            form_md,                                    # config_help
            json_str,                                   # config_json (pre-filled template)
            f"Selected **{connector_display}**. Fill in the config below and click **Connect**.",
        )
    else:
        # No template — show raw JSON editor
        return (
            f"### Configure: `{connector_name}`\n\nNo template available. Enter raw config JSON.",
            "{}",
            f"Selected **{connector_display}**. Enter config JSON and click **Connect**.",
        )


def handle_connect(
    connector_display: str,
    config_json: str,
) -> tuple:
    """
    Connect to the source: validate config, check connection, discover streams.
    Returns: (status_md, stream_choices, stream_dropdown)
    """
    global _current_connector, _current_source_config, _discovered_streams

    connector_name = resolve_connector_name(connector_display)

    # Parse config JSON
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return (
            f"❌ Invalid JSON config: {e}",
            gr.Dropdown(choices=[]),
            gr.CheckboxGroup(choices=[]),
        )

    _current_connector = connector_name
    _current_source_config = config

    # Check connection
    ok, msg = check_source(connector_name, config)
    if not ok:
        return (
            msg,
            gr.Dropdown(choices=[]),
            gr.CheckboxGroup(choices=[]),
        )

    # Discover streams
    streams = discover_streams(connector_name, config)
    _discovered_streams = streams

    if not streams:
        return (
            "✅ Connected, but no streams discovered. Check your config.",
            gr.Dropdown(choices=[]),
            gr.CheckboxGroup(choices=[]),
        )

    stream_names = [s.name for s in streams]
    status_lines = [
        f"### ✅ Connected to `{connector_name}`\n",
        f"**{len(stream_names)} stream(s) available:**",
    ]
    for name in stream_names[:20]:
        status_lines.append(f"- `{name}`")
    if len(stream_names) > 20:
        status_lines.append(f"- ... +{len(stream_names) - 20} more")

    return (
        "\n".join(status_lines),
        gr.Dropdown(choices=stream_names, value=stream_names[0] if stream_names else None),
        gr.CheckboxGroup(choices=stream_names, value=stream_names),
    )


def handle_preview(
    connector_display: str,
    config_json: str,
    preview_stream_name: str,
) -> tuple:
    """
    Preview data from a single stream.
    Returns: (preview_df, schema_df, status)
    """
    connector_name = resolve_connector_name(connector_display)

    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return pd.DataFrame(), pd.DataFrame(), f"❌ Invalid config JSON: {e}"

    if not preview_stream_name:
        return pd.DataFrame(), pd.DataFrame(), "❌ Select a stream to preview."

    result = preview_stream(connector_name, config, preview_stream_name, max_records=100)

    if not result.success:
        return pd.DataFrame(), pd.DataFrame(), f"❌ Preview failed: {result.error}"

    if result.dataframe is None or result.dataframe.empty:
        return pd.DataFrame(), pd.DataFrame(), "⚠️ Stream returned 0 records."

    # Build schema table
    schema_df = pd.DataFrame(result.schema_info) if result.schema_info else pd.DataFrame()

    status = (
        f"✅ Preview: **{result.row_count}** rows × **{result.column_count}** columns "
        f"from stream `{preview_stream_name}`"
    )
    return result.dataframe.head(20), schema_df, status


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SYNC TO BIGQUERY — handlers
# ══════════════════════════════════════════════════════════════════════════════

def handle_sync(
    connector_display: str,
    config_json: str,
    selected_streams: list[str],
    bq_project: str,
    bq_dataset: str,
    credentials_path: str,
    progress=gr.Progress(track_tqdm=False),
) -> str:
    """Execute the sync from source → BigQuery."""
    global _sync_history

    connector_name = resolve_connector_name(connector_display)

    # Validate inputs
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return f"❌ Invalid config JSON: {e}"

    if not selected_streams:
        return "❌ Select at least one stream to sync."

    project = (bq_project or "").strip() or (_config.project if _config else "")
    if not project:
        return "❌ No GCP project configured. Pass `--project` or enter a project ID."

    dataset = (bq_dataset or "").strip() or (_config.bq_dataset if _config and _config.bq_dataset else "")
    if not dataset:
        return "❌ Enter a BigQuery dataset name."

    creds = (credentials_path or "").strip() or (_config.credentials_path if _config and _config.credentials_path else "")
    creds = creds if creds else None

    progress(0, desc=f"Syncing {len(selected_streams)} stream(s) to BigQuery...")

    result = sync_to_bigquery(
        connector_name=connector_name,
        config=config,
        streams=selected_streams,
        bq_project=project,
        bq_dataset=dataset,
        credentials_path=creds,
    )

    # Record history
    _sync_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": connector_name,
        "streams": ", ".join(result.streams_synced),
        "destination": result.destination,
        "records": result.total_records,
        "seconds": result.elapsed_seconds,
        "status": "✅ Success" if result.success else "❌ Failed",
        "error": result.error or "",
    })

    if result.success:
        lines = [
            "### ✅ Sync Complete\n",
            f"- **Source:** `{connector_name}`",
            f"- **Destination:** `{result.destination}`",
            f"- **Streams synced:** {len(result.streams_synced)}",
            f"- **Total records:** {result.total_records:,}",
            f"- **Time:** {result.elapsed_seconds}s",
        ]
        if result.warnings:
            lines.append("\n**Warnings:**")
            for w in result.warnings:
                lines.append(f"- ⚠️ {w}")
        return "\n".join(lines)
    else:
        return f"### ❌ Sync Failed\n\n**Error:** {result.error}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: HISTORY — handlers
# ══════════════════════════════════════════════════════════════════════════════

def get_history_display() -> pd.DataFrame:
    """Return sync history as a DataFrame."""
    if not _sync_history:
        return pd.DataFrame(columns=[
            "Timestamp", "Source", "Streams", "Destination",
            "Records", "Seconds", "Status",
        ])
    return pd.DataFrame(_sync_history).rename(columns={
        "timestamp": "Timestamp",
        "source": "Source",
        "streams": "Streams",
        "destination": "Destination",
        "records": "Records",
        "seconds": "Seconds",
        "status": "Status",
    })


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

def build_ui():
    curated_names = get_curated_display_names()

    with gr.Blocks(title="WB Data Loader", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 📥 Workbench Data Loader\n"
            "**Ingest data from 600+ sources into BigQuery using "
            "[PyAirbyte](https://github.com/airbytehq/pyairbyte) connectors.**"
        )

        mode_text = (
            "🟢 **Full Mode** — BigQuery enabled"
            if _config and _config.project
            else "🟡 **Preview Mode** — no GCP project. Pass `--project` to enable BigQuery sync."
        )
        gr.Markdown(mode_text)

        # ══════════════════════════════════════════════════════════
        # TAB 1: SOURCE SETUP
        # ══════════════════════════════════════════════════════════
        with gr.Tab("1️⃣  Connect Source"):
            gr.Markdown("### Select a source connector and configure it")

            with gr.Row():
                connector_dropdown = gr.Dropdown(
                    choices=curated_names,
                    value=curated_names[0] if curated_names else None,
                    label="Source Connector",
                    info="Pick a pre-built connector or type any Airbyte connector name",
                    allow_custom_value=True,
                    scale=3,
                )
                connect_btn = gr.Button(
                    "🔌 Connect & Discover", variant="primary", scale=1,
                )

            config_help = gr.Markdown("*Select a connector above*")
            config_json = gr.Code(
                label="Source Config (JSON)",
                language="json",
                value="{}",
                lines=10,
            )

            connect_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Stream Discovery & Preview")

            with gr.Row():
                preview_stream_dropdown = gr.Dropdown(
                    choices=[], label="Preview Stream", scale=2,
                )
                preview_btn = gr.Button("👁️ Preview Data", variant="secondary", scale=1)

            stream_selector = gr.CheckboxGroup(
                choices=[], label="Streams to Sync (select for Tab 2)",
            )

            preview_status = gr.Markdown("")
            data_preview = gr.Dataframe(
                label="Sample Data (first 20 rows)",
                interactive=False,
                wrap=True,
            )
            schema_table = gr.Dataframe(
                label="Column Schema",
                interactive=False,
                wrap=True,
            )

            # ─ Event handlers ─
            connector_dropdown.change(
                fn=on_connector_select,
                inputs=[connector_dropdown],
                outputs=[config_help, config_json, connect_status],
            )

            connect_btn.click(
                fn=handle_connect,
                inputs=[connector_dropdown, config_json],
                outputs=[connect_status, preview_stream_dropdown, stream_selector],
            )

            preview_btn.click(
                fn=handle_preview,
                inputs=[connector_dropdown, config_json, preview_stream_dropdown],
                outputs=[data_preview, schema_table, preview_status],
            )

        # ══════════════════════════════════════════════════════════
        # TAB 2: SYNC TO BIGQUERY
        # ══════════════════════════════════════════════════════════
        with gr.Tab("2️⃣  Sync to BigQuery"):
            gr.Markdown("### Configure destination and run the sync")

            with gr.Row():
                with gr.Column(scale=2):
                    bq_project = gr.Textbox(
                        label="BigQuery Project ID",
                        value=(_config.project if _config and _config.project else ""),
                        placeholder="your-gcp-project-id",
                    )
                    bq_dataset = gr.Textbox(
                        label="BigQuery Dataset",
                        value=(_config.bq_dataset if _config and _config.bq_dataset else ""),
                        placeholder="my_dataset (will be created if needed)",
                    )
                    credentials_path = gr.Textbox(
                        label="Service Account JSON Path (optional)",
                        value=(_config.credentials_path if _config and _config.credentials_path else ""),
                        placeholder="(uses Application Default Credentials if empty)",
                    )

                with gr.Column(scale=1):
                    gr.Markdown(
                        "#### How It Works\n\n"
                        "PyAirbyte reads data from your selected source connector and writes it "
                        "directly to BigQuery tables.\n\n"
                        "- Each **stream** becomes a **table** in BigQuery\n"
                        "- Table names are auto-normalized\n"
                        "- Schema is auto-detected\n"
                        "- Uses **Application Default Credentials** by default\n\n"
                        "#### Supported Sources\n"
                        "600+ connectors including:\n"
                        "- ☁️ S3, GCS, Azure Blob\n"
                        "- 🐘 PostgreSQL, MySQL, SQL Server\n"
                        "- ❄️ Snowflake, BigQuery\n"
                        "- 📊 Google Sheets, Excel\n"
                        "- 🔗 Salesforce, HubSpot, GitHub, ..."
                    )

            gr.Markdown("---")
            sync_btn = gr.Button("🚀 Sync to BigQuery", variant="primary", size="lg")
            sync_status = gr.Markdown(
                "*Configure source (Tab 1), select streams, then click Sync.*"
            )

            sync_btn.click(
                fn=handle_sync,
                inputs=[
                    connector_dropdown, config_json, stream_selector,
                    bq_project, bq_dataset, credentials_path,
                ],
                outputs=[sync_status],
            )

        # ══════════════════════════════════════════════════════════
        # TAB 3: SYNC HISTORY
        # ══════════════════════════════════════════════════════════
        with gr.Tab("3️⃣  Sync History"):
            gr.Markdown("### Past Syncs (this session)")
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            history_table = gr.Dataframe(interactive=False, wrap=True)
            refresh_btn.click(fn=get_history_display, outputs=[history_table])

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
