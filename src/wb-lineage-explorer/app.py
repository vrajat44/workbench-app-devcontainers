"""
WB Lineage Explorer — Gradio App
Visual FHIR data lineage using Provenance tables from Cortex dev-stable BigQuery.

Supports three datasets:
  - operational_fhir_mirror
  - landing_fhir_mirror
  - cdr_fhir_mirror

Two views:
  1. High-Level Lineage: aggregate flow of resource types through activities
     — with optional focus-by-resource-type + profile filter
  2. Instance-Level Lineage: select a final resource, trace backward through
     every FHIR resource, enrichment, and process that led to it

Provenance resources themselves are hidden from graphs — only
clinical / administrative resources, activities, and transformations are shown.

Run locally:
    python app.py --port=8080
"""

from __future__ import annotations

import argparse
import os
import traceback

import gradio as gr
import pandas as pd


# ── CLI Arguments ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="WB Lineage Explorer")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_PORT", "8080")),
        help="Port to run Gradio on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share link",
    )
    return parser.parse_args()


# ── Datasets ─────────────────────────────────────────────────────────────────

DATASET_CHOICES = [
    "operational_fhir_mirror",
    "landing_fhir_mirror",
    "cdr_fhir_mirror",
]

# ── State ─────────────────────────────────────────────────────────────────────
# (no global state needed — handlers are self-contained)


# ── High-Level Handlers ──────────────────────────────────────────────────────


def load_high_level(dataset: str, progress=gr.Progress()):
    """Load and render the full (unfiltered) high-level lineage graph."""
    try:
        from bq_lineage import (
            get_high_level_lineage,
            get_activity_summary,
            get_target_type_summary,
            get_filterable_target_types,
        )
        from lineage_graph import build_high_level_graph

        progress(0.1, desc="Querying activity summary...")
        activity_df = get_activity_summary(dataset)

        progress(0.3, desc="Querying target type summary...")
        target_df = get_target_type_summary(dataset)

        progress(0.4, desc="Loading filterable resource types...")
        filterable_types = get_filterable_target_types(dataset)

        progress(0.5, desc="Querying lineage flows (this may take a moment)...")
        lineage_df = get_high_level_lineage(dataset)

        progress(0.8, desc="Building graph...")
        graph_html = build_high_level_graph(lineage_df)

        # Build summary markdown
        summary_lines = [
            f"### 📊 Dataset: `{dataset}` — Full Universe",
            "",
            "#### Top Activities",
            "| Activity | Count |",
            "|----------|-------|",
        ]
        for _, row in activity_df.head(15).iterrows():
            cnt = int(row["cnt"])
            summary_lines.append(f"| {row['activity']} | {cnt:,} |")

        summary_lines.extend([
            "",
            "#### Top Target Resource Types",
            "| Resource Type | Count |",
            "|---------------|-------|",
        ])
        for _, row in target_df.head(15).iterrows():
            cnt = int(row["cnt"])
            summary_lines.append(f"| {row['target_type']} | {cnt:,} |")

        summary_md = "\n".join(summary_lines)

        # Update the focus-resource-type dropdown
        focus_type_update = gr.update(
            choices=["(none)"] + filterable_types,
            value="(none)",
        )
        # Reset the profile dropdown
        profile_update = gr.update(choices=["(all)"], value="(all)", interactive=False)

        progress(1.0, desc="Done!")
        return (
            f"<iframe srcdoc='{_escape_for_iframe(graph_html)}' "
            f"width='100%' height='720px' frameborder='0'></iframe>",
            summary_md,
            lineage_df,
            focus_type_update,
            profile_update,
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return (
            "<div style='padding:20px;color:red'>Error loading graph</div>",
            error_msg,
            pd.DataFrame(),
            gr.update(),
            gr.update(),
        )


def on_focus_type_changed(dataset: str, focus_type: str, progress=gr.Progress()):
    """When the user picks a focus resource type, load its profiles."""
    if not focus_type or focus_type == "(none)":
        return gr.update(choices=["(all)"], value="(all)", interactive=False)

    try:
        from bq_lineage import get_resource_profiles

        progress(0.3, desc=f"Querying {focus_type} profiles...")
        df = get_resource_profiles(dataset, focus_type)

        if df.empty:
            return gr.update(choices=["(all — no profiles found)"], value="(all — no profiles found)", interactive=False)

        # Build dropdown labels with counts
        choices = ["(all)"]
        for _, row in df.iterrows():
            prof = row["profile"]
            cnt = int(row["cnt"])
            choices.append(prof)

        progress(1.0, desc="Done!")
        return gr.update(choices=choices, value="(all)", interactive=True)
    except Exception as e:
        return gr.update(choices=[f"(error: {e})"], value=None, interactive=False)


def load_filtered_lineage(dataset: str, focus_type: str, profile: str, progress=gr.Progress()):
    """Load lineage filtered by target resource type + profile."""
    if not focus_type or focus_type == "(none)":
        # Fall back to full universe
        return load_high_level(dataset, progress=progress)

    if not profile or profile.startswith("(all"):
        # Focus on resource type but no profile filter — use regular query
        # filtered to just flows ending at this target type
        try:
            from bq_lineage import get_high_level_lineage
            from lineage_graph import build_high_level_graph

            progress(0.3, desc=f"Querying lineage for {focus_type}...")
            lineage_df = get_high_level_lineage(dataset)

            # Filter to rows where target_type matches
            filtered = lineage_df[lineage_df["target_type"] == focus_type]

            if filtered.empty:
                return (
                    "<div style='padding:40px;text-align:center;color:#888'>"
                    f"<h3>No lineage flows found targeting {focus_type}</h3></div>",
                    f"No data for target type `{focus_type}`.",
                    pd.DataFrame(),
                    gr.update(),
                    gr.update(),
                )

            progress(0.7, desc="Building graph...")
            graph_html = build_high_level_graph(filtered)

            summary_md = (
                f"### 📊 Focused on target: `{focus_type}`\n\n"
                f"Showing **{len(filtered)}** lineage flow(s) ending at `{focus_type}`.\n\n"
                f"Select a profile from the dropdown to narrow further, "
                f"or click **Back to Full Universe** to reset."
            )

            progress(1.0, desc="Done!")
            return (
                f"<iframe srcdoc='{_escape_for_iframe(graph_html)}' "
                f"width='100%' height='720px' frameborder='0'></iframe>",
                summary_md,
                filtered,
                gr.update(),
                gr.update(),
            )
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            return (
                "<div style='padding:20px;color:red'>Error loading graph</div>",
                error_msg,
                pd.DataFrame(),
                gr.update(),
                gr.update(),
            )

    # Both focus type AND profile are set — run the profile-joined query
    try:
        from bq_lineage import get_high_level_lineage_by_profile
        from lineage_graph import build_high_level_graph

        progress(0.3, desc=f"Querying lineage for {focus_type} with profile filter...")
        lineage_df = get_high_level_lineage_by_profile(dataset, focus_type, profile)

        if lineage_df.empty:
            return (
                "<div style='padding:40px;text-align:center;color:#888'>"
                f"<h3>No lineage flows found for {focus_type} with this profile</h3></div>",
                f"No data for `{focus_type}` with profile `{profile}`.",
                pd.DataFrame(),
                gr.update(),
                gr.update(),
            )

        progress(0.7, desc="Building graph...")
        graph_html = build_high_level_graph(lineage_df)

        # Shorten profile for display
        short_profile = profile.rsplit("/", 1)[-1] if "/" in profile else profile
        summary_md = (
            f"### 📊 Filtered: `{focus_type}` with profile `{short_profile}`\n\n"
            f"Showing **{len(lineage_df)}** lineage flow(s).\n\n"
            f"Click **Back to Full Universe** to reset."
        )

        progress(1.0, desc="Done!")
        return (
            f"<iframe srcdoc='{_escape_for_iframe(graph_html)}' "
            f"width='100%' height='720px' frameborder='0'></iframe>",
            summary_md,
            lineage_df,
            gr.update(),
            gr.update(),
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return (
            "<div style='padding:20px;color:red'>Error loading filtered graph</div>",
            error_msg,
            pd.DataFrame(),
            gr.update(),
            gr.update(),
        )


# ── Instance-Level Handler ────────────────────────────────────────────────────


def trace_resource_lineage(
    dataset: str,
    resource_id: str,
    max_depth: int,
    progress=gr.Progress(),
):
    """
    Given just a FHIR Resource ID, auto-detect its type and trace the full
    lineage chain from the very first source to that resource.
    Returns: (graph_html, summary_md, raw_df)
    """
    resource_id = (resource_id or "").strip()
    if not resource_id:
        return (
            "<div style='padding:40px;text-align:center;color:#888'>"
            "<h3>Paste a FHIR Resource ID and click 'Trace Full Lineage'</h3></div>",
            "",
            pd.DataFrame(),
        )

    try:
        from bq_lineage import resolve_resource_type, get_multi_hop_lineage
        from lineage_graph import build_instance_graph

        progress(0.1, desc="Resolving resource type...")
        resource_type = resolve_resource_type(dataset, resource_id)

        if not resource_type:
            return (
                "<div style='padding:40px;text-align:center;color:#c62828'>"
                f"<h3>Resource ID not found in {dataset}</h3>"
                f"<p>No Provenance target matched <code>{resource_id}</code>. "
                "Double-check the ID and dataset.</p></div>",
                f"❌ Could not find `{resource_id}` in `{dataset}` Provenance targets.",
                pd.DataFrame(),
            )

        progress(0.3, desc=f"Found type: {resource_type}. Tracing lineage (depth {max_depth})...")

        df = get_multi_hop_lineage(
            dataset, resource_type, resource_id, max_depth=max_depth
        )

        progress(0.8, desc="Building graph...")

        if df.empty:
            return (
                "<div style='padding:40px;text-align:center;color:#888'>"
                f"<h3>No lineage data found for {resource_type}/{resource_id}</h3></div>",
                f"Resource type resolved to **{resource_type}**, but no provenance chain was found.",
                pd.DataFrame(),
            )

        graph_html = build_instance_graph(df, resource_type, resource_id)

        # Build summary
        activities = df["activity"].value_counts()
        entity_types = (
            df[df["entity_type"] != "(none)"]["entity_type"].value_counts()
            if "entity_type" in df.columns
            else pd.Series()
        )
        depths = df["depth"].nunique() if "depth" in df.columns else 1
        max_traced = int(df["depth"].max()) if "depth" in df.columns else 1

        summary_lines = [
            f"### 🔍 Lineage for `{resource_type}/{resource_id}`",
            f"*Resource type auto-detected as **{resource_type}***",
            "",
            f"- **{len(df)}** provenance records in chain",
            f"- **{max_traced}** depth level(s) traced back",
            f"- **{len(activities)}** distinct activities",
            f"- **{len(entity_types)}** distinct source entity types",
            "",
            "#### Activities in Chain",
            "| Activity | Count |",
            "|----------|-------|",
        ]
        for act, cnt in activities.items():
            summary_lines.append(f"| {act} | {cnt} |")

        if not entity_types.empty:
            summary_lines.extend([
                "",
                "#### Source Entity Types",
                "| Entity Type | Count |",
                "|-------------|-------|",
            ])
            for et, cnt in entity_types.items():
                summary_lines.append(f"| {et} | {cnt} |")

        summary = "\n".join(summary_lines)

        progress(1.0, desc="Done!")
        return (
            f"<iframe srcdoc='{_escape_for_iframe(graph_html)}' "
            f"width='100%' height='720px' frameborder='0'></iframe>",
            summary,
            df,
        )
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return (
            "<div style='padding:20px;color:red'>Error tracing lineage</div>",
            error_msg,
            pd.DataFrame(),
        )


def _escape_for_iframe(html_str: str) -> str:
    """Escape HTML for embedding inside an iframe srcdoc attribute."""
    return (
        html_str
        .replace("&", "&amp;")
        .replace("'", "&#39;")
        .replace('"', "&quot;")
    )


# ── Build UI ──────────────────────────────────────────────────────────────────


def build_ui():
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="WB Lineage Explorer",
    ) as app:

        gr.Markdown(
            """
            # 🔗 Workbench Lineage Explorer
            **Visualize FHIR data lineage** from Cortex Provenance tables in BigQuery dev-stable.

            Trace how resources flow through ingestion, enrichment, and workflow processes.
            Graphs flow **left → right**: Source → Activity/Transformation → Target.
            *(Provenance resources are hidden — only clinical/admin resources and activities are shown.)*

            > **Datasets:** `prj-d-1v-ucd` · `operational_fhir_mirror` · `landing_fhir_mirror` · `cdr_fhir_mirror`
            """
        )

        # ── Dataset Selector (shared) ────────────────────────────────────
        with gr.Row():
            dataset_selector = gr.Dropdown(
                choices=DATASET_CHOICES,
                value=DATASET_CHOICES[0],
                label="📦 Select Dataset",
                interactive=True,
                scale=2,
            )
            gr.Markdown(
                "*Choose which FHIR mirror to explore. "
                "Landing = raw ingested, Operational = enriched, CDR = curated.*",
            )

        # ── Tabs ─────────────────────────────────────────────────────────
        with gr.Tabs():

            # ════════════════════════════════════════════════════════════
            # TAB 1: High-Level Lineage
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🗺️ High-Level Lineage", id="high-level"):
                gr.Markdown(
                    """
                    **Aggregate view**: Shows how resource types flow through activities.
                    - **Circles (left)** = Source FHIR resource types
                    - **Diamonds (middle)** = Activities / transformations / enrichments
                    - **Circles (right)** = Target FHIR resource types
                    - **Edges** = Flow direction with volume

                    *Click "Load High-Level Lineage" for the full universe, then optionally
                    focus on a target resource type and filter by its FHIR profile.*
                    """
                )

                with gr.Row():
                    hl_load_btn = gr.Button(
                        "🔄 Load High-Level Lineage",
                        variant="primary",
                        size="lg",
                        scale=1,
                    )
                    hl_focus_type = gr.Dropdown(
                        choices=["(none)"],
                        value="(none)",
                        label="🎯 Focus Target Resource Type",
                        interactive=True,
                        scale=1,
                        info="Pick a target type to zoom in, then filter by profile",
                    )
                    hl_profile = gr.Dropdown(
                        choices=["(all)"],
                        value="(all)",
                        label="🏷️ Filter by Profile",
                        interactive=False,
                        scale=1,
                        info="Auto-populated when you pick a resource type",
                    )
                    hl_reset_btn = gr.Button(
                        "🔙 Back to Full Universe",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        hl_graph = gr.HTML(
                            value="<div style='padding:60px;text-align:center;color:#aaa;background:#f5f5f5;border-radius:12px'>"
                            "<h3>Click 'Load High-Level Lineage' to begin</h3>"
                            "<p>This queries aggregate flow data from the Provenance table</p></div>",
                            label="Lineage Graph",
                        )
                    with gr.Column(scale=1):
                        hl_summary = gr.Markdown("*Summary will appear here*")

                with gr.Accordion("📋 Raw Lineage Data", open=False):
                    hl_table = gr.Dataframe(
                        label="Lineage Flows (entity_type → activity → target_type)",
                        interactive=False,
                    )

                # ── High-level event wiring ──

                # Load full universe
                hl_load_btn.click(
                    fn=load_high_level,
                    inputs=[dataset_selector],
                    outputs=[hl_graph, hl_summary, hl_table, hl_focus_type, hl_profile],
                )

                # Focus type changed → load profiles for that type
                hl_focus_type.change(
                    fn=on_focus_type_changed,
                    inputs=[dataset_selector, hl_focus_type],
                    outputs=[hl_profile],
                )

                # Profile changed → re-render filtered graph
                hl_profile.change(
                    fn=load_filtered_lineage,
                    inputs=[dataset_selector, hl_focus_type, hl_profile],
                    outputs=[hl_graph, hl_summary, hl_table, hl_focus_type, hl_profile],
                )

                # Reset → reload full universe
                hl_reset_btn.click(
                    fn=load_high_level,
                    inputs=[dataset_selector],
                    outputs=[hl_graph, hl_summary, hl_table, hl_focus_type, hl_profile],
                )

            # ════════════════════════════════════════════════════════════
            # TAB 2: Instance-Level Lineage
            # ════════════════════════════════════════════════════════════
            with gr.TabItem("🔬 Instance-Level Lineage", id="instance"):
                gr.Markdown(
                    """
                    **Trace a specific resource**: Paste a FHIR Resource ID and see
                    the **full lineage graph** — every source resource, enrichment,
                    and process that led to it, from the very start.

                    The resource type is **auto-detected**. Graph flows **left → right**.
                    """
                )

                with gr.Row():
                    inst_id_input = gr.Textbox(
                        label="🔑 FHIR Resource ID",
                        placeholder="e.g. a82af88d-55b5-4354-8247-9b797cd131ef",
                        scale=3,
                    )
                    inst_depth = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Max Trace Depth",
                        info="How many hops back to trace (higher = more complete)",
                        scale=1,
                    )
                    inst_trace_btn = gr.Button(
                        "🔗 Trace Full Lineage",
                        variant="primary",
                        size="lg",
                        scale=1,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        inst_graph = gr.HTML(
                            value="<div style='padding:60px;text-align:center;color:#aaa;background:#f5f5f5;border-radius:12px'>"
                            "<h3>Paste a Resource ID and click 'Trace Full Lineage'</h3>"
                            "<p>The resource type will be auto-detected from the Provenance table</p></div>",
                            label="Instance Lineage Graph",
                        )
                    with gr.Column(scale=1):
                        inst_summary = gr.Markdown("*Lineage summary will appear here*")

                with gr.Accordion("📋 Raw Provenance Chain", open=False):
                    inst_raw_table = gr.Dataframe(
                        label="Provenance Records",
                        interactive=False,
                    )

                # ── Instance event wiring ─────────────────────────────────

                inst_trace_btn.click(
                    fn=trace_resource_lineage,
                    inputs=[dataset_selector, inst_id_input, inst_depth],
                    outputs=[inst_graph, inst_summary, inst_raw_table],
                )

        # ── Footer ───────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **WB Lineage Explorer** · Querying `prj-d-1v-ucd` BigQuery dev-stable
            · Provenance tables from Cortex FHIR mirrors
            · Built for Verily Workbench
            """,
        )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
