"""
Step 4: Gradio Chat UI
The main application entry point — a Gradio-based chat interface for
natural language data exploration in Verily Workbench.

Run:
    python app.py                                    # Metadata-only mode (no BQ)
    python app.py --project=YOUR_GCP_PROJECT_ID      # Full mode with BigQuery
    python app.py --json-dir=/path/to/json/          # Custom metadata path

In a Workbench JupyterLab terminal:
    pip install -r requirements.txt
    python app.py --project=$(wb workspace describe --format=json | jq -r '.gcpProjectId')
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

# Default path to FHIR JSON metadata (relative to this file)
_DEFAULT_JSON_DIR = str(
    Path(__file__).parent.parent.parent
    / "product_mgmnt"
    / "Metadata"
    / "Metadata JSON for Demo"
    / "JSON Metadata"
)


def parse_args():
    parser = argparse.ArgumentParser(description="WB Data Explorer")
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GCP_PROJECT_ID"),
        help="GCP project ID for Vertex AI / LLM calls",
    )
    parser.add_argument(
        "--data-project",
        type=str,
        nargs="+",
        default=os.environ.get("DATA_PROJECT_ID", "").split(",") if os.environ.get("DATA_PROJECT_ID") else None,
        help="GCP project ID(s) where BigQuery data lives (from Workbench Data Collections). "
             "Pass multiple projects to scan data across them. "
             "If not set, defaults to --project.",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=os.environ.get("METADATA_JSON_DIR", _DEFAULT_JSON_DIR),
        help="Path to FHIR StructureDefinition JSON metadata directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_PORT", "7860")),
        help="Port to run Gradio on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model name",
    )
    return parser.parse_args()


# ── App State (initialized at startup) ────────────────────────────────────────

_schemas = None
_agent = None
_conversation_history = None
_config = None


def initialize(args):
    """Load metadata, resolve against BQ, and optionally create the agent."""
    global _schemas, _agent, _conversation_history, _config
    _config = args

    from metadata_loader import load_metadata, resolve_against_bigquery

    data_projects = args.data_project or ([args.project] if args.project else None)

    print(f"📂 Loading metadata from: {args.json_dir}")
    _schemas = load_metadata(args.json_dir)
    print(f"✓ Loaded {len(_schemas)} table schemas from FHIR JSONs")

    # Runtime resolution: match FHIR metadata to real BQ tables
    if data_projects:
        print(f"🔍 Resolving metadata against BQ project(s): {data_projects}")
        _schemas = resolve_against_bigquery(_schemas, data_projects)
    else:
        print("ℹ No data project specified — using raw FHIR table names")

    if args.project:
        print(f"🔧 Initializing agent (LLM project: {args.project}, Data project(s): {data_projects})")
        try:
            from agent import create_agent

            _agent, _, _ = create_agent(
                json_metadata_dir=args.json_dir,
                bq_project_id=args.project,
                data_project_id=data_projects,
                llm_model=args.llm_model,
            )
            print("✓ Agent ready with BigQuery + Gemini")
        except Exception as e:
            print(f"⚠ Agent creation failed: {e}")
            print("  Falling back to metadata-only mode")
            _agent = None
    else:
        print("ℹ No --project specified. Running in metadata-only mode (no BQ queries)")
        _agent = None

    _conversation_history = []


# ── Chat Handler ──────────────────────────────────────────────────────────────

def chat_handler(user_message: str, chat_history: list) -> tuple:
    """
    Process a user message and return the response.

    In agent mode: routes through LangGraph for tool-calling.
    In metadata-only mode: calls Gemini directly with schema context.
    """
    global _conversation_history

    if not user_message.strip():
        return chat_history, ""

    agent_sql = ""
    agent_tool_output = ""

    if _agent:
        # Full agent mode
        from agent import run_agent

        try:
            result_dict, _conversation_history = run_agent(
                _agent, user_message, _conversation_history
            )
            response = result_dict["response"]
            agent_sql = result_dict.get("sql", "")
            agent_tool_output = result_dict.get("tool_output", "")
        except Exception as e:
            response = f"❌ Error: {str(e)}"

    elif _schemas:
        # Metadata-only mode — direct Gemini call
        try:
            from prompt_engine import build_system_prompt, call_gemini

            system_prompt = build_system_prompt(_schemas, _config.project)
            response = call_gemini(
                system_prompt=system_prompt,
                user_message=user_message,
                project_id=_config.project,
            )
        except Exception as e:
            response = f"❌ LLM call failed: {str(e)}\n\nRunning in metadata-only mode. Pass --project=YOUR_PROJECT to enable Gemini."

    else:
        response = "⚠ No metadata loaded. Check the --json-dir path."

    # Ensure response is always a string
    if isinstance(response, list):
        response = "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in response
        )
    response = str(response) if response else ""

    # Update chat history (Gradio 6 uses dict format)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})

    # Extract SQL for display panel — prefer agent tool output, fallback to regex
    sql_display = agent_sql or _extract_sql_display(response)

    return chat_history, sql_display


def _extract_sql_display(response: str) -> str:
    """Extract SQL from response for the SQL display panel."""
    import re

    pattern = r"```sql\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_dataframe(tool_output: str):
    """Try to parse a table from tool output into a pandas DataFrame for display."""
    import io
    import re

    if not tool_output:
        return None

    # Look for pipe-delimited table rows (output from dataframe_to_display)
    lines = tool_output.strip().split("\n")
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        # Detect table rows: have pipes and are not separator lines
        if "|" in stripped and not re.match(r"^[\s|+-]+$", stripped):
            table_lines.append(stripped)
            in_table = True
        elif in_table and re.match(r"^[\s|+-]+$", stripped):
            # Separator line between header and data — skip
            continue
        elif in_table and not stripped:
            break  # End of table

    if len(table_lines) >= 2:
        try:
            # Parse pipe-delimited table
            header = [c.strip() for c in table_lines[0].split("|") if c.strip()]
            rows = []
            for row_line in table_lines[1:]:
                cells = [c.strip() for c in row_line.split("|") if c.strip()]
                if len(cells) == len(header):
                    rows.append(cells)

            if rows:
                return pd.DataFrame(rows, columns=header)
        except Exception:
            pass

    # Fallback: look for CSV-like output
    csv_match = re.search(r"rows returned.*?\n\n(.*)", tool_output, re.DOTALL)
    if csv_match:
        try:
            return pd.read_csv(io.StringIO(csv_match.group(1).strip()))
        except Exception:
            pass

    return None


def clear_handler():
    """Reset conversation."""
    global _conversation_history
    _conversation_history = []
    return [], ""


def download_csv(chat_history: list) -> Optional[str]:
    """Export last query result as CSV (placeholder — connected in Step 6)."""
    return None


# ── Schema Sidebar ────────────────────────────────────────────────────────────

def get_schema_sidebar() -> str:
    """Generate the schema sidebar content."""
    if not _schemas:
        return "No metadata loaded."

    from metadata_loader import format_table_summary

    return format_table_summary(_schemas)


def get_table_detail(table_name: str) -> str:
    """Get detailed info for a specific table."""
    if not _schemas:
        return "No metadata loaded."

    table = _schemas.get(table_name)
    if not table:
        # Try partial match
        for name, t in _schemas.items():
            if table_name.lower() in name.lower():
                table = t
                break

    if not table:
        return f"Table '{table_name}' not found."

    lines = [
        f"## {table.title}",
        f"**Table:** `{table.bq_table_name}`",
        f"**Description:** {table.description}",
        f"**Grain:** {table.purpose}",
        f"**Primary Key:** `{table.primary_key}`",
        "",
        "### Columns",
        "| Column | Type | Required | Sensitivity | Description |",
        "|--------|------|----------|-------------|-------------|",
    ]

    for col in table.columns:
        sens = col.sensitivity_label or "-"
        req = "✓" if col.is_required else ""
        desc = col.short_description[:60]
        lines.append(f"| `{col.name}` | {col.data_type} | {req} | {sens} | {desc} |")

    join_targets = [l.target_table_name or "?" for l in table.join_links]
    if join_targets:
        lines.append("")
        lines.append(f"### Joins To")
        for target in join_targets:
            lines.append(f"- `{target}`")

    return "\n".join(lines)


# ── Build Gradio UI ──────────────────────────────────────────────────────────

def build_ui():
    """Build the Gradio Blocks interface."""

    table_names = sorted(_schemas.keys()) if _schemas else []

    with gr.Blocks(
        title="WB Data Explorer",
    ) as app:
        gr.Markdown(
            """
            # 🔬 Workbench Data Explorer
            **Explore your data collections using natural language.**
            Ask questions about your data, create cohorts, and run SQL — all through conversation.
            """
        )

        mode_text = "🟢 **Full Mode** (BigQuery + Gemini)" if _config and _config.project else "🟡 **Metadata-Only Mode** (no BQ — pass `--project` to enable)"
        gr.Markdown(mode_text)

        with gr.Row():
            # Left: Chat panel (main area)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    buttons=["copy"],
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your data... (e.g., 'How many BHS participants are eligible?')",
                        label="Your question",
                        scale=5,
                        lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear Chat", size="sm")
                    # download_btn = gr.Button("📥 Download Results CSV", size="sm")

                # SQL display
                sql_display = gr.Code(
                    label="Generated SQL",
                    language="sql",
                    interactive=False,
                    visible=True,
                )

            # Right: Schema browser sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Available Tables")

                table_dropdown = gr.Dropdown(
                    choices=table_names,
                    label="Select a table",
                    interactive=True,
                )

                table_detail = gr.Markdown(
                    value=get_schema_sidebar(),
                    elem_classes=["schema-sidebar"],
                )

                # Example queries
                gr.Markdown("### 💡 Example Queries")
                examples = [
                    "What tables are available and what do they contain?",
                    "How many participants are eligible for the study?",
                    "Show top 10 participants with highest ASCVD risk scores at Year 1",
                    "What are the most common diagnoses?",
                    "Join cohort eligibility with ASCVD risk scores for Year 1 visits",
                    "How many participants have abnormal AUDIT-C scores by visit?",
                ]
                for ex in examples:
                    gr.Button(ex, size="sm").click(
                        fn=lambda x=ex: x,
                        outputs=msg_input,
                    )

        # ── Event Handlers ────────────────────────────────────────────────

        send_btn.click(
            fn=chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, sql_display],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        msg_input.submit(
            fn=chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, sql_display],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        clear_btn.click(
            fn=clear_handler,
            outputs=[chatbot, sql_display],
        )

        table_dropdown.change(
            fn=get_table_detail,
            inputs=table_dropdown,
            outputs=table_detail,
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

