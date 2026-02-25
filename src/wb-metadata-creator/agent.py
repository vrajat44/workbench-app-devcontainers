"""
LangGraph Agent for WB Metadata Creator.

Provides chat-based refinement of generated FHIR StructureDefinition JSONs.
The agent has tools to view and modify the generated metadata.

Tools:
  1. view_current_json — View the current state of a generated JSON
  2. modify_json — Apply a natural-language modification to a JSON
  3. get_table_schema — Get BQ schema for a table
  4. list_generated_files — List all generated metadata files

Usage:
    from agent import create_agent, run_agent

    agent, state = create_agent(
        generated_jsons={"table_name": {...}},
        bq_tables=[...],
        project_id="your-project",
    )
    result, history = run_agent(agent, "Change phq9_1 to PHI sensitivity")
"""

from __future__ import annotations

import json
from typing import Annotated, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from fhir_generator import BQTableInfo, format_bq_schema_for_prompt
from prompt_engine import (
    REFINEMENT_SYSTEM_PROMPT,
    call_gemini_fast,
    extract_json_from_response,
)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State for the LangGraph refinement agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Shared Mutable State ─────────────────────────────────────────────────────
# The generated JSONs are stored here and modified in-place by tools.

_generated_jsons: dict[str, dict] = {}
_generated_value_sets: dict[str, list[dict]] = {}
_bq_tables: dict[str, BQTableInfo] = {}


def set_agent_state(
    generated_jsons: dict[str, dict],
    value_sets: dict[str, list[dict]],
    bq_tables: list[BQTableInfo],
):
    """Update the shared state that agent tools operate on."""
    global _generated_jsons, _generated_value_sets, _bq_tables
    _generated_jsons = generated_jsons
    _generated_value_sets = value_sets
    _bq_tables = {t.fq_name: t for t in bq_tables}


# ── Tool Definitions ──────────────────────────────────────────────────────────

def _make_tools(project_id: Optional[str] = None):
    """Create tool functions for the refinement agent."""

    @tool
    def view_current_json(table_name: str) -> str:
        """View the current state of a generated FHIR StructureDefinition JSON.
        Provide the table name (e.g., 'project.dataset.TABLE').
        Returns the full JSON content.
        """
        # Try exact match
        json_obj = _generated_jsons.get(table_name)

        # Try partial match
        if not json_obj:
            for name, obj in _generated_jsons.items():
                if table_name.lower() in name.lower():
                    json_obj = obj
                    table_name = name
                    break

        if not json_obj:
            available = ", ".join(sorted(_generated_jsons.keys()))
            return f"Table '{table_name}' not found. Available: {available}"

        return f"Current JSON for {table_name}:\n```json\n{json.dumps(json_obj, indent=2)}\n```"

    @tool
    def modify_json(table_name: str, instruction: str) -> str:
        """Modify a generated FHIR StructureDefinition JSON based on a natural language instruction.
        Provide the table name and the modification instruction.
        Examples:
        - "Change column phq9_1 sensitivity to PHI"
        - "Add a structural link to bhs-analysis-diagnoses"
        - "Update the description to mention depression screening"
        """
        # Find the JSON
        actual_name = None
        json_obj = _generated_jsons.get(table_name)
        if json_obj:
            actual_name = table_name
        else:
            for name, obj in _generated_jsons.items():
                if table_name.lower() in name.lower():
                    json_obj = obj
                    actual_name = name
                    break

        if not json_obj:
            available = ", ".join(sorted(_generated_jsons.keys()))
            return f"Table '{table_name}' not found. Available: {available}"

        # Build existing metadata summary
        summaries = []
        for name, obj in _generated_jsons.items():
            if name != actual_name:
                summaries.append(f"  - {obj.get('id', name)}: {obj.get('title', 'N/A')}")
        existing_summary = "\n".join(summaries) if summaries else "None"

        system_prompt = REFINEMENT_SYSTEM_PROMPT.format(
            current_json=json.dumps(json_obj, indent=2),
            existing_metadata_summary=existing_summary,
        )

        try:
            response = call_gemini_fast(
                system_prompt=system_prompt,
                user_message=instruction,
                project_id=project_id,
            )
            updated_json = extract_json_from_response(response)
            if updated_json:
                _generated_jsons[actual_name] = updated_json
                return f"✅ Successfully updated {actual_name}. The JSON has been modified."
            else:
                return f"❌ Could not parse the modified JSON. Response preview: {response[:300]}"
        except Exception as e:
            return f"❌ Modification failed: {str(e)}"

    @tool
    def get_table_schema(table_name: str) -> str:
        """Get the BigQuery schema (columns, types) for a table.
        Returns column names, data types, and descriptions from INFORMATION_SCHEMA.
        """
        info = _bq_tables.get(table_name)
        if not info:
            for name, t in _bq_tables.items():
                if table_name.lower() in name.lower():
                    info = t
                    break

        if not info:
            available = ", ".join(sorted(_bq_tables.keys()))
            return f"Table '{table_name}' not found. Available: {available}"

        return format_bq_schema_for_prompt(info)

    @tool
    def list_generated_files() -> str:
        """List all generated FHIR metadata files with their current status.
        Shows StructureDefinitions, ValueSets, and DataProfiles.
        """
        lines = ["Generated Metadata Files:", ""]

        for name, obj in sorted(_generated_jsons.items()):
            sd_id = obj.get("id", "unknown")
            title = obj.get("title", "N/A")
            col_count = len([
                e for e in obj.get("differential", {}).get("element", [])
                if "." in e.get("path", "")
            ])
            lines.append(f"📄 {name}")
            lines.append(f"   ID: {sd_id} | Title: {title} | Columns: {col_count}")

            vs_list = _generated_value_sets.get(name, [])
            if vs_list:
                for vs in vs_list:
                    lines.append(f"   📋 ValueSet: {vs.get('id', 'unknown')}")
            lines.append("")

        if not _generated_jsons:
            lines.append("  (No files generated yet)")

        return "\n".join(lines)

    return [view_current_json, modify_json, get_table_schema, list_generated_files]


# ── Agent Graph ───────────────────────────────────────────────────────────────

def create_agent(
    generated_jsons: dict[str, dict],
    value_sets: dict[str, list[dict]],
    bq_tables: list[BQTableInfo],
    project_id: Optional[str] = None,
    llm_model: str = "gemini-2.5-pro",
    llm_location: str = "us-central1",
):
    """
    Create the LangGraph refinement agent.

    Args:
        generated_jsons: Dict of table_name → StructureDefinition JSON.
        value_sets: Dict of table_name → list of ValueSet JSONs.
        bq_tables: List of BQ table info objects.
        project_id: GCP project for Vertex AI.
        llm_model: LLM model for refinement.
        llm_location: Vertex AI region.

    Returns:
        Compiled LangGraph and system prompt.
    """
    # Set shared mutable state
    set_agent_state(generated_jsons, value_sets, bq_tables)

    # Create tools
    tools = _make_tools(project_id)

    # Create LLM with tool binding
    from langchain_google_vertexai import ChatVertexAI

    llm = ChatVertexAI(
        model_name=llm_model,
        project=project_id,
        location=llm_location,
        temperature=0.1,
        max_output_tokens=65536,
    )
    llm_with_tools = llm.bind_tools(tools)

    # System prompt for the agent
    agent_system_prompt = """You are a FHIR metadata specialist helping a data steward refine
generated FHIR StructureDefinition JSON files.

You have tools to:
1. View current JSONs — use view_current_json to see the full content
2. Modify JSONs — use modify_json to apply changes based on instructions
3. View BQ schemas — use get_table_schema to see the original BigQuery schema
4. List files — use list_generated_files to see all generated metadata

When the user asks to make a change:
1. First understand which table/column they're referring to
2. Use modify_json to apply the change
3. Confirm what was changed

When the user asks a question about the metadata:
1. Use view_current_json or list_generated_files to look up information
2. Provide a clear answer

Be helpful, specific, and always confirm changes were applied."""

    # Define graph nodes
    def call_model(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=agent_system_prompt))
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Run Agent ─────────────────────────────────────────────────────────────────

def _content_to_str(content) -> str:
    """Ensure content is always a string."""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content) if content else ""


def run_agent(
    compiled_graph,
    user_message: str,
    conversation_history: Optional[list[BaseMessage]] = None,
) -> tuple[dict, list[BaseMessage]]:
    """
    Run the refinement agent with a user message.

    Returns:
        Tuple of (result_dict, updated_conversation_history)
        result_dict has keys: response, modified_tables
    """
    messages = list(conversation_history or [])
    messages.append(HumanMessage(content=user_message))

    result = compiled_graph.invoke({"messages": messages})
    updated_messages = list(result["messages"])

    # Extract final AI response
    response_text = ""
    for msg in reversed(updated_messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            response_text = _content_to_str(msg.content)
            break

    if not response_text:
        last = updated_messages[-1]
        response_text = _content_to_str(getattr(last, "content", str(last)))

    # Check if any modifications were made (look for success messages in tool outputs)
    modified_tables = []
    for msg in updated_messages:
        if hasattr(msg, "type") and msg.type == "tool":
            content = _content_to_str(msg.content)
            if "Successfully updated" in content:
                # Extract table name from "Successfully updated TABLE_NAME"
                import re
                match = re.search(r"Successfully updated (.+?)\.", content)
                if match:
                    modified_tables.append(match.group(1))

    return {
        "response": response_text,
        "modified_tables": modified_tables,
    }, updated_messages


# ── Get Current State ─────────────────────────────────────────────────────────

def get_current_jsons() -> dict[str, dict]:
    """Get the current state of all generated JSONs (may have been modified by agent)."""
    return _generated_jsons.copy()


def get_current_json(table_name: str) -> Optional[dict]:
    """Get the current JSON for a specific table."""
    return _generated_jsons.get(table_name)
