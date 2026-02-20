"""
Step 5: LangGraph Agent
Combines the metadata loader, prompt engine, and BQ executor into an agentic
loop with tool-calling. Supports multi-turn conversations.

The agent has access to these tools:
  1. query_bigquery — Execute a SQL query and return results
  2. get_table_schema — Look up column details for a specific table
  3. list_available_tables — Show all tables with descriptions

Usage:
    from agent import create_agent, run_agent

    agent = create_agent(
        json_metadata_dir="/path/to/json/",
        bq_project_id="your-project",
    )
    response = run_agent(agent, "Show me BHS participants with high cardiac risk")
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from bq_executor import BigQueryExecutor, QueryResult, dataframe_to_display
from metadata_loader import (
    TableSchema,
    format_schemas_for_prompt,
    format_table_summary,
    load_metadata,
    resolve_against_bigquery,
)
from prompt_engine import build_system_prompt, extract_sql_from_response


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Track the last query result for display
    last_query_result: Optional[QueryResult]


# ── Tool Definitions ──────────────────────────────────────────────────────────
# These are created as closures inside create_agent() to capture schemas/executor

def _make_tools(
    schemas: dict[str, TableSchema],
    executor: Optional[BigQueryExecutor],
):
    """Create tool functions with access to schemas and executor."""

    @tool
    def query_bigquery(sql_query: str) -> str:
        """Execute a BigQuery SQL query and return results as a formatted table.
        Use this when you have a SQL query ready to run.
        The query should use fully-qualified table names from the available schema.
        Returns the first 100 rows of results, or an error message if the query fails.
        """
        if executor is None:
            return "ERROR: BigQuery executor not configured. Need a GCP project ID."

        result = executor.execute_query(sql_query)

        if result.success:
            row_count = result.row_count
            bytes_mb = result.bytes_processed_mb
            header = f"Query successful: {row_count} rows returned"
            if bytes_mb:
                header += f" ({bytes_mb} MB processed)"
            header += f"\n\nExecuted SQL:\n```sql\n{sql_query}\n```"

            table_str = dataframe_to_display(result.dataframe, max_rows=100)
            return f"{header}\n\n{table_str}"
        else:
            return f"QUERY FAILED:\n{result.error_message}\n\nOriginal SQL:\n```sql\n{sql_query}\n```"

    @tool
    def query_bigquery_with_retry(sql_query: str) -> str:
        """Execute a BigQuery SQL query with automatic error fixing.
        If the query fails, an LLM will attempt to fix it and retry up to 3 times.
        Use this for complex queries that might have issues.
        """
        if executor is None:
            return "ERROR: BigQuery executor not configured. Need a GCP project ID."

        result = executor.execute_with_retry(sql_query, schemas, max_retries=3)

        if result.success:
            row_count = result.row_count
            attempts = result.attempts
            header = f"Query successful after {attempts} attempt(s): {row_count} rows returned"
            if attempts > 1:
                header += "\n(Auto-fixed query errors during retry)"
            header += f"\n\nFinal SQL:\n```sql\n{result.sql}\n```"

            table_str = dataframe_to_display(result.dataframe, max_rows=100)
            return f"{header}\n\n{table_str}"
        else:
            history_str = "\n".join(
                f"  Attempt {h['attempt']}: {'✓' if h['success'] else '✗'} {h.get('error', '')[:200]}"
                for h in result.attempt_history
            )
            return f"QUERY FAILED after {result.attempts} attempts:\n{result.error_message}\n\nAttempt history:\n{history_str}"

    @tool
    def get_table_schema(table_name: str) -> str:
        """Get detailed column information for a specific table.
        Provide the table name as it appears in the metadata (e.g., 'bhs.admin.COEVAL').
        Returns column names, types, descriptions, and sensitivity labels.
        """
        # Try exact match first
        table = schemas.get(table_name)

        # Try partial match
        if not table:
            for name, t in schemas.items():
                if table_name.lower() in name.lower():
                    table = t
                    break

        if not table:
            available = ", ".join(sorted(schemas.keys()))
            return f"Table '{table_name}' not found. Available tables: {available}"

        lines = [
            f"TABLE: {table.bq_table_name}",
            f"Title: {table.title}",
            f"Description: {table.description}",
            f"Grain: {table.purpose}",
            f"Primary Key: {table.primary_key}",
            f"Confidentiality: {table.confidentiality or 'N/A'}",
            f"Compliance: {table.compliance_zone or 'N/A'}",
            "",
            "COLUMNS:",
        ]

        for col in table.columns:
            sens = f" [{col.sensitivity_label}]" if col.sensitivity_label else ""
            req = "REQUIRED" if col.is_required else "OPTIONAL"
            lines.append(f"  {col.name} ({col.data_type}, {req}{sens})")
            lines.append(f"    {col.full_description}")
            if col.value_set_binding:
                lines.append(f"    Values: {col.value_set_binding}")
            if col.comment:
                lines.append(f"    Note: {col.comment}")

        join_targets = [l.target_table_name or l.target_profile_url for l in table.join_links]
        if join_targets:
            lines.append("")
            lines.append(f"JOINS TO: {', '.join(join_targets)}")

        return "\n".join(lines)

    @tool
    def list_available_tables() -> str:
        """List all available tables with their descriptions and column counts.
        Use this to understand what data is available before writing queries.
        """
        return format_table_summary(schemas)

    return [query_bigquery, query_bigquery_with_retry, get_table_schema, list_available_tables]


# ── Agent Graph ───────────────────────────────────────────────────────────────

def create_agent(
    json_metadata_dir: str,
    bq_project_id: Optional[str] = None,
    data_project_id: Optional[str | list[str]] = None,
    llm_model: str = "gemini-2.5-pro",
    llm_location: str = "us-central1",
):
    """
    Create the LangGraph agent with all tools configured.

    Args:
        json_metadata_dir: Path to FHIR StructureDefinition JSON files.
        bq_project_id: GCP project for Vertex AI / LLM calls and BQ job execution.
        data_project_id: GCP project(s) where BQ data lives (from Workbench Data Collections).
                         Can be a single string or list of strings.
                         If None, defaults to [bq_project_id].
        llm_model: Gemini model name.
        llm_location: Vertex AI region for Gemini.

    Returns:
        Compiled LangGraph, schemas dict, and system prompt.
    """
    # The data projects may differ from the LLM project
    data_projects = data_project_id or ([bq_project_id] if bq_project_id else [])
    if isinstance(data_projects, str):
        data_projects = [data_projects]

    # Load metadata from FHIR JSONs (supports local path or gs:// URI)
    schemas = load_metadata(json_metadata_dir)

    # Resolve FHIR table names against real BQ tables at runtime
    if data_projects:
        print(f"🔍 Resolving metadata against BQ project(s): {data_projects}")
        schemas = resolve_against_bigquery(schemas, data_projects)

    # Build system prompt with resolved table names (no extra project prefix needed)
    system_prompt = build_system_prompt(schemas)

    # Create BQ executor:
    # - billing_project = LLM project (has bigquery.jobs.create permission)
    # - data is accessed via fully-qualified table names in SQL
    executor = None
    if data_projects:
        try:
            executor = BigQueryExecutor(
                project_id=data_projects[0],
                billing_project_id=bq_project_id,  # Use LLM project for job execution
            )
        except Exception as e:
            print(f"Warning: Could not initialize BigQuery executor: {e}")

    # Create tools
    tools = _make_tools(schemas, executor)

    # Create LLM with tool binding — use Vertex AI (enterprise quotas)
    from langchain_google_vertexai import ChatVertexAI

    llm = ChatVertexAI(
        model_name=llm_model,
        project=bq_project_id,
        location="us-central1",
        temperature=0.1,
        max_output_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(tools)

    # Define graph nodes
    def call_model(state: AgentState) -> dict:
        """Invoke the LLM with conversation history."""
        messages = list(state["messages"])

        # Ensure system prompt is first message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Decide whether to call tools or finish."""
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

    compiled = graph.compile()

    return compiled, schemas, system_prompt


def _content_to_str(content) -> str:
    """Ensure content is always a string (handles list-of-blocks from Vertex AI)."""
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
    Run the agent with a user message.

    Args:
        compiled_graph: The compiled LangGraph.
        user_message: The user's natural language input.
        conversation_history: Previous messages for multi-turn.

    Returns:
        Tuple of (result_dict, updated_conversation_history)
        result_dict has keys: response, sql, tool_output
    """
    import re

    messages = list(conversation_history or [])
    messages.append(HumanMessage(content=user_message))

    result = compiled_graph.invoke({"messages": messages})

    updated_messages = list(result["messages"])

    # Extract final AI response text
    response_text = ""
    for msg in reversed(updated_messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            response_text = _content_to_str(msg.content)
            break

    if not response_text:
        last = updated_messages[-1]
        response_text = _content_to_str(getattr(last, "content", str(last)))

    # Extract SQL and tool output from tool messages (most recent first)
    sql_found = ""
    tool_output = ""
    for msg in reversed(updated_messages):
        if hasattr(msg, "type") and msg.type == "tool":
            tool_text = _content_to_str(msg.content)
            if not tool_output:
                tool_output = tool_text
            # Look for SQL in the tool response
            if not sql_found:
                sql_match = re.search(r"```sql\s*\n(.*?)```", tool_text, re.DOTALL)
                if sql_match:
                    sql_found = sql_match.group(1).strip()

    # Also check if the AI response itself contains SQL
    if not sql_found:
        sql_match = re.search(r"```sql\s*\n(.*?)```", response_text, re.DOTALL)
        if sql_match:
            sql_found = sql_match.group(1).strip()

    return {
        "response": response_text,
        "sql": sql_found,
        "tool_output": tool_output,
    }, updated_messages


# ── Lightweight mode (no LangGraph, for Step 2-3 testing) ─────────────────────

def run_simple_query(
    user_message: str,
    schemas: dict[str, TableSchema],
    data_project_id: Optional[str] = None,
    llm_model: str = "gemini-2.5-pro",
    llm_project_id: Optional[str] = None,
    llm_location: str = "us-central1",
) -> dict:
    """
    Simple non-agentic flow: NL → SQL → Execute → Return.
    Useful for testing Steps 2-3 without LangGraph.

    Args:
        user_message: Natural language question.
        schemas: Already-resolved table schemas.
        data_project_id: GCP project where BQ data lives (for query execution).
        llm_model: Gemini model name.
        llm_project_id: GCP project for Vertex AI (defaults to data_project_id).
        llm_location: Vertex AI region.

    Returns dict with keys: response, sql, result, error
    """
    from prompt_engine import build_system_prompt, call_gemini, extract_sql_from_response

    # System prompt uses resolved table names — no project prefix needed
    system_prompt = build_system_prompt(schemas)

    # Step 2: NL → SQL via Gemini
    llm_response = call_gemini(
        system_prompt=system_prompt,
        user_message=user_message,
        model_name=llm_model,
        project_id=llm_project_id or data_project_id,
        location=llm_location,
    )

    sql = extract_sql_from_response(llm_response)

    result_dict = {
        "response": llm_response,
        "sql": sql,
        "result": None,
        "error": None,
    }

    # Step 3: Execute SQL if we got it and have a data project
    if sql and data_project_id:
        executor = BigQueryExecutor(
            project_id=data_project_id,
            billing_project_id=llm_project_id or data_project_id,
        )
        query_result = executor.execute_with_retry(sql, schemas)

        if query_result.success:
            result_dict["result"] = query_result
        else:
            result_dict["error"] = query_result.error_message
            result_dict["result"] = query_result

    return result_dict


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LangGraph Agent — Step 5")
    print()
    print("To create and run the agent:")
    print()
    print("  from agent import create_agent, run_agent")
    print()
    print("  graph, schemas, prompt = create_agent(")
    print("      json_metadata_dir='/path/to/json/',")
    print("      bq_project_id='your-project-id',")
    print("  )")
    print()
    print("  response, history = run_agent(graph, 'How many BHS participants are eligible?')")
    print("  print(response)")
    print()
    print("  # Multi-turn:")
    print("  response2, history = run_agent(graph, 'Now filter for high risk cardio', history)")

