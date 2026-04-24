"""
Agent chat mode with LangGraph + BigQuery tool calling.

Requires optional dependencies: pip install verily-chat[agent]
"""

from __future__ import annotations

import re
from typing import Annotated, Optional, Sequence

from verily_chat.context import build_catalog_system_prompt
from verily_chat.models import ChatContext, ChatMessage

try:
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

    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False


def _check_deps():
    if not _AGENT_AVAILABLE:
        raise ImportError(
            "Agent mode requires extra dependencies. "
            "Install with: pip install verily-chat[agent]"
        )


if _AGENT_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[Sequence["BaseMessage"], add_messages]
        last_sql: Optional[str]
else:
    class AgentState(TypedDict):  # type: ignore[no-redef]
        messages: list
        last_sql: Optional[str]


def _make_tools(context: ChatContext, billing_project: str | None):
    """Create LangGraph tools with access to profiling context."""

    @tool
    def query_bigquery(sql_query: str) -> str:
        """Execute a BigQuery SQL query and return results as a formatted table.
        Use fully-qualified table names from the schema context.
        Returns the first 100 rows, or an error message on failure.
        """
        from google.cloud import bigquery

        try:
            client = bigquery.Client(project=billing_project or context.project_id)
            job = client.query(sql_query)
            result = job.result(max_results=100)
            rows = list(result) if result is not None else []
            if not rows:
                return f"Query successful: 0 rows returned.\n\nSQL:\n```sql\n{sql_query}\n```"

            schema_fields = result.schema or job.schema or []
            schema = [f.name for f in schema_fields] if schema_fields else list(rows[0].keys())

            lines = [" | ".join(schema)]
            lines.append(" | ".join("---" for _ in schema))
            for row in rows:
                try:
                    values = [str(row.get(name)) if hasattr(row, "get") else str(row[name]) for name in schema]
                except Exception:
                    values = [str(v) for v in (row.values() or [])]
                lines.append(" | ".join(values))

            header = f"Query successful: {len(rows)} rows returned"
            if getattr(job, "total_bytes_processed", None):
                mb = job.total_bytes_processed / (1024 * 1024)
                header += f" ({mb:.1f} MB processed)"
            return f"{header}\n\nSQL:\n```sql\n{sql_query}\n```\n\n{chr(10).join(lines)}"
        except Exception as e:
            import traceback
            print(f"[query_bigquery] Error: {e}")
            traceback.print_exc()
            return f"QUERY FAILED:\n{type(e).__name__}: {e}\n\nSQL:\n```sql\n{sql_query}\n```"

    @tool
    def get_table_schema(table_name: str) -> str:
        """Get detailed column information for a table.
        Accepts fully-qualified name (project.dataset.table) or partial match.
        """
        all_fqs = set(list(context.tech_profiles.keys()) + list(context.sem_profiles.keys()))

        match_fq = None
        for fq in all_fqs:
            if fq == table_name or fq.endswith(f".{table_name}"):
                match_fq = fq
                break
        if not match_fq:
            for fq in all_fqs:
                if table_name.lower() in fq.lower():
                    match_fq = fq
                    break

        if not match_fq:
            return f"Table '{table_name}' not found. Available: {', '.join(sorted(all_fqs))}"

        from verily_chat.context import format_table_for_prompt

        tech = context.tech_profiles.get(match_fq)
        sem = context.sem_profiles.get(match_fq)
        return format_table_for_prompt(match_fq, tech, sem)

    @tool
    def list_available_tables() -> str:
        """List all available tables with descriptions and row counts."""
        all_fqs = sorted(set(
            list(context.tech_profiles.keys()) + list(context.sem_profiles.keys())
        ))
        if not all_fqs:
            return "No profiled tables available."

        lines: list[str] = []
        for fq in all_fqs:
            tech = context.tech_profiles.get(fq, {})
            sem = context.sem_profiles.get(fq, {})
            name = sem.get("business_name", fq.split(".")[-1])
            desc = sem.get("table_definition", "")[:80]
            rc = tech.get("row_count")
            rc_str = f" ({rc:,} rows)" if rc else ""
            lines.append(f"- `{fq}` — {name}{rc_str}")
            if desc:
                lines.append(f"  {desc}")
        return "\n".join(lines)

    @tool
    def find_joinable_tables(table_name: str) -> str:
        """Find all tables that can be joined to the given table.
        Returns structural links grouped by link type (entity_key, foreign_key, shared_dimension).
        Use this when the user asks 'what can I join to X?' or needs cross-table queries.
        """
        all_fqs = set(list(context.tech_profiles.keys()) + list(context.sem_profiles.keys()))

        match_fq = None
        for fq in all_fqs:
            if fq == table_name or fq.endswith(f".{table_name}"):
                match_fq = fq
                break
        if not match_fq:
            for fq in all_fqs:
                if table_name.lower() in fq.lower():
                    match_fq = fq
                    break

        if not match_fq:
            return f"Table '{table_name}' not found. Available: {', '.join(sorted(all_fqs))}"

        sem = context.sem_profiles.get(match_fq, {})
        links = sem.get("structural_links", [])

        reverse_links: list[dict] = []
        for fq in all_fqs:
            if fq == match_fq:
                continue
            other_sem = context.sem_profiles.get(fq, {})
            for sl in other_sem.get("structural_links", []):
                target = sl.get("target_table", "")
                if match_fq.endswith(target) or target.endswith(match_fq.split(".")[-1]):
                    reverse_links.append({
                        "source_table": fq,
                        "source_column": sl.get("source_column", ""),
                        "target_column": sl.get("target_column", ""),
                        "link_type": sl.get("link_type", ""),
                        "cardinality": sl.get("cardinality", ""),
                    })

        if not links and not reverse_links:
            return f"No structural links found for `{match_fq}`. Try profiling with neighbor context."

        lines = [f"Joinable tables for `{match_fq}`:\n"]

        by_type: dict[str, list[str]] = {}
        for sl in links:
            lt = sl.get("link_type", "unknown")
            entry = f"  {sl.get('source_column','')} → {sl.get('target_table','')}.{sl.get('target_column','')} ({sl.get('cardinality','')})"
            by_type.setdefault(lt, []).append(entry)

        for rl in reverse_links:
            lt = rl.get("link_type", "unknown")
            entry = f"  {rl['target_column']} ← {rl['source_table']}.{rl['source_column']} ({rl.get('cardinality','')})"
            by_type.setdefault(lt, []).append(entry)

        for lt in ["entity_key", "foreign_key", "shared_dimension", "temporal"]:
            if lt in by_type:
                lines.append(f"**{lt}** ({len(by_type[lt])}):")
                lines.extend(by_type[lt])
                lines.append("")
                del by_type[lt]
        for lt, entries in by_type.items():
            lines.append(f"**{lt}** ({len(entries)}):")
            lines.extend(entries)
            lines.append("")

        return "\n".join(lines)

    @tool
    def find_concept_across_tables(concept: str) -> str:
        """Find all columns across all tables that represent a given concept.
        Search by concept code (e.g. 'LOINC:44261-6'), display name (e.g. 'PHQ-9'),
        or column name pattern (e.g. 'phq9'). Use this when the user asks
        'find all depression data' or 'where is PHQ-9 across datasets?'
        """
        concept_lower = concept.lower()
        results: list[str] = []

        all_fqs = sorted(set(list(context.tech_profiles.keys()) + list(context.sem_profiles.keys())))

        for fq in all_fqs:
            sem = context.sem_profiles.get(fq, {})
            for col in sem.get("columns", []):
                col_name = col.get("name", "")
                matched = False
                match_reason = ""

                cb = col.get("concept_binding")
                if cb and cb.get("code"):
                    code_match = concept_lower in f"{cb.get('system','').split('/')[-1]}:{cb['code']}".lower()
                    display_match = concept_lower in cb.get("display", "").lower()
                    if code_match or display_match:
                        matched = True
                        sys_short = cb["system"].split("/")[-1]
                        match_reason = f"Fixed Concept: {sys_short}:{cb['code']} — {cb.get('display', '')}"

                csb = col.get("code_system_binding")
                if not matched and csb and csb.get("system"):
                    sys_match = concept_lower in csb.get("system", "").lower()
                    display_match = concept_lower in csb.get("display", "").lower()
                    if sys_match or display_match:
                        matched = True
                        sys_short = csb["system"].split("/")[-1]
                        match_reason = f"Code System: {sys_short} — {csb.get('display', '')}"

                if not matched:
                    for tb in col.get("terminology_bindings", []):
                        code_str = f"{tb.get('system','').split('/')[-1]}:{tb.get('code','')}".lower()
                        if concept_lower in code_str or concept_lower in tb.get("display", "").lower():
                            matched = True
                            match_reason = f"Binding: {tb.get('system','').split('/')[-1]}:{tb.get('code','')} — {tb.get('display', '')}"
                            break

                if not matched and concept_lower in col_name.lower():
                    matched = True
                    match_reason = f"Column name match"

                if matched:
                    method = col.get("measurement_method", "")
                    method_str = f" [Method: {method}]" if method else ""
                    results.append(f"  `{fq}`.{col_name} — {match_reason}{method_str}")

        if not results:
            return f"No columns found matching concept '{concept}' across {len(all_fqs)} tables."

        header = f"Found {len(results)} column(s) matching '{concept}':\n"
        return header + "\n".join(results)

    return [query_bigquery, get_table_schema, list_available_tables, find_joinable_tables, find_concept_across_tables]


def create_chat_agent(
    context: ChatContext,
    model: str = "gemini-2.5-flash",
    project_id: str | None = None,
    location: str = "global",
):
    """
    Create a LangGraph agent with BigQuery tools.

    Args:
        context: Profiling metadata context.
        model: Gemini model name.
        project_id: GCP project for Vertex AI and BQ billing.

    Returns:
        Tuple of (compiled_graph, system_prompt).
    """
    _check_deps()
    from langchain_google_vertexai import ChatVertexAI

    bp = project_id or context.billing_project or context.project_id
    system_prompt = build_catalog_system_prompt(context, mode="agent")
    tools = _make_tools(context, bp)

    llm = ChatVertexAI(
        model_name=model,
        project=bp,
        location=location,
        temperature=0.1,
        max_output_tokens=65536,
    )
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state):
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(), system_prompt


def chat_agent(
    message: str,
    compiled_graph,
    history: list | None = None,
) -> tuple[ChatMessage, list]:
    """
    Run the agent with a user message.

    Args:
        message: The user's question.
        compiled_graph: From create_chat_agent().
        history: Previous LangChain messages for multi-turn.

    Returns:
        Tuple of (ChatMessage, updated_history).
    """
    _check_deps()

    messages = list(history or [])
    messages.append(HumanMessage(content=message))

    result = compiled_graph.invoke({"messages": messages})
    updated = list(result["messages"])

    response_text = ""
    for msg in reversed(updated):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, list):
                text_parts: list[str] = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") in (None, "text") and b.get("text"):
                            text_parts.append(b["text"])
                    elif isinstance(b, str):
                        text_parts.append(b)
                response_text = "\n".join(text_parts)
            else:
                response_text = str(content) if content else ""
            break

    if not response_text and updated:
        last = updated[-1]
        response_text = str(getattr(last, "content", str(last)))

    sql = None
    for msg in reversed(updated):
        if hasattr(msg, "type") and msg.type == "tool":
            tool_text = str(msg.content) if msg.content else ""
            m = re.search(r"```sql\s*\n(.*?)```", tool_text, re.DOTALL)
            if m:
                sql = m.group(1).strip()
                break
    if not sql:
        m = re.search(r"```sql\s*\n(.*?)```", response_text, re.DOTALL)
        if m:
            sql = m.group(1).strip()

    return ChatMessage(
        role="assistant",
        content=response_text,
        sql=sql,
        mode="agent",
    ), updated
