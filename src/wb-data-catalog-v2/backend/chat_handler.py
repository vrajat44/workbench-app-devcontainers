"""
Chat session management for the Data Catalog backend.

Uses a tiered context strategy for speed:
  - Tier 1: Pre-generated _catalog_context.md (one GCS read, cached in session)
  - Tier 2: Full table profile loaded on-demand when user asks about a specific table
  - Session cache: prompt cached after first build, zero GCS reads on subsequent messages
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from threading import Lock

logger = logging.getLogger(__name__)
from typing import Any, AsyncIterator, Optional

from verily_chat import ChatContext, ChatMessage, ChatSession, chat as metadata_chat
from verily_chat.chat import chat_stream as metadata_chat_stream
from verily_profiler import read_tech_profile, read_sem_profile
from verily_profiler.storage import read_catalog_context


MAX_SESSIONS = 100


class ChatSessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(
        self,
        session_id: Optional[str],
        context: ChatContext,
        mode: str = "metadata",
    ) -> ChatSession:
        with self._lock:
            if session_id and session_id in self._sessions:
                sess = self._sessions[session_id]
                sess.context = context
                sess.mode = mode
                return sess
            if len(self._sessions) >= MAX_SESSIONS:
                oldest_key = next(iter(self._sessions))
                del self._sessions[oldest_key]
            sess = ChatSession(context=context, mode=mode)
            self._sessions[sess.session_id] = sess
            return sess

    def get(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                return True
            return False

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


chat_store = ChatSessionStore()

_agent_cache: dict[str, Any] = {}

_context_md_cache: dict[str, str] = {}
_table_detail_cache: dict[str, dict] = {}


def _load_catalog_context_md(
    data_project: str,
    bucket: str,
    billing_project: str,
) -> str:
    """Load the pre-generated .md context, with in-memory cache."""
    cache_key = f"{data_project}:{bucket}"
    if cache_key in _context_md_cache:
        return _context_md_cache[cache_key]

    md = read_catalog_context(bucket, data_project, billing_project_id=billing_project)
    if md:
        _context_md_cache[cache_key] = md
        return md
    return ""


def _load_table_detail(
    fq_table: str,
    bucket: str,
    billing_project: str,
) -> tuple[Optional[dict], Optional[dict]]:
    """Load full tech + sem profiles for one table, with cache."""
    if fq_table in _table_detail_cache:
        cached = _table_detail_cache[fq_table]
        return cached.get("tech"), cached.get("sem")

    tech = read_tech_profile(bucket, fq_table, project_id=billing_project)
    sem = read_sem_profile(bucket, fq_table, project_id=billing_project)
    _table_detail_cache[fq_table] = {"tech": tech, "sem": sem}
    return tech, sem


_all_profiles_cache: dict[str, tuple[dict[str, dict], dict[str, dict]]] = {}


def _load_all_profiles(
    bucket: str,
    data_project: str,
    billing_project: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load all available tech + sem profiles for agent mode, with cache."""
    cache_key = f"{data_project}:{bucket}"
    if cache_key in _all_profiles_cache:
        return _all_profiles_cache[cache_key]

    from verily_profiler.storage import scan_profile_availability

    tech_profiles: dict[str, dict] = {}
    sem_profiles: dict[str, dict] = {}

    try:
        avail = scan_profile_availability(bucket, data_project, billing_project_id=billing_project)
        for fq, info in avail.items():
            if info.get("technical"):
                tech = read_tech_profile(bucket, fq, project_id=billing_project)
                if tech:
                    tech_profiles[fq] = tech
            if info.get("semantic"):
                sem = read_sem_profile(bucket, fq, project_id=billing_project)
                if sem:
                    sem_profiles[fq] = sem
    except Exception as e:
        logger.warning(f"Failed to load all profiles for agent mode: {e}")

    _all_profiles_cache[cache_key] = (tech_profiles, sem_profiles)
    return tech_profiles, sem_profiles


def _schema_link(
    question: str,
    catalog_md: str,
    billing_project: str,
) -> list[str]:
    """Use a fast LLM call to identify which tables are relevant to the question."""
    from verily_profiler.llm import call_gemini, extract_json_from_response, detect_available_model

    model = detect_available_model(billing_project)
    if not model or not catalog_md:
        return []

    prompt = f"""Given the catalog of tables below, identify which tables are relevant to answer this question.
Return ONLY a JSON array of fully-qualified table names (e.g. ["project.dataset.table1", "project.dataset.table2"]).
Return at most 5 tables. If no tables are relevant, return [].

QUESTION: {question}

CATALOG:
{catalog_md}"""

    try:
        raw = call_gemini(prompt, model=model, project_id=billing_project, temperature=0.0)
        tables = extract_json_from_response(raw)
        if isinstance(tables, list):
            return [str(t) for t in tables[:5]]
    except Exception as e:
        logger.warning(f"Schema linking failed: {e}")
    return []


def invalidate_context_cache(data_project: str = "", bucket: str = ""):
    """Clear cached context (called after profiling)."""
    if data_project and bucket:
        key = f"{data_project}:{bucket}"
        _context_md_cache.pop(key, None)
        _all_profiles_cache.pop(key, None)
    else:
        _context_md_cache.clear()
        _all_profiles_cache.clear()
    _table_detail_cache.clear()
    _agent_cache.clear()


def build_context(
    fq_table: Optional[str],
    data_project: str,
    billing_project: str,
    bucket: str,
    mode: str = "metadata",
    detail_level: str = "summary",
    question: str = "",
) -> ChatContext:
    """
    Build ChatContext using tiered strategy:
    - summary: catalog .md only (fast, good for discovery)
    - full: schema-linked — LLM identifies relevant tables, loads their full profiles
    - Focused table always gets full profiles loaded
    """
    tech_profiles: dict[str, dict] = {}
    sem_profiles: dict[str, dict] = {}

    catalog_md = _load_catalog_context_md(data_project, bucket, billing_project)

    if fq_table:
        tech, sem = _load_table_detail(fq_table, bucket, billing_project)
        if tech:
            tech_profiles[fq_table] = tech
        if sem:
            sem_profiles[fq_table] = sem

    if detail_level == "full" and question and catalog_md:
        linked_tables = _schema_link(question, catalog_md, billing_project)
        if linked_tables:
            logger.info(f"Schema linked {len(linked_tables)} tables: {linked_tables}")
            for fq in linked_tables:
                if fq not in tech_profiles:
                    tech, sem = _load_table_detail(fq, bucket, billing_project)
                    if tech:
                        tech_profiles[fq] = tech
                    if sem:
                        sem_profiles[fq] = sem
        else:
            tech_profiles, sem_profiles = _load_all_profiles(bucket, data_project, billing_project)
    elif mode == "agent" and not fq_table:
        tech_profiles, sem_profiles = _load_all_profiles(bucket, data_project, billing_project)

    table_summaries: list[dict] = []
    if catalog_md:
        table_summaries = [{"_catalog_context_md": catalog_md}]

    return ChatContext(
        project_id=data_project,
        billing_project=billing_project,
        fq_table=fq_table,
        tech_profiles=tech_profiles,
        sem_profiles=sem_profiles,
        table_summaries=table_summaries,
    )


async def handle_chat_message(
    message: str,
    mode: str,
    fq_table: Optional[str],
    session_id: Optional[str],
    data_project: str,
    billing_project: str,
    bucket: str,
    model: Optional[str],
    detail_level: str = "summary",
) -> dict[str, Any]:
    ctx = build_context(fq_table, data_project, billing_project, bucket, mode=mode, detail_level=detail_level, question=message)
    session = chat_store.get_or_create(session_id, ctx, mode)
    session.add_message(ChatMessage(role="user", content=message, mode=mode))

    from verily_profiler.llm import detect_available_model
    detected = detect_available_model(billing_project)
    metadata_model = model or detected
    agent_model = model or detected

    if mode == "agent":
        reply = await _handle_agent(message, session, agent_model, billing_project)
    else:
        reply = await asyncio.to_thread(
            metadata_chat,
            message,
            ctx,
            history=session.messages[:-1],
            model=metadata_model,
            project_id=billing_project,
        )

    session.add_message(reply)

    return {
        "session_id": session.session_id,
        "message": reply.to_json_dict(),
    }


async def _handle_agent(
    message: str,
    session: ChatSession,
    model: str,
    billing_project: str,
) -> ChatMessage:
    try:
        from verily_chat.agent import create_chat_agent, chat_agent
    except ImportError:
        return ChatMessage(
            role="assistant",
            content="Agent mode is not available. Install verily-chat[agent] dependencies.",
            mode="agent",
        )

    # Include profile count in cache key so the graph is rebuilt when profiles are loaded
    n_profiles = len(session.context.tech_profiles) + len(session.context.sem_profiles)
    graph_key = f"{session.context.project_id}:{session.context.fq_table or 'all'}:{model}:{n_profiles}"

    if graph_key not in _agent_cache:
        graph, _ = create_chat_agent(
            session.context,
            model=model,
            project_id=billing_project,
        )
        _agent_cache[graph_key] = {"graph": graph}

    graph = _agent_cache[graph_key]["graph"]

    agent_history = getattr(session, "_agent_history", [])

    def _run():
        try:
            reply, updated_history = chat_agent(
                message,
                graph,
                history=agent_history,
            )
            session._agent_history = updated_history
            return reply
        except Exception as e:
            session._agent_history = []
            raise

    return await asyncio.to_thread(_run)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def handle_chat_stream(
    message: str,
    mode: str,
    fq_table: Optional[str],
    session_id: Optional[str],
    data_project: str,
    billing_project: str,
    bucket: str,
    model: Optional[str],
    detail_level: str = "summary",
) -> AsyncIterator[str]:
    """Yield SSE events for a streaming chat response."""
    try:
        ctx = build_context(
            fq_table, data_project, billing_project, bucket,
            mode=mode, detail_level=detail_level, question=message,
        )
        session = chat_store.get_or_create(session_id, ctx, mode)
        session.add_message(ChatMessage(role="user", content=message, mode=mode))

        yield _sse({"type": "session", "session_id": session.session_id})

        from verily_profiler.llm import detect_available_model
        detected = detect_available_model(billing_project)
        resolved_model = model or detected

        queue: asyncio.Queue = asyncio.Queue()

        if mode == "agent":
            def _run_agent():
                for event in _stream_agent_events(message, session, resolved_model, billing_project):
                    queue.put_nowait(event)
                queue.put_nowait(None)

            task = asyncio.get_event_loop().run_in_executor(None, _run_agent)
        else:
            yield _sse({"type": "status", "text": "Thinking..."})

            def _run_metadata():
                for chunk in metadata_chat_stream(
                    message, ctx,
                    history=session.messages[:-1],
                    model=resolved_model,
                    project_id=billing_project,
                ):
                    queue.put_nowait(("chunk", chunk))
                queue.put_nowait(None)

            task = asyncio.get_event_loop().run_in_executor(None, _run_metadata)

        full_text = ""
        sql = None
        while True:
            item = await queue.get()
            if item is None:
                break
            event_type, data = item
            if event_type == "chunk":
                full_text += data
                yield _sse({"type": "chunk", "text": data})
            elif event_type == "status":
                yield _sse({"type": "status", "text": data})
            elif event_type == "history":
                session._agent_history = data
            elif event_type == "error":
                yield _sse({"type": "error", "text": data})
                await task
                return

        await task

        m = re.search(r"```sql\s*\n(.*?)```", full_text, re.DOTALL)
        if m:
            sql = m.group(1).strip()

        reply = ChatMessage(role="assistant", content=full_text, sql=sql, mode=mode)
        session.add_message(reply)
        yield _sse({"type": "done", "sql": sql, "mode": mode})

    except Exception as e:
        import traceback
        logger.error(f"Stream error:\n{traceback.format_exc()}")
        yield _sse({"type": "error", "text": str(e)})


def _stream_agent_events(
    message: str,
    session: ChatSession,
    model: str,
    billing_project: str,
):
    """Synchronous generator yielding (event_type, data) tuples for agent mode."""
    try:
        from verily_chat.agent import create_chat_agent, chat_agent_stream
    except ImportError:
        yield ("chunk", "Agent mode is not available.")
        return

    n_profiles = len(session.context.tech_profiles) + len(session.context.sem_profiles)
    graph_key = f"{session.context.project_id}:{session.context.fq_table or 'all'}:{model}:{n_profiles}"

    if graph_key not in _agent_cache:
        graph, _ = create_chat_agent(
            session.context,
            model=model,
            project_id=billing_project,
        )
        _agent_cache[graph_key] = {"graph": graph}

    graph = _agent_cache[graph_key]["graph"]
    agent_history = getattr(session, "_agent_history", [])

    try:
        for event_type, data in chat_agent_stream(message, graph, history=agent_history):
            if event_type == "status":
                yield ("status", data)
            elif event_type == "text":
                yield ("chunk", data)
            elif event_type == "history":
                yield ("history", data)
    except Exception as e:
        session._agent_history = []
        logger.error(f"Agent stream error: {e}")
        yield ("error", str(e))
