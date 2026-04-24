"""
Generate a compact markdown catalog context file from profiling output.

Used as fast LLM context for chat: one GCS read instead of hundreds.
Regenerated after each profiling run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def generate_catalog_context_md(
    project_id: str,
    profiles: dict[str, dict[str, Any]],
) -> str:
    """
    Generate a compact markdown summary of all profiled tables.

    Args:
        project_id: The data project ID.
        profiles: dict of fq_table -> {"tech": dict|None, "sem": dict|None}

    Returns:
        Markdown string suitable for LLM context injection.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    table_count = len(profiles)
    datasets = set()
    for fq in profiles:
        parts = fq.split(".")
        if len(parts) >= 2:
            datasets.add(parts[1])

    lines = [
        f"# Data Catalog: {project_id}",
        f"Generated: {now} | {len(datasets)} datasets, {table_count} tables",
        "",
    ]

    sorted_fqs = sorted(profiles.keys())
    current_ds = ""

    for fq in sorted_fqs:
        data = profiles[fq]
        tech = data.get("tech")
        sem = data.get("sem")
        parts = fq.split(".")
        ds = parts[1] if len(parts) >= 2 else "unknown"
        table = parts[2] if len(parts) >= 3 else fq

        if ds != current_ds:
            lines.append(f"---\n## Dataset: {ds}\n")
            current_ds = ds

        lines.append(f"### {fq}")
        lines.append(_format_table_summary(table, tech, sem))
        lines.append("")

    return "\n".join(lines)


def _format_table_summary(table: str, tech: Optional[dict], sem: Optional[dict]) -> str:
    parts: list[str] = []

    if sem:
        if sem.get("business_name"):
            parts.append(f"**{sem['business_name']}**")
        if sem.get("table_definition"):
            defn = sem["table_definition"]
            if len(defn) > 200:
                defn = defn[:200] + "..."
            parts.append(defn)

        domain = sem.get("semantic_domain", {})
        if domain.get("primary"):
            sub = f" / {domain['sub_domain']}" if domain.get("sub_domain") else ""
            parts.append(f"Domain: {domain['primary']}{sub}")

        if sem.get("granularity"):
            parts.append(f"Granularity: {sem['granularity']}")

        pk = sem.get("primary_key", {})
        if pk.get("columns") and pk.get("pk_type") != "none":
            cols = ", ".join(pk["columns"])
            parts.append(f"PK: {cols} ({pk.get('pk_type', '?')}, {pk.get('confidence', '?')})")

        if sem.get("entity_anchor"):
            etype = sem.get("entity_type", "")
            etype_str = f" ({etype})" if etype else ""
            parts.append(f"Entity: {sem['entity_anchor']}{etype_str}")
        if sem.get("cohort_dimensions"):
            dims = ", ".join(sem["cohort_dimensions"][:10])
            parts.append(f"Cohort dimensions: {dims}")
        if sem.get("structural_links"):
            link_strs = []
            for sl in sem["structural_links"][:5]:
                link_strs.append(f"{sl.get('source_column','')} → {sl.get('target_table','')}.{sl.get('target_column','')} ({sl.get('link_type','')})")
            parts.append(f"Joins: {'; '.join(link_strs)}")

    if tech:
        stats: list[str] = []
        if tech.get("row_count") is not None:
            stats.append(f"Rows: {tech['row_count']:,}")
        if tech.get("size_bytes") is not None:
            mb = tech["size_bytes"] / (1024 * 1024)
            stats.append(f"Size: {mb:.1f} MB")
        col_count = len(tech.get("columns", []))
        if col_count:
            stats.append(f"Cols: {col_count}")
        if stats:
            parts.append(" | ".join(stats))

    columns_block = _format_columns_compact(tech, sem)
    if columns_block:
        parts.append(columns_block)

    return "\n".join(parts)


def _format_columns_compact(tech: Optional[dict], sem: Optional[dict]) -> str:
    """One-line-per-column compact format."""
    tech_cols = {c.get("name", c.get("column_name", "")): c for c in (tech or {}).get("columns", [])}
    sem_cols = {c.get("name", c.get("column_name", "")): c for c in (sem or {}).get("columns", [])}

    all_names: list[str] = list(tech_cols.keys())
    for n in sem_cols:
        if n not in all_names:
            all_names.append(n)

    if not all_names:
        return ""

    lines = ["Columns:"]
    for name in all_names:
        tc = tech_cols.get(name, {})
        sc = sem_cols.get(name, {})
        dtype = tc.get("data_type", "?")
        nullable = "NULL" if tc.get("nullable", True) else "NOT NULL"

        info: list[str] = [f"`{name}` ({dtype}, {nullable})"]

        defn = sc.get("definition", "")
        if defn:
            short = defn[:80] + "..." if len(defn) > 80 else defn
            info.append(short)

        extras: list[str] = []
        if sc.get("sensitivity"):
            extras.append(f"[{sc['sensitivity']}]")
        if sc.get("unit_of_measure"):
            extras.append(f"Unit: {sc['unit_of_measure']}")
        if sc.get("measurement_method"):
            extras.append(f"Method: {sc['measurement_method']}")
        if tc.get("null_percent") is not None:
            extras.append(f"Nulls: {tc['null_percent']}%")
        if tc.get("distinct_count") is not None:
            extras.append(f"Distinct: {tc['distinct_count']:,}")

        cb = sc.get("concept_binding")
        if cb and cb.get("system") and cb.get("code"):
            sys_short = cb["system"].split("/")[-1]
            extras.append(f"Concept: {sys_short}:{cb['code']} ({cb.get('display', '')})")
        csb = sc.get("code_system_binding")
        if csb and csb.get("system"):
            sys_short = csb["system"].split("/")[-1]
            extras.append(f"CodeSystem: {sys_short} ({csb.get('display', '')})")

        if not cb and not csb:
            bindings = sc.get("terminology_bindings", [])
            if bindings:
                codes = [f"{b.get('system','').split('/')[-1]}:{b.get('code','')}" for b in bindings[:2]]
                extras.append(f"Terms: {', '.join(codes)}")

        vsb = sc.get("value_set_binding", [])
        if vsb:
            vals = ", ".join(vsb[:5])
            suffix = f" +{len(vsb)-5} more" if len(vsb) > 5 else ""
            extras.append(f"Values: [{vals}{suffix}]")

        if extras:
            info.append(" | ".join(extras))

        lines.append("- " + " — ".join(info))

    return "\n".join(lines)
