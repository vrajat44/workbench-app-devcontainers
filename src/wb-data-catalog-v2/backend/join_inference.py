"""
Cross-table join inference.

Analyzes semantic and technical profiles across tables in a dataset to
infer likely join paths based on column name matching and terminology
code alignment.
"""

from __future__ import annotations

from typing import Any

# Column names that are too generic to infer a meaningful join relationship.
_GENERIC_SKIP_NAMES = frozenset({
    "id", "name", "type", "status", "description",
    "created_at", "updated_at",
})

# Data types that are compatible for join matching.
_COMPATIBLE_TYPE_GROUPS: list[set[str]] = [
    {"STRING", "BYTES"},
    {"INT64", "INTEGER", "INT", "SMALLINT", "TINYINT", "BIGINT", "NUMERIC", "BIGNUMERIC"},
    {"FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC"},
    {"DATE", "DATETIME", "TIMESTAMP"},
]


def _types_compatible(t1: str, t2: str) -> bool:
    """Check if two BigQuery data types are compatible for joining."""
    t1 = t1.upper().strip()
    t2 = t2.upper().strip()
    if t1 == t2:
        return True
    for group in _COMPATIBLE_TYPE_GROUPS:
        if t1 in group and t2 in group:
            return True
    return False


def _extract_term_codes(col_data: dict) -> set[str]:
    """Extract terminology binding keys (system|code) from a semantic column."""
    codes: set[str] = set()
    for tb in col_data.get("terminology_bindings", []):
        if isinstance(tb, dict):
            system = tb.get("system", "")
            code = tb.get("code", "")
            if system and code:
                codes.add(f"{system}|{code}")
    # Also check value_set_binding for older profile format
    for vsb in col_data.get("value_set_binding", []):
        if isinstance(vsb, dict):
            system = vsb.get("system", "")
            code = vsb.get("code", "")
            if system and code:
                codes.add(f"{system}|{code}")
    return codes


def infer_joins(
    profiles: list[dict],
    tech_profiles: list[dict],
) -> dict[str, list[dict]]:
    """
    Given semantic + tech profiles for all tables in a dataset,
    infer cross-table join paths.

    Args:
        profiles: List of semantic profile dicts (each with "table", "columns", etc.)
        tech_profiles: List of technical profile dicts (each with "table", "columns")

    Returns:
        {fq_table: [{"source_column": str, "target": "table.column", "confidence": "high"|"medium"|"low"}]}
    """
    # Build indexes: column name -> list of (fq_table, col_name, data_type, term_codes)
    col_index: dict[str, list[tuple[str, str, str, set[str]]]] = {}
    # term_code -> list of (fq_table, col_name)
    term_index: dict[str, list[tuple[str, str]]] = {}

    # Build tech type lookup: fq_table -> col_name -> data_type
    tech_types: dict[str, dict[str, str]] = {}
    for tp in tech_profiles:
        table_name = tp.get("table", "")
        if not table_name:
            continue
        types: dict[str, str] = {}
        for col in tp.get("columns", []):
            cname = col.get("name", col.get("column_name", ""))
            dtype = col.get("data_type", "STRING")
            if cname:
                types[cname] = dtype
        tech_types[table_name] = types

    # Build column and terminology indexes from semantic profiles
    for sp in profiles:
        table_name = sp.get("table", "")
        if not table_name:
            continue
        table_tech = tech_types.get(table_name, {})
        for col in sp.get("columns", []):
            col_name = col.get("name", col.get("column_name", ""))
            if not col_name:
                continue
            data_type = table_tech.get(col_name, "STRING")
            term_codes = _extract_term_codes(col)

            entry = (table_name, col_name, data_type, term_codes)
            col_index.setdefault(col_name, []).append(entry)

            for tc in term_codes:
                term_index.setdefault(tc, []).append((table_name, col_name))

    result: dict[str, list[dict]] = {}

    # Strategy 1: Exact name match across tables
    for col_name, entries in col_index.items():
        if col_name.lower() in _GENERIC_SKIP_NAMES:
            continue
        if len(entries) < 2:
            continue

        for i, (table_a, cname_a, dtype_a, codes_a) in enumerate(entries):
            for j, (table_b, cname_b, dtype_b, codes_b) in enumerate(entries):
                if i >= j:
                    continue
                if table_a == table_b:
                    continue

                compatible = _types_compatible(dtype_a, dtype_b)
                shared_codes = codes_a & codes_b
                has_code_match = bool(shared_codes)

                if not compatible:
                    continue

                if has_code_match:
                    confidence = "high"
                else:
                    confidence = "medium"

                # Add bidirectional join paths
                result.setdefault(table_a, []).append({
                    "source_column": cname_a,
                    "target": f"{table_b}.{cname_b}",
                    "confidence": confidence,
                })
                result.setdefault(table_b, []).append({
                    "source_column": cname_b,
                    "target": f"{table_a}.{cname_a}",
                    "confidence": confidence,
                })

    # Strategy 2: Terminology code match (different column names)
    for term_code, code_entries in term_index.items():
        if len(code_entries) < 2:
            continue
        for i, (table_a, col_a) in enumerate(code_entries):
            for j, (table_b, col_b) in enumerate(code_entries):
                if i >= j:
                    continue
                if table_a == table_b:
                    continue
                if col_a == col_b:
                    # Already handled by name-match strategy above
                    continue

                target_key_ab = f"{table_b}.{col_b}"
                target_key_ba = f"{table_a}.{col_a}"

                # Check if this join path already exists from strategy 1
                existing_a = result.get(table_a, [])
                already_exists = any(
                    jp["source_column"] == col_a and jp["target"] == target_key_ab
                    for jp in existing_a
                )
                if already_exists:
                    continue

                result.setdefault(table_a, []).append({
                    "source_column": col_a,
                    "target": target_key_ab,
                    "confidence": "low",
                })
                result.setdefault(table_b, []).append({
                    "source_column": col_b,
                    "target": target_key_ba,
                    "confidence": "low",
                })

    # Deduplicate within each table's join list
    for fq_table in result:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for jp in result[fq_table]:
            key = (jp["source_column"], jp["target"])
            if key not in seen:
                seen.add(key)
                deduped.append(jp)
        result[fq_table] = deduped

    return result
