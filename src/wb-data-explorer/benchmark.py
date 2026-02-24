#!/usr/bin/env python3
"""
Backend Benchmark for WB Data Explorer
=======================================
Runs all 25 benchmarking questions directly against the LangGraph agent
(no Gradio frontend). For each question it captures:

  - Generated SQL
  - Which metadata tables/columns were used to build the query
  - Query result (row count, sample data)
  - Full LLM response text
  - Timing
  - Auto-detected scoring hints

Usage:
    python benchmark.py \
      --project wb-glittery-carrot-8816 \
      --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
      --json-dir gs://metadata-json-wb-shrewd-papaya-8403

    # Or with local metadata for faster iteration:
    python benchmark.py \
      --project wb-glittery-carrot-8816 \
      --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
      --json-dir /path/to/local/json/

    # Dry-run (metadata analysis only, no BQ/LLM calls):
    python benchmark.py --json-dir gs://metadata-json-wb-shrewd-papaya-8403 --dry-run

    # Run a subset of questions:
    python benchmark.py --questions 1 6 11 16 21 ...

Output:
    benchmark_results_<timestamp>.json   — full structured results
    stdout                               — human-readable summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metadata_loader import (
    TableSchema,
    format_schemas_for_prompt,
    load_metadata,
    resolve_against_bigquery,
)

# ── Benchmark Questions ───────────────────────────────────────────────────────

QUESTIONS = [
    # Level 1 — Simple Exploration (5 questions)
    {"id": 1,  "level": 1, "label": "Simple Exploration",       "question": "How many participants are in each study?"},
    {"id": 2,  "level": 1, "label": "Simple Exploration",       "question": "What demographic information do we have?"},
    {"id": 3,  "level": 1, "label": "Simple Exploration",       "question": "Show me the first few rows of the depression survey data"},
    {"id": 4,  "level": 1, "label": "Simple Exploration",       "question": "What kinds of data are available across all studies?"},
    {"id": 5,  "level": 1, "label": "Simple Exploration",       "question": "Are there any participants flagged as having long COVID?"},

    # Level 2 — Clinical Queries (10 questions: original 5 + 5 new targeting untested tables)
    {"id": 6,  "level": 2, "label": "Clinical Queries",         "question": "How many people screened positive for depression?"},
    {"id": 7,  "level": 2, "label": "Clinical Queries",         "question": "What's the average anxiety score across all visits?"},
    {"id": 8,  "level": 2, "label": "Clinical Queries",         "question": "Show the distribution of disability severity"},
    {"id": 9,  "level": 2, "label": "Clinical Queries",         "question": "Do we have lung function measurements? What's the average?"},
    {"id": 10, "level": 2, "label": "Clinical Queries",         "question": "How does self-reported quality of life vary by visit?"},
    # NEW — Untested tables
    {"id": 26, "level": 2, "label": "Clinical Queries",         "question": "How many PRESCO participants are progressors versus non-progressors?"},
    {"id": 27, "level": 2, "label": "Clinical Queries",         "question": "What is the average cardiovascular risk score?"},
    {"id": 28, "level": 2, "label": "Clinical Queries",         "question": "Show me alcohol screening results"},
    {"id": 29, "level": 2, "label": "Clinical Queries",         "question": "What are the demographics broken down by sex and race?"},
    {"id": 30, "level": 2, "label": "Clinical Queries",         "question": "Which participants have lab assay data and what was measured?"},

    # Level 3 — Relationships & Joins (8 questions: original 5 + 3 new PRESCO/cross-domain)
    {"id": 11, "level": 3, "label": "Relationships & Joins",    "question": "Are people with depression also more likely to have anxiety?"},
    {"id": 12, "level": 3, "label": "Relationships & Joins",    "question": "Is there a relationship between lung function and disability?"},
    {"id": 13, "level": 3, "label": "Relationships & Joins",    "question": "Compare depression scores between eligible and ineligible cohort members"},
    {"id": 14, "level": 3, "label": "Relationships & Joins",    "question": "Which participants completed all the mental health questionnaires?"},
    {"id": 15, "level": 3, "label": "Relationships & Joins",    "question": "Do participants with more diagnoses have worse quality of life?"},
    # NEW — PRESCO joins + cross-domain
    {"id": 31, "level": 3, "label": "Relationships & Joins",    "question": "Show immune cell subset frequencies for PASC versus non-PASC participants"},
    {"id": 32, "level": 3, "label": "Relationships & Joins",    "question": "What are the top expressed genes in the PRESCO data?"},
    {"id": 33, "level": 3, "label": "Relationships & Joins",    "question": "Is cardiovascular risk related to disability severity?"},

    # Level 4 — Cross-Study Cohort Building (8 questions: original 5 + 3 new cross-study)
    {"id": 16, "level": 4, "label": "Cross-Study Cohort",       "question": "Build me a combined mental health cohort across both studies"},
    {"id": 17, "level": 4, "label": "Cross-Study Cohort",       "question": "Can we compare demographics between the two study populations?"},
    {"id": 18, "level": 4, "label": "Cross-Study Cohort",       "question": "I need everyone with immune data AND mental health data, regardless of study"},
    {"id": 19, "level": 4, "label": "Cross-Study Cohort",       "question": "Which participants have evidence of both physical and mental health impairment across any study?"},
    {"id": 20, "level": 4, "label": "Cross-Study Cohort",       "question": "What overlapping data domains exist between the studies? Could we do a combined analysis?"},
    # NEW — Cross-study heavy
    {"id": 34, "level": 4, "label": "Cross-Study Cohort",       "question": "Compare disability scores between the two study populations"},
    {"id": 35, "level": 4, "label": "Cross-Study Cohort",       "question": "What biological and clinical data exists across studies for COVID recovery research?"},
    {"id": 36, "level": 4, "label": "Cross-Study Cohort",       "question": "Build a combined dataset with immune markers and mental health data from all available studies"},

    # Level 5 — Ambiguous / Edge Cases (5 questions, unchanged)
    {"id": 21, "level": 5, "label": "Ambiguous / Edge Cases",   "question": "What data do we have on respiratory outcomes?"},
    {"id": 22, "level": 5, "label": "Ambiguous / Edge Cases",   "question": "Can we link participants across the two studies?"},
    {"id": 23, "level": 5, "label": "Ambiguous / Edge Cases",   "question": "I want to study recovery trajectories — what's available?"},
    {"id": 24, "level": 5, "label": "Ambiguous / Edge Cases",   "question": "Which tables should I use if I'm writing a grant about post-COVID disability?"},
    {"id": 25, "level": 5, "label": "Ambiguous / Edge Cases",   "question": "Find me everyone who got worse over time"},
]


# ── Metadata Usage Analysis ──────────────────────────────────────────────────

def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract fully-qualified table names (project.dataset.table) from SQL."""
    if not sql:
        return []
    # Match backtick-quoted table names: `project.dataset.table`
    backtick_tables = re.findall(r"`([^`]+\.[^`]+\.[^`]+)`", sql)
    # Also match unquoted three-part names (project.dataset.table)
    unquoted_tables = re.findall(
        r"(?:FROM|JOIN|from|join)\s+(\w[\w-]*\.\w+\.\w+)", sql
    )
    all_tables = list(dict.fromkeys(backtick_tables + unquoted_tables))  # deduplicate, preserve order
    # Filter out INFORMATION_SCHEMA references
    return [t for t in all_tables if "INFORMATION_SCHEMA" not in t.upper()]


def extract_columns_from_sql(sql: str) -> list[str]:
    """Extract column names referenced in SQL (best-effort)."""
    if not sql:
        return []
    # Remove string literals and comments to avoid false matches
    cleaned = re.sub(r"'[^']*'", "", sql)
    cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
    # Extract identifiers that look like column refs (word characters, not keywords)
    sql_keywords = {
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "NOT", "IN",
        "AS", "IS", "NULL", "TRUE", "FALSE", "CASE", "WHEN", "THEN", "ELSE",
        "END", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "UNION",
        "ALL", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "BETWEEN",
        "LIKE", "EXISTS", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
        "ALTER", "TABLE", "INTO", "VALUES", "SET", "WITH", "LEFT", "RIGHT",
        "INNER", "OUTER", "CROSS", "FULL", "NATURAL", "USING", "ASC", "DESC",
        "IF", "IFNULL", "COALESCE", "CAST", "EXTRACT", "DATE", "TIMESTAMP",
        "INT64", "FLOAT64", "STRING", "BOOL", "STRUCT", "ARRAY", "UNNEST",
        "OVER", "PARTITION", "ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD",
        "FIRST_VALUE", "LAST_VALUE", "NTILE", "PERCENT_RANK", "CUME_DIST",
        "ROWS", "RANGE", "UNBOUNDED", "PRECEDING", "FOLLOWING", "CURRENT",
        "ROW", "RECURSIVE", "EXCEPT", "INTERSECT", "ROLLUP", "CUBE",
        "GROUPING", "SETS", "QUALIFY", "PIVOT", "UNPIVOT", "TABLESAMPLE",
        "SAFE_DIVIDE", "ROUND", "CONCAT", "LENGTH", "UPPER", "LOWER",
        "TRIM", "SUBSTR", "REPLACE", "REGEXP_CONTAINS", "REGEXP_EXTRACT",
        "DATE_DIFF", "DATE_ADD", "DATE_SUB", "FORMAT_DATE", "PARSE_DATE",
        "TIMESTAMP_DIFF", "CURRENT_DATE", "CURRENT_TIMESTAMP",
    }
    # Find words that look like column names (after . or standalone)
    # Pattern: table_alias.column_name or just column_name in SELECT/WHERE
    col_refs = re.findall(r"(?:\w+\.)?(\w+)", cleaned)
    columns = []
    seen = set()
    for c in col_refs:
        upper = c.upper()
        if upper not in sql_keywords and not c.isdigit() and c not in seen:
            seen.add(c)
            columns.append(c)
    return columns


def analyze_metadata_usage(
    sql: str,
    response_text: str,
    schemas: dict[str, TableSchema],
) -> dict:
    """
    Analyze how metadata was used to create a query.

    Returns a dict with:
      - tables_referenced: list of table names found in the SQL
      - tables_from_metadata: which of those are in our metadata catalog
      - columns_used_per_table: for each metadata table, which of its columns appeared in the SQL
      - metadata_columns_available: total columns available in referenced metadata tables
      - metadata_coverage: what fraction of available metadata columns were used
      - join_keys_used: which primary/join keys were used
      - tables_mentioned_in_response: tables mentioned in the LLM's natural language response
    """
    sql_tables = extract_tables_from_sql(sql)
    sql_columns = extract_columns_from_sql(sql)
    sql_columns_upper = {c.upper() for c in sql_columns}

    tables_from_metadata = []
    columns_used_per_table = {}
    metadata_columns_available = {}
    join_keys_used = []

    for tbl_name in sql_tables:
        schema = schemas.get(tbl_name)
        if schema:
            tables_from_metadata.append(tbl_name)
            # Find which columns from this table's schema appear in the SQL
            schema_col_names = [col.name for col in schema.columns]
            used_cols = [c for c in schema_col_names if c.upper() in sql_columns_upper]
            columns_used_per_table[tbl_name] = used_cols
            metadata_columns_available[tbl_name] = schema_col_names

            # Check if primary key was used
            pk = schema.primary_key
            if pk and pk.upper() in sql_columns_upper:
                join_keys_used.append({"table": tbl_name, "key": pk, "role": "primary_key"})

    # Tables mentioned in the natural-language response (not SQL)
    tables_mentioned_in_response = []
    for tbl_name in schemas:
        # Check both the full FQ name and the short table name
        short_name = tbl_name.split(".")[-1] if "." in tbl_name else tbl_name
        if short_name in response_text or tbl_name in response_text:
            tables_mentioned_in_response.append(tbl_name)

    # Compute coverage
    total_available = sum(len(cols) for cols in metadata_columns_available.values())
    total_used = sum(len(cols) for cols in columns_used_per_table.values())
    coverage = round(total_used / total_available, 3) if total_available > 0 else 0.0

    return {
        "tables_in_sql": sql_tables,
        "tables_from_metadata": tables_from_metadata,
        "tables_not_in_metadata": [t for t in sql_tables if t not in schemas],
        "columns_used_per_table": columns_used_per_table,
        "metadata_columns_available": metadata_columns_available,
        "join_keys_used": join_keys_used,
        "tables_mentioned_in_response": tables_mentioned_in_response,
        "total_metadata_columns_available": total_available,
        "total_metadata_columns_used": total_used,
        "metadata_column_coverage": coverage,
    }


# ── Result Structures ─────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Result for a single benchmark question."""
    question_id: int
    level: int
    level_label: str
    question: str
    # Agent output
    response_text: str = ""
    sql: str = ""
    tool_output: str = ""
    # Metadata usage
    metadata_usage: dict = field(default_factory=dict)
    # Query results
    query_success: Optional[bool] = None
    row_count: int = 0
    sample_data: Optional[list[dict]] = None
    error_message: Optional[str] = None
    # Timing
    elapsed_seconds: float = 0.0
    # Scoring hints
    identified_tables: bool = False
    mapped_columns: bool = False
    sql_executed: bool = False
    results_sensible: Optional[bool] = None  # manual scoring
    explained_reasoning: bool = False
    acknowledged_limitations: bool = False


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def _flush_print(*args, **kwargs):
    """Print and immediately flush stdout (important for nohup / background runs)."""
    print(*args, **kwargs)
    sys.stdout.flush()


def _load_completed_ids(output_file: str) -> set[int]:
    """Load question IDs already completed in a previous (partial) results file."""
    try:
        with open(output_file) as f:
            data = json.load(f)
        completed = set()
        for r in data.get("results", []):
            qid = r.get("question_id")
            # Only count as completed if there's a response or an error (i.e. it actually ran)
            if qid is not None and (r.get("sql") or r.get("response_text") or r.get("error_message")):
                completed.add(qid)
        return completed
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return set()


def _load_previous_results(output_file: str) -> list[QuestionResult]:
    """Load QuestionResult objects from a previous (partial) run."""
    try:
        with open(output_file) as f:
            data = json.load(f)
        results = []
        for r in data.get("results", []):
            qr = QuestionResult(
                question_id=r["question_id"],
                level=r["level"],
                level_label=r["level_label"],
                question=r["question"],
                response_text=r.get("response_text", ""),
                sql=r.get("sql", ""),
                tool_output=r.get("tool_output", ""),
                metadata_usage=r.get("metadata_usage", {}),
                query_success=r.get("query_success"),
                row_count=r.get("row_count", 0),
                error_message=r.get("error_message"),
                elapsed_seconds=r.get("elapsed_seconds", 0.0),
            )
            sh = r.get("scoring_hints", {})
            qr.identified_tables = sh.get("identified_relevant_tables", False)
            qr.mapped_columns = sh.get("mapped_clinical_concept_to_columns", False)
            qr.sql_executed = sh.get("sql_executed_without_error", False)
            qr.explained_reasoning = sh.get("explained_reasoning", False)
            qr.acknowledged_limitations = sh.get("acknowledged_limitations", False)
            results.append(qr)
        return results
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return []


def _save_incremental(
    results: list[QuestionResult],
    schemas: dict[str, TableSchema],
    system_prompt: str,
    total_elapsed: float,
    output_file: str,
):
    """Save current results to disk (called after every question for crash-safety)."""
    report = build_report(results, schemas, system_prompt, total_elapsed)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


def run_benchmark(
    schemas: dict[str, TableSchema],
    system_prompt: str,
    compiled_graph,
    questions: list[dict],
    output_file: str,
    resume: bool = False,
):
    """Run all benchmark questions and save results.

    If resume=True and output_file already exists, previously completed
    questions are skipped and their results are merged into the final report.
    Results are saved incrementally after each question so that interrupted
    runs never lose progress.
    """
    from agent import run_agent, _content_to_str

    # ── Resume: load previous results if available ────────────────────────
    results: list[QuestionResult] = []
    completed_ids: set[int] = set()

    if resume:
        completed_ids = _load_completed_ids(output_file)
        if completed_ids:
            results = _load_previous_results(output_file)
            _flush_print(f"\n🔄 Resuming — {len(completed_ids)} questions already completed, "
                         f"skipping IDs: {sorted(completed_ids)}")

    remaining = [q for q in questions if q["id"] not in completed_ids]
    total_start = time.time()

    _flush_print(f"\n{'=' * 80}")
    _flush_print(f"RUNNING BENCHMARK — {len(remaining)} questions "
                 f"({'resuming' if completed_ids else 'fresh run'}, "
                 f"{len(questions)} total)")
    _flush_print(f"{'=' * 80}\n")

    for i, q in enumerate(remaining, 1):
        qid = q["id"]
        level = q["level"]
        label = q["label"]
        question = q["question"]

        _flush_print(f"[{i}/{len(remaining)}] Q{qid} (L{level}: {label})")
        _flush_print(f"  ❓ {question}")

        qr = QuestionResult(
            question_id=qid,
            level=level,
            level_label=label,
            question=question,
        )

        start = time.time()
        try:
            result, _history = run_agent(compiled_graph, question)
            qr.elapsed_seconds = round(time.time() - start, 2)

            qr.response_text = result.get("response", "")
            qr.sql = result.get("sql", "")
            qr.tool_output = result.get("tool_output", "")

            # Parse tool output for query success/failure indicators
            tool_text = qr.tool_output
            if tool_text:
                if "Query successful" in tool_text:
                    qr.query_success = True
                    qr.sql_executed = True
                    # Extract row count
                    rc_match = re.search(r"(\d+)\s*rows?\s*returned", tool_text)
                    if rc_match:
                        qr.row_count = int(rc_match.group(1))
                elif "QUERY FAILED" in tool_text:
                    qr.query_success = False
                    qr.sql_executed = True
                    err_match = re.search(r"QUERY FAILED.*?:\n(.*?)(?:\n\nOriginal SQL|\n\nAttempt)", tool_text, re.DOTALL)
                    if err_match:
                        qr.error_message = err_match.group(1).strip()[:500]

            # Metadata usage analysis
            qr.metadata_usage = analyze_metadata_usage(qr.sql, qr.response_text, schemas)

            # Auto-score hints
            qr.identified_tables = len(qr.metadata_usage.get("tables_from_metadata", [])) > 0
            qr.mapped_columns = qr.metadata_usage.get("total_metadata_columns_used", 0) > 0

            # Check if LLM explained reasoning (look for explanation markers)
            response_lower = qr.response_text.lower()
            qr.explained_reasoning = any(marker in response_lower for marker in [
                "this query", "i used", "i joined", "the query", "this means",
                "i'm looking", "i selected", "this joins", "the reason",
                "logic", "approach", "strategy",
            ])
            qr.acknowledged_limitations = any(marker in response_lower for marker in [
                "limitation", "caveat", "note that", "keep in mind",
                "doesn't include", "not available", "no direct",
                "cannot", "missing", "unfortunately", "however",
                "assumption", "assuming",
            ])

            # Print summary
            status = "✅" if qr.query_success else ("❌" if qr.query_success is False else "ℹ️")
            tables_used = qr.metadata_usage.get("tables_from_metadata", [])
            tables_short = [t.split(".")[-1] for t in tables_used]
            _flush_print(f"  {status} SQL: {'Yes' if qr.sql else 'No'} | "
                         f"Rows: {qr.row_count} | "
                         f"Tables: {', '.join(tables_short) if tables_short else 'none'} | "
                         f"Cols used: {qr.metadata_usage.get('total_metadata_columns_used', 0)} | "
                         f"Time: {qr.elapsed_seconds}s")

        except Exception as e:
            qr.elapsed_seconds = round(time.time() - start, 2)
            qr.error_message = str(e)
            _flush_print(f"  💥 Exception: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

        results.append(qr)
        _flush_print()

        # ── Incremental save after every question ─────────────────────────
        elapsed_so_far = round(time.time() - total_start, 1)
        _save_incremental(results, schemas, system_prompt, elapsed_so_far, output_file)
        _flush_print(f"  💾 Progress saved ({len(results)}/{len(questions)} questions)")

    total_elapsed = round(time.time() - total_start, 1)

    # Final save (updates total timing)
    report = build_report(results, schemas, system_prompt, total_elapsed)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _flush_print(f"\n📄 Full results saved to: {output_file}")

    # Print summary
    print_summary(results, total_elapsed)

    return results


def build_report(
    results: list[QuestionResult],
    schemas: dict[str, TableSchema],
    system_prompt: str,
    total_elapsed: float,
) -> dict:
    """Build the full JSON report."""
    # Metadata catalog summary
    metadata_catalog = {}
    for tbl_name, schema in sorted(schemas.items()):
        metadata_catalog[tbl_name] = {
            "title": schema.title,
            "description": schema.description,
            "primary_key": schema.primary_key,
            "column_count": len(schema.columns),
            "columns": [
                {
                    "name": col.name,
                    "type": col.data_type,
                    "description": col.short_description,
                    "sensitivity": col.sensitivity_label,
                }
                for col in schema.columns
            ],
            "joins_to": [
                link.target_table_name or link.target_profile_url
                for link in schema.join_links
            ],
        }

    # Per-question results
    question_results = []
    for qr in results:
        entry = {
            "question_id": qr.question_id,
            "level": qr.level,
            "level_label": qr.level_label,
            "question": qr.question,
            "elapsed_seconds": qr.elapsed_seconds,
            # Generated output
            "sql": qr.sql,
            "response_text": qr.response_text,
            # Metadata usage — the key section
            "metadata_usage": qr.metadata_usage,
            # Query outcome
            "query_success": qr.query_success,
            "row_count": qr.row_count,
            "error_message": qr.error_message,
            # Auto-scoring hints
            "scoring_hints": {
                "identified_relevant_tables": qr.identified_tables,
                "mapped_clinical_concept_to_columns": qr.mapped_columns,
                "sql_executed_without_error": qr.query_success is True,
                "explained_reasoning": qr.explained_reasoning,
                "acknowledged_limitations": qr.acknowledged_limitations,
                "results_clinically_sensible": None,  # requires manual review
            },
        }
        question_results.append(entry)

    return {
        "benchmark_run": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "total_elapsed_seconds": total_elapsed,
            "system_prompt_length": len(system_prompt),
        },
        "metadata_catalog": metadata_catalog,
        "results": question_results,
    }


def print_summary(results: list[QuestionResult], total_elapsed: float):
    """Print a human-readable summary table."""
    print(f"\n{'=' * 100}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 100}")

    # Header
    print(f"{'Q#':<4} {'Lvl':<4} {'Question':<60} {'SQL':>4} {'OK':>4} {'Rows':>6} {'Tbls':>5} {'Cols':>5} {'Time':>6}")
    print("-" * 100)

    by_level = {}
    total_sql = 0
    total_ok = 0
    total_tables_id = 0
    total_cols_mapped = 0

    for qr in results:
        has_sql = "✓" if qr.sql else "✗"
        ok = "✓" if qr.query_success else ("✗" if qr.query_success is False else "—")
        tbls = qr.metadata_usage.get("total_metadata_columns_available", 0) and len(qr.metadata_usage.get("tables_from_metadata", []))
        cols = qr.metadata_usage.get("total_metadata_columns_used", 0)
        q_short = qr.question[:58] + ".." if len(qr.question) > 60 else qr.question

        print(f"Q{qr.question_id:<3} L{qr.level:<3} {q_short:<60} {has_sql:>4} {ok:>4} {qr.row_count:>6} {tbls:>5} {cols:>5} {qr.elapsed_seconds:>5.1f}s")

        # Aggregate by level
        if qr.level not in by_level:
            by_level[qr.level] = {"total": 0, "sql": 0, "ok": 0, "tables": 0, "cols": 0}
        by_level[qr.level]["total"] += 1
        if qr.sql:
            by_level[qr.level]["sql"] += 1
            total_sql += 1
        if qr.query_success:
            by_level[qr.level]["ok"] += 1
            total_ok += 1
        if qr.identified_tables:
            by_level[qr.level]["tables"] += 1
            total_tables_id += 1
        if qr.mapped_columns:
            by_level[qr.level]["cols"] += 1
            total_cols_mapped += 1

    print("-" * 100)

    # Level summary
    print(f"\n{'Level':<35} {'SQL Gen':>8} {'Exec OK':>8} {'Tbls ID':>8} {'Cols Map':>8}")
    print("-" * 70)
    level_labels = {1: "D1: Simple Exploration", 2: "D2: Clinical Queries",
                    3: "D3: Relationships & Joins", 4: "D4: Cross-Study Cohort",
                    5: "D5: Ambiguous / Edge Cases"}
    for lvl in sorted(by_level):
        d = by_level[lvl]
        print(f"{level_labels[lvl]:<35} {d['sql']}/{d['total']:>6} {d['ok']}/{d['total']:>6} {d['tables']}/{d['total']:>6} {d['cols']}/{d['total']:>6}")

    print("-" * 70)
    n = len(results)
    print(f"{'TOTAL':<35} {total_sql}/{n:>6} {total_ok}/{n:>6} {total_tables_id}/{n:>6} {total_cols_mapped}/{n:>6}")
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Metadata usage highlights
    print(f"\n{'=' * 100}")
    print("METADATA USAGE PER QUESTION")
    print(f"{'=' * 100}")
    for qr in results:
        mu = qr.metadata_usage
        tables = mu.get("tables_from_metadata", [])
        cols_per_table = mu.get("columns_used_per_table", {})

        print(f"\nQ{qr.question_id} (L{qr.level}): {qr.question}")
        if not tables:
            if qr.sql:
                unrec = mu.get("tables_not_in_metadata", [])
                print(f"  Tables in SQL (not in metadata): {unrec}")
            else:
                print(f"  No SQL generated — LLM answered from metadata context only")
        else:
            for tbl in tables:
                short = tbl.split(".")[-1]
                used = cols_per_table.get(tbl, [])
                avail = mu.get("metadata_columns_available", {}).get(tbl, [])
                print(f"  📊 {tbl}")
                print(f"     Columns used ({len(used)}/{len(avail)}): {', '.join(used) if used else '—'}")
        jk = mu.get("join_keys_used", [])
        if jk:
            keys_str = ", ".join(f"{j['key']}({j['table'].split('.')[-1]})" for j in jk)
            print(f"  🔗 Join keys: {keys_str}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Backend benchmark for WB Data Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GCP_PROJECT_ID"),
        help="GCP project for Vertex AI / BQ billing (default: $GCP_PROJECT_ID)",
    )
    parser.add_argument(
        "--data-project",
        type=str,
        nargs="+",
        default=None,
        help="BQ data project(s), space-separated (default: same as --project)",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=os.environ.get("METADATA_SOURCE", "gs://metadata-json-wb-shrewd-papaya-8403"),
        help="Path or gs:// URI for FHIR metadata JSONs",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-pro",
        help="Vertex AI model name",
    )
    parser.add_argument(
        "--questions",
        type=int,
        nargs="+",
        default=None,
        help="Run only these question IDs (e.g. --questions 1 6 11 16 21)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: benchmark_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load metadata and show what would run, but skip LLM/BQ calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run — skip questions already completed in the output file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine output file — defaults into benchmark_results/ subfolder
    results_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    if args.output:
        output_file = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(results_dir / f"benchmark_results_{ts}.json")

    # Filter questions if subset requested
    questions = QUESTIONS
    if args.questions:
        questions = [q for q in QUESTIONS if q["id"] in args.questions]
        if not questions:
            print(f"❌ No matching questions for IDs: {args.questions}")
            sys.exit(1)

    print("=" * 80)
    print("WB DATA EXPLORER — BACKEND BENCHMARK")
    print("=" * 80)
    print(f"  Metadata source:  {args.json_dir}")
    print(f"  Billing project:  {args.project or '(none — dry run?)'}")
    print(f"  Data project(s):  {args.data_project or '(same as billing)'}")
    print(f"  LLM model:        {args.llm_model}")
    print(f"  Questions:        {len(questions)} of {len(QUESTIONS)}")
    print(f"  Output:           {output_file}")
    print(f"  Dry run:          {args.dry_run}")
    print(f"  Resume:           {args.resume}")
    print()

    # ── Step 1: Load metadata ─────────────────────────────────────────────
    print("─── Loading Metadata ───")
    schemas = load_metadata(args.json_dir)
    print(f"Loaded {len(schemas)} table schemas from FHIR JSONs")
    for name in sorted(schemas.keys()):
        s = schemas[name]
        print(f"  • {name} ({len(s.columns)} columns, PK={s.primary_key})")

    # ── Step 2: Resolve against BigQuery ──────────────────────────────────
    data_projects = args.data_project or ([args.project] if args.project else [])
    if data_projects:
        print(f"\n─── Resolving Against BigQuery ({data_projects}) ───")
        schemas = resolve_against_bigquery(schemas, data_projects)
        print(f"✅ {len(schemas)} tables resolved and ready")
    else:
        print("\n⚠️  No data projects specified — skipping BQ resolution")

    # ── Step 3: Build system prompt ───────────────────────────────────────
    from prompt_engine import build_system_prompt
    system_prompt = build_system_prompt(schemas)
    print(f"\nSystem prompt: {len(system_prompt)} chars, {system_prompt.count(chr(10))} lines")

    # ── Dry run: just show metadata + questions ───────────────────────────
    if args.dry_run:
        print(f"\n{'=' * 80}")
        print("DRY RUN — Questions that would be run:")
        print(f"{'=' * 80}")
        for q in questions:
            print(f"  Q{q['id']:>2} (L{q['level']}): {q['question']}")

        print(f"\nMetadata tables available for the LLM:")
        for tbl_name, schema in sorted(schemas.items()):
            col_names = [c.name for c in schema.columns]
            print(f"  {tbl_name}")
            print(f"    PK: {schema.primary_key} | Cols: {', '.join(col_names)}")

        print(f"\n✅ Dry run complete. Remove --dry-run to execute.")
        return

    # ── Step 4: Create LangGraph agent ────────────────────────────────────
    if not args.project:
        print("❌ --project is required for non-dry-run benchmark (need Vertex AI + BQ billing)")
        sys.exit(1)

    print(f"\n─── Creating LangGraph Agent ───")
    from agent import create_agent
    compiled_graph, resolved_schemas, agent_prompt = create_agent(
        json_metadata_dir=args.json_dir,
        bq_project_id=args.project,
        data_project_id=data_projects,
        llm_model=args.llm_model,
    )
    # Use the agent's resolved schemas (they may differ slightly from ours if create_agent reloads)
    schemas = resolved_schemas
    system_prompt = agent_prompt
    print(f"✅ Agent ready ({len(schemas)} tables)")

    # ── Step 5: Run benchmark ─────────────────────────────────────────────
    run_benchmark(
        schemas=schemas,
        system_prompt=system_prompt,
        compiled_graph=compiled_graph,
        questions=questions,
        output_file=output_file,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
