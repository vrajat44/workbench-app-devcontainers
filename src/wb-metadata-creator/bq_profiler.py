"""
BigQuery schema introspection and data profiling.

Handles:
  - Dataset / table discovery from INFORMATION_SCHEMA
  - Data profiling: NULL counts, distinct counts, top values, string/numeric stats
"""

from __future__ import annotations

from typing import Optional

from models import BQColumnInfo, BQTableInfo, ColumnProfile, TableProfile


# BQ types that are numeric (for choosing numeric vs string profile metrics)
_NUMERIC_BQ_TYPES = {"INT64", "INTEGER", "FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC"}


# ── Dataset & Table Discovery ─────────────────────────────────────────────────

def discover_bq_datasets(
    project_id: str,
    billing_project_id: Optional[str] = None,
) -> list[str]:
    """List all datasets in a BigQuery project."""
    from google.cloud import bigquery

    client = bigquery.Client(project=billing_project_id or project_id)
    try:
        datasets = list(client.list_datasets(project_id))
        return sorted([ds.dataset_id for ds in datasets])
    except Exception as e:
        print(f"⚠ Could not list datasets in {project_id}: {e}")
        return []


def discover_bq_tables(
    project_id: str,
    dataset_id: str,
    billing_project_id: Optional[str] = None,
) -> list[BQTableInfo]:
    """
    Discover all tables in a BigQuery dataset with column details.
    Queries INFORMATION_SCHEMA for complete schema information.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=billing_project_id or project_id)
    tables = []

    # Get table list with metadata
    try:
        tables_sql = f"""
        SELECT table_name, table_type, row_count, size_bytes
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name
        """
        table_rows = client.query(tables_sql).result()
        table_meta = {
            row.table_name: {
                "table_type": row.table_type,
                "row_count": row.row_count,
                "size_bytes": row.size_bytes,
            }
            for row in table_rows
        }
    except Exception as e:
        print(f"⚠ Could not query INFORMATION_SCHEMA.TABLES: {e}")
        table_meta = {}

    # Get column details for all tables
    try:
        cols_sql = f"""
        SELECT table_name, column_name, data_type, is_nullable, description, ordinal_position
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
        ORDER BY table_name, ordinal_position
        """
        col_rows = list(client.query(cols_sql).result())
    except Exception:
        # Fallback to simpler view
        try:
            cols_sql = f"""
            SELECT table_name, column_name, data_type, is_nullable, ordinal_position
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            ORDER BY table_name, ordinal_position
            """
            col_rows = list(client.query(cols_sql).result())
        except Exception as e:
            print(f"⚠ Could not query INFORMATION_SCHEMA.COLUMNS: {e}")
            col_rows = []

    # Group columns by table
    table_columns: dict[str, list[BQColumnInfo]] = {}
    for row in col_rows:
        col = BQColumnInfo(
            column_name=row.column_name,
            data_type=row.data_type,
            is_nullable=getattr(row, "is_nullable", "YES"),
            description=getattr(row, "description", None),
            ordinal_position=getattr(row, "ordinal_position", 0),
        )
        table_columns.setdefault(row.table_name, []).append(col)

    # Build table info objects
    all_table_names = set(table_meta.keys()) | set(table_columns.keys())
    for table_name in sorted(all_table_names):
        meta = table_meta.get(table_name, {})
        info = BQTableInfo(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_name,
            columns=table_columns.get(table_name, []),
            row_count=meta.get("row_count"),
            size_bytes=meta.get("size_bytes"),
            table_type=meta.get("table_type", "BASE TABLE"),
        )
        tables.append(info)

    return tables


def format_bq_schema_for_prompt(table_info: BQTableInfo) -> str:
    """Format BQ table schema information for inclusion in a generation prompt."""
    lines = [
        f"Table: {table_info.fq_name}",
        f"Table Type: {table_info.table_type}",
    ]
    if table_info.row_count is not None:
        lines.append(f"Row Count: {table_info.row_count:,}")
    if table_info.size_bytes is not None:
        size_mb = table_info.size_bytes / (1024 * 1024)
        lines.append(f"Size: {size_mb:.1f} MB")

    lines.append(f"\nColumns ({len(table_info.columns)}):")
    for col in table_info.columns:
        nullable = "NULLABLE" if col.is_nullable == "YES" else "REQUIRED"
        desc_part = f'  -- "{col.description}"' if col.description else ""
        lines.append(f"  {col.column_name}: {col.data_type} ({nullable}){desc_part}")

    return "\n".join(lines)


# ── Data Profiling ────────────────────────────────────────────────────────────

def profile_bq_table(
    table_info: BQTableInfo,
    billing_project_id: Optional[str] = None,
) -> TableProfile:
    """
    Profile a BQ table: NULL counts, distinct counts, and top values for coded columns.

    Runs four phases:
      1. One aggregate query for null/distinct counts (all columns).
      2. Follow-up queries for top values on low-cardinality columns.
      3. String length stats for STRING columns.
      4. Numeric stats (min/max/stddev/median) for numeric columns.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google.cloud import bigquery

    start = time.time()
    client = bigquery.Client(project=billing_project_id or table_info.project_id)
    fq_table = f"`{table_info.project_id}.{table_info.dataset_id}.{table_info.table_id}`"

    columns = table_info.columns
    profile = TableProfile(table_name=table_info.fq_name)

    # ── Phase 1: NULL counts + distinct counts in one scan ──
    parts = ["COUNT(*) AS total_rows"]
    for i, col in enumerate(columns):
        cn = col.column_name.replace("`", "")
        parts.append(f"COUNTIF(`{cn}` IS NULL) AS n_{i}")
        parts.append(f"APPROX_COUNT_DISTINCT(`{cn}`) AS d_{i}")

    sql = f"SELECT {', '.join(parts)} FROM {fq_table}"

    try:
        row = next(iter(client.query(sql).result()))
        profile.total_rows = row.total_rows or 0

        for i, col in enumerate(columns):
            nulls = getattr(row, f"n_{i}", 0) or 0
            distinct = getattr(row, f"d_{i}", 0) or 0
            pct = round(100.0 * nulls / profile.total_rows, 1) if profile.total_rows > 0 else 0.0
            profile.columns[col.column_name] = ColumnProfile(
                column_name=col.column_name,
                null_count=nulls,
                null_percent=pct,
                distinct_count=distinct,
            )
    except Exception as e:
        print(f"⚠ Profiling query failed: {e}")
        for col in columns:
            profile.columns[col.column_name] = ColumnProfile(column_name=col.column_name)
        return profile

    # ── Phase 2: Top values for low-cardinality columns (parallel) ──
    coded_cols = [
        col for col in columns
        if 1 < profile.columns[col.column_name].distinct_count <= 50
    ]

    def _fetch_top_values(col_info: BQColumnInfo) -> tuple[str, list[str], dict[str, int]]:
        cn = col_info.column_name.replace("`", "")
        q = f"""
        SELECT CAST(`{cn}` AS STRING) AS val, COUNT(*) AS cnt
        FROM {fq_table}
        WHERE `{cn}` IS NOT NULL
        GROUP BY 1 ORDER BY cnt DESC LIMIT 25
        """
        try:
            rows = list(client.query(q).result())
            values = [r.val for r in rows if r.val]
            counts = {r.val: r.cnt for r in rows if r.val}
            return col_info.column_name, values, counts
        except Exception:
            return col_info.column_name, [], {}

    if coded_cols:
        with ThreadPoolExecutor(max_workers=min(8, len(coded_cols))) as ex:
            futures = {ex.submit(_fetch_top_values, c): c for c in coded_cols}
            for f in as_completed(futures):
                cn, vals, val_counts = f.result()
                profile.columns[cn].top_values = vals
                profile.columns[cn].value_counts = val_counts

    # ── Phase 3: String length stats for STRING columns ──
    string_cols = [
        col for col in columns
        if col.data_type.upper().split("<")[0].strip() in ("STRING", "BYTES")
    ]
    if string_cols:
        len_parts = []
        for i, col in enumerate(string_cols):
            cn = col.column_name.replace("`", "")
            len_parts.append(f"MIN(LENGTH(`{cn}`)) AS smin_{i}")
            len_parts.append(f"MAX(LENGTH(`{cn}`)) AS smax_{i}")
            len_parts.append(f"AVG(LENGTH(`{cn}`)) AS savg_{i}")

        len_sql = f"SELECT {', '.join(len_parts)} FROM {fq_table}"
        try:
            row = next(iter(client.query(len_sql).result()))
            for i, col in enumerate(string_cols):
                cp = profile.columns.get(col.column_name)
                if cp:
                    cp.min_length = getattr(row, f"smin_{i}", None)
                    cp.max_length = getattr(row, f"smax_{i}", None)
                    avg = getattr(row, f"savg_{i}", None)
                    cp.avg_length = round(float(avg), 1) if avg is not None else None
        except Exception as e:
            print(f"  ⚠ String length profiling failed: {e}")

    # ── Phase 4: Numeric stats for numeric columns ──
    numeric_cols = [
        col for col in columns
        if col.data_type.upper().split("<")[0].strip() in _NUMERIC_BQ_TYPES
    ]
    if numeric_cols:
        num_parts = []
        for i, col in enumerate(numeric_cols):
            cn = col.column_name.replace("`", "")
            num_parts.append(f"MIN(`{cn}`) AS nmin_{i}")
            num_parts.append(f"MAX(`{cn}`) AS nmax_{i}")
            num_parts.append(f"STDDEV(`{cn}`) AS nstd_{i}")

        num_sql = f"SELECT {', '.join(num_parts)} FROM {fq_table}"
        try:
            row = next(iter(client.query(num_sql).result()))
            for i, col in enumerate(numeric_cols):
                cp = profile.columns.get(col.column_name)
                if cp:
                    nmin = getattr(row, f"nmin_{i}", None)
                    nmax = getattr(row, f"nmax_{i}", None)
                    nstd = getattr(row, f"nstd_{i}", None)
                    cp.min_value = float(nmin) if nmin is not None else None
                    cp.max_value = float(nmax) if nmax is not None else None
                    cp.stddev = round(float(nstd), 4) if nstd is not None else None
        except Exception as e:
            print(f"  ⚠ Numeric stats profiling failed: {e}")

        # Median requires PERCENTILE_CONT — separate per column to handle NULLs
        def _fetch_median(col_info: BQColumnInfo) -> tuple[str, Optional[float]]:
            cn = col_info.column_name.replace("`", "")
            q = f"""
            SELECT APPROX_QUANTILES(`{cn}`, 2)[OFFSET(1)] AS median_val
            FROM {fq_table}
            WHERE `{cn}` IS NOT NULL
            """
            try:
                row = next(iter(client.query(q).result()))
                return col_info.column_name, float(row.median_val) if row.median_val is not None else None
            except Exception:
                return col_info.column_name, None

        with ThreadPoolExecutor(max_workers=min(8, len(numeric_cols))) as ex:
            futures = {ex.submit(_fetch_median, c): c for c in numeric_cols}
            for f in as_completed(futures):
                cn, med = f.result()
                if cn in profile.columns and med is not None:
                    profile.columns[cn].median = med

    elapsed = time.time() - start
    print(f"  📊 Profiled {table_info.fq_name}: {profile.total_rows:,} rows, "
          f"{len(columns)} cols, {len(coded_cols)} coded, "
          f"{len(string_cols)} string, {len(numeric_cols)} numeric — {elapsed:.1f}s")
    return profile
