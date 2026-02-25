"""
Step 3: BigQuery Executor with Error Recovery
Executes SQL queries against BigQuery, catches errors, asks the LLM to fix
them, and retries — up to a configurable number of attempts.

Also handles large result sets via pagination and CSV export.

Usage:
    from bq_executor import BigQueryExecutor

    executor = BigQueryExecutor(project_id="my-project")
    result = executor.execute_with_retry(sql, schemas, max_retries=3)
    print(result.dataframe)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class QueryResult:
    """Result of a BigQuery query execution."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    sql: str = ""
    error_message: Optional[str] = None
    row_count: int = 0
    total_bytes_processed: Optional[int] = None
    attempts: int = 1
    attempt_history: list[dict] = field(default_factory=list)

    @property
    def is_large(self) -> bool:
        """True if result has more than 1000 rows."""
        return self.row_count > 1000

    @property
    def bytes_processed_mb(self) -> Optional[float]:
        if self.total_bytes_processed is not None:
            return round(self.total_bytes_processed / (1024 * 1024), 2)
        return None


class BigQueryExecutor:
    """
    Executes BigQuery SQL queries with automatic error recovery via LLM.

    In Workbench cloud apps, authentication is handled automatically via ADC.
    For local testing, run `gcloud auth application-default login` first.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        billing_project_id: Optional[str] = None,
        max_results_display: int = 500,
    ):
        """
        Args:
            project_id: GCP project ID where data lives. If None, uses ADC default.
            billing_project_id: GCP project ID for running BQ jobs (must have
                bigquery.jobs.create permission). If None, defaults to project_id.
                This is needed when data lives in a different project (e.g.
                Workbench Data Collection in a separate project).
            max_results_display: Max rows to show in UI (full results still available).
        """
        from google.cloud import bigquery

        self.project_id = project_id
        self.billing_project_id = billing_project_id or project_id
        self.max_results_display = max_results_display

        # BQ client uses the BILLING project for job execution.
        # Do NOT set location — let BQ auto-detect from the dataset location.
        # This is critical when data spans multiple regions (e.g. US and us-central1).
        self.client = bigquery.Client(
            project=self.billing_project_id,
        )

    def execute_query(self, sql: str) -> QueryResult:
        """
        Execute a single SQL query against BigQuery.

        Returns:
            QueryResult with success status, dataframe (if successful), or error message.
        """
        from google.cloud import bigquery

        try:
            # Configure job
            job_config = bigquery.QueryJobConfig(
                use_legacy_sql=False,
                # Dry run first to check for errors and get bytes estimate
            )

            # First: dry run to catch syntax errors cheaply
            dry_config = bigquery.QueryJobConfig(
                use_legacy_sql=False,
                dry_run=True,
            )
            dry_job = self.client.query(sql, job_config=dry_config)
            bytes_estimate = dry_job.total_bytes_processed

            # Actual query execution
            query_job = self.client.query(sql, job_config=job_config)
            result = query_job.result()  # Waits for completion

            # Convert to DataFrame
            df = result.to_dataframe()

            return QueryResult(
                success=True,
                dataframe=df,
                sql=sql,
                row_count=len(df),
                total_bytes_processed=bytes_estimate,
            )

        except Exception as e:
            error_msg = str(e)
            # Clean up common BQ error formatting
            error_msg = _clean_bq_error(error_msg)

            return QueryResult(
                success=False,
                sql=sql,
                error_message=error_msg,
            )

    def execute_with_retry(
        self,
        sql: str,
        schemas: dict,
        max_retries: int = 3,
        llm_project_id: Optional[str] = None,
        llm_model: str = "gemini-2.5-pro",
    ) -> QueryResult:
        """
        Execute SQL with automatic LLM-powered error recovery.

        If a query fails, sends the error + original SQL + schema context to
        the LLM to generate a fixed query, then retries.

        Args:
            sql: The SQL query to execute.
            schemas: Table schemas from metadata_loader (for LLM context).
            max_retries: Max number of retry attempts.
            llm_project_id: GCP project for Gemini calls. Defaults to self.project_id.
            llm_model: Gemini model to use for fixing queries.

        Returns:
            QueryResult with the final outcome and full attempt history.
        """
        from prompt_engine import build_error_fix_prompt, call_gemini, extract_sql_from_response

        llm_project = llm_project_id or self.project_id
        current_sql = sql
        history = []

        for attempt in range(1, max_retries + 2):  # +2 because first attempt isn't a retry
            result = self.execute_query(current_sql)

            history.append({
                "attempt": attempt,
                "sql": current_sql,
                "success": result.success,
                "error": result.error_message,
            })

            if result.success:
                result.attempts = attempt
                result.attempt_history = history
                return result

            # If last attempt, give up
            if attempt > max_retries:
                result.attempts = attempt
                result.attempt_history = history
                return result

            # Ask LLM to fix the query
            try:
                fix_prompt = build_error_fix_prompt(
                    original_query=current_sql,
                    error_message=result.error_message or "Unknown error",
                    schemas=schemas,
                )
                fix_response = call_gemini(
                    system_prompt="You are a BigQuery SQL expert. Fix the query error.",
                    user_message=fix_prompt,
                    model_name=llm_model,
                    project_id=llm_project,
                )
                fixed_sql = extract_sql_from_response(fix_response)
                if fixed_sql:
                    current_sql = fixed_sql
                else:
                    # LLM didn't return parseable SQL — give up
                    result.attempts = attempt
                    result.attempt_history = history
                    result.error_message = (
                        f"Original error: {result.error_message}\n"
                        f"LLM fix attempt failed: could not parse SQL from response"
                    )
                    return result
            except Exception as llm_err:
                result.attempts = attempt
                result.attempt_history = history
                result.error_message = (
                    f"Original error: {result.error_message}\n"
                    f"LLM fix call failed: {str(llm_err)}"
                )
                return result

        # Should not reach here, but just in case
        return result

    def get_table_info(self, dataset_table: str) -> Optional[pd.DataFrame]:
        """
        Query INFORMATION_SCHEMA to get live column metadata for a table.

        Args:
            dataset_table: Table reference in format "dataset.table" or "project.dataset.table"

        Returns:
            DataFrame with column_name, data_type, is_nullable, description
        """
        parts = dataset_table.replace("`", "").split(".")
        if len(parts) == 3:
            project, dataset, table = parts
        elif len(parts) == 2:
            project = self.project_id
            dataset, table = parts
        else:
            return None

        sql = f"""
        SELECT
            column_name,
            data_type,
            is_nullable,
            description
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
        """

        try:
            return self.client.query(sql).result().to_dataframe()
        except Exception:
            # Fallback to simpler INFORMATION_SCHEMA view
            sql_fallback = f"""
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
            """
            try:
                return self.client.query(sql_fallback).result().to_dataframe()
            except Exception:
                return None

    def list_tables(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        List all tables in a BigQuery dataset.

        Args:
            dataset: Dataset reference in format "dataset" or "project.dataset"
        """
        parts = dataset.replace("`", "").split(".")
        if len(parts) == 2:
            project, ds = parts
        elif len(parts) == 1:
            project = self.project_id
            ds = parts[0]
        else:
            return None

        sql = f"""
        SELECT
            table_name,
            table_type,
            row_count,
            size_bytes,
            creation_time,
            last_modified_time
        FROM `{project}.{ds}.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name
        """

        try:
            return self.client.query(sql).result().to_dataframe()
        except Exception as e:
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_bq_error(error_msg: str) -> str:
    """Clean up BigQuery error messages for LLM consumption."""
    # Remove HTTP status codes and trace IDs
    error_msg = re.sub(r"\d{3}\s+", "", error_msg, count=1)
    # Truncate very long errors
    if len(error_msg) > 1000:
        error_msg = error_msg[:1000] + "... (truncated)"
    return error_msg.strip()


def dataframe_to_display(
    df: pd.DataFrame,
    max_rows: int = 100,
    max_col_width: int = 50,
) -> str:
    """
    Format a DataFrame for text display (for non-Gradio contexts).
    Truncates long values and limits rows.
    """
    if df is None or df.empty:
        return "(No results)"

    display_df = df.head(max_rows).copy()

    # Truncate long string values
    for col in display_df.select_dtypes(include=["object"]).columns:
        display_df[col] = display_df[col].apply(
            lambda x: str(x)[:max_col_width] + "..." if isinstance(x, str) and len(str(x)) > max_col_width else x
        )

    result = display_df.to_string(index=False)

    if len(df) > max_rows:
        result += f"\n\n... showing {max_rows} of {len(df)} rows"

    return result


# ── Main (for testing) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("BigQuery Executor — Step 3")
    print()
    print("This module requires a GCP project with BigQuery access.")
    print("To test:")
    print()
    print("  from bq_executor import BigQueryExecutor")
    print("  executor = BigQueryExecutor(project_id='your-project-id')")
    print("  result = executor.execute_query('SELECT 1 as test')")
    print("  print(result.dataframe)")
    print()
    print("For error retry testing:")
    print("  from metadata_loader import load_metadata_from_json_dir")
    print("  schemas = load_metadata_from_json_dir('/path/to/json/')")
    print("  result = executor.execute_with_retry(bad_sql, schemas)")

