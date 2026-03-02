"""
PyAirbyte Connector Wrapper for WB Data Loader.

Uses PyAirbyte (https://github.com/airbytehq/pyairbyte) to provide:
  - Dynamic connector discovery  (600+ pre-built sources)
  - Stream listing & selection
  - Data preview (get_records)
  - Full read into BigQuery via BigQueryCache

All Airbyte connector install/config happens automatically via PyAirbyte.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ConnectorInfo:
    """Metadata for an available Airbyte connector."""
    name: str              # e.g. "source-s3", "source-gcs"
    display_name: str      # e.g. "Amazon S3", "Google Cloud Storage"
    language: str          # "python", "manifest-only"
    version: str = ""


@dataclass
class StreamInfo:
    """Metadata for a stream within a source."""
    name: str
    json_schema: dict = field(default_factory=dict)


@dataclass
class PreviewResult:
    """Result of previewing data from a source stream."""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    row_count: int = 0
    column_count: int = 0
    stream_name: str = ""
    error: Optional[str] = None
    schema_info: list[dict] = field(default_factory=list)


@dataclass
class SyncResult:
    """Result of a full sync operation."""
    success: bool
    streams_synced: list[str] = field(default_factory=list)
    total_records: int = 0
    elapsed_seconds: float = 0
    destination: str = ""
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# ── Curated Source List ──────────────────────────────────────────────────────
# We surface the most useful data-ingestion connectors by default.
# Users can also type any connector name from the full Airbyte registry.

CURATED_SOURCES: dict[str, dict[str, str]] = {
    "source-s3": {
        "display": "Amazon S3",
        "description": "Read CSV, Parquet, JSON, JSONL, Avro from S3 buckets",
    },
    "source-gcs": {
        "display": "Google Cloud Storage",
        "description": "Read CSV, Parquet, JSON, JSONL, Avro from GCS buckets",
    },
    "source-bigquery": {
        "display": "BigQuery (source)",
        "description": "Read tables/views from a BigQuery dataset",
    },
    "source-postgres": {
        "display": "PostgreSQL",
        "description": "Read tables from a PostgreSQL database",
    },
    "source-mysql": {
        "display": "MySQL",
        "description": "Read tables from a MySQL database",
    },
    "source-snowflake": {
        "display": "Snowflake",
        "description": "Read tables from Snowflake",
    },
    "source-google-sheets": {
        "display": "Google Sheets",
        "description": "Read data from Google Sheets spreadsheets",
    },
    "source-file": {
        "display": "File (CSV, JSON, Excel, ...)",
        "description": "Read a single file from URL, S3, GCS, HTTPS, or SFTP",
    },
    "source-faker": {
        "display": "Faker (sample data)",
        "description": "Generate fake sample data for testing",
    },
    "source-github": {
        "display": "GitHub",
        "description": "Read issues, PRs, commits from GitHub repos",
    },
    "source-salesforce": {
        "display": "Salesforce",
        "description": "Read objects from Salesforce",
    },
    "source-hubspot": {
        "display": "HubSpot",
        "description": "Read contacts, deals, etc. from HubSpot",
    },
}


def get_curated_source_list() -> list[dict[str, str]]:
    """Return curated list for the UI dropdown."""
    items = []
    for name, info in CURATED_SOURCES.items():
        items.append({
            "name": name,
            "display": info["display"],
            "description": info["description"],
        })
    return items


def get_curated_display_names() -> list[str]:
    """Return display name list for dropdown: 'Amazon S3 (source-s3)'."""
    return [
        f"{info['display']}  ({name})"
        for name, info in CURATED_SOURCES.items()
    ]


def resolve_connector_name(display_or_name: str) -> str:
    """
    Resolve a display string like 'Amazon S3 (source-s3)' or raw name 'source-s3'
    back to the connector name.
    """
    # If it contains parentheses, extract the connector name
    if "(" in display_or_name and ")" in display_or_name:
        return display_or_name.split("(")[-1].rstrip(")")
    # Otherwise treat as raw connector name
    return display_or_name.strip()


# ── Connector Configuration Templates ────────────────────────────────────────
# Provide UI-friendly config templates for common sources so users don't have
# to guess the config JSON structure.

SOURCE_CONFIG_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "source-s3": [
        {"key": "bucket", "label": "S3 Bucket Name", "type": "string", "required": True,
         "placeholder": "my-data-bucket"},
        {"key": "aws_access_key_id", "label": "AWS Access Key ID", "type": "password",
         "required": False, "placeholder": "(uses default credentials if empty)"},
        {"key": "aws_secret_access_key", "label": "AWS Secret Access Key", "type": "password",
         "required": False},
        {"key": "region_name", "label": "AWS Region", "type": "string", "required": False,
         "placeholder": "us-east-1"},
        {"key": "path_prefix", "label": "Path Prefix (folder)", "type": "string",
         "required": False, "placeholder": "data/raw/"},
        {"key": "streams", "label": "File Config (JSON array)", "type": "json", "required": True,
         "placeholder": '[{"name": "my_data", "format": {"filetype": "csv"}, "globs": ["**/*.csv"]}]'},
    ],
    "source-gcs": [
        {"key": "service_account", "label": "Service Account JSON", "type": "json",
         "required": False, "placeholder": "(uses ADC if empty)"},
        {"key": "bucket", "label": "GCS Bucket Name", "type": "string", "required": True,
         "placeholder": "my-gcs-bucket"},
        {"key": "streams", "label": "File Config (JSON array)", "type": "json", "required": True,
         "placeholder": '[{"name": "my_data", "format": {"filetype": "csv"}, "globs": ["**/*.csv"]}]'},
    ],
    "source-postgres": [
        {"key": "host", "label": "Host", "type": "string", "required": True,
         "placeholder": "localhost"},
        {"key": "port", "label": "Port", "type": "number", "required": True, "placeholder": "5432"},
        {"key": "database", "label": "Database", "type": "string", "required": True},
        {"key": "username", "label": "Username", "type": "string", "required": True},
        {"key": "password", "label": "Password", "type": "password", "required": True},
        {"key": "schemas", "label": "Schemas (comma-separated)", "type": "string",
         "required": False, "placeholder": "public"},
    ],
    "source-mysql": [
        {"key": "host", "label": "Host", "type": "string", "required": True},
        {"key": "port", "label": "Port", "type": "number", "required": True, "placeholder": "3306"},
        {"key": "database", "label": "Database", "type": "string", "required": True},
        {"key": "username", "label": "Username", "type": "string", "required": True},
        {"key": "password", "label": "Password", "type": "password", "required": True},
    ],
    "source-bigquery": [
        {"key": "project_id", "label": "GCP Project ID", "type": "string", "required": True},
        {"key": "dataset_id", "label": "Dataset ID", "type": "string", "required": True},
        {"key": "credentials_json", "label": "Service Account JSON", "type": "json",
         "required": False, "placeholder": "(uses ADC if empty)"},
    ],
    "source-google-sheets": [
        {"key": "spreadsheet_id", "label": "Spreadsheet ID", "type": "string", "required": True,
         "placeholder": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"},
        {"key": "credentials", "label": "Credentials JSON", "type": "json", "required": True},
    ],
    "source-faker": [
        {"key": "count", "label": "Number of Records", "type": "number", "required": False,
         "placeholder": "1000"},
        {"key": "seed", "label": "Random Seed", "type": "number", "required": False,
         "placeholder": "0"},
    ],
    "source-file": [
        {"key": "url", "label": "File URL", "type": "string", "required": True,
         "placeholder": "https://example.com/data.csv"},
        {"key": "format", "label": "Format", "type": "string", "required": False,
         "placeholder": "csv"},
        {"key": "dataset_name", "label": "Dataset Name", "type": "string", "required": True,
         "placeholder": "my_data"},
    ],
    "source-snowflake": [
        {"key": "host", "label": "Snowflake Account URL", "type": "string", "required": True,
         "placeholder": "myaccount.snowflakecomputing.com"},
        {"key": "role", "label": "Role", "type": "string", "required": True},
        {"key": "warehouse", "label": "Warehouse", "type": "string", "required": True},
        {"key": "database", "label": "Database", "type": "string", "required": True},
        {"key": "schema", "label": "Schema", "type": "string", "required": True},
        {"key": "username", "label": "Username", "type": "string", "required": True},
        {"key": "password", "label": "Password", "type": "password", "required": True},
    ],
}


def get_config_template(connector_name: str) -> list[dict[str, Any]]:
    """Return the config template for a connector, or an empty list for raw JSON config."""
    return SOURCE_CONFIG_TEMPLATES.get(connector_name, [])


# ── PyAirbyte Operations ─────────────────────────────────────────────────────

def create_source(
    connector_name: str,
    config: dict[str, Any],
    streams: Optional[list[str]] = None,
):
    """
    Create and return a PyAirbyte Source object.

    Args:
        connector_name: e.g. "source-s3", "source-gcs", "source-postgres"
        config: connector config dict
        streams: optional list of stream names to select

    Returns:
        airbyte.sources.Source instance
    """
    import airbyte as ab

    source = ab.get_source(
        connector_name,
        config=config,
        install_if_missing=True,
    )
    if streams:
        source.select_streams(streams)
    return source


def check_source(connector_name: str, config: dict[str, Any]) -> tuple[bool, str]:
    """
    Verify that a source connection is valid.

    Returns:
        (success: bool, message: str)
    """
    try:
        source = create_source(connector_name, config)
        source.check()
        return True, "✅ Connection verified successfully."
    except Exception as e:
        return False, f"❌ Connection check failed: {e}"


def discover_streams(connector_name: str, config: dict[str, Any]) -> list[StreamInfo]:
    """
    Discover available streams for a source.

    Returns a list of StreamInfo objects.
    """
    try:
        source = create_source(connector_name, config)
        stream_names = source.get_available_streams()
        return [StreamInfo(name=s) for s in stream_names]
    except Exception as e:
        logger.error(f"Failed to discover streams: {e}")
        return []


def preview_stream(
    connector_name: str,
    config: dict[str, Any],
    stream_name: str,
    max_records: int = 100,
) -> PreviewResult:
    """
    Preview records from a single stream.

    Uses source.get_records() which does NOT cache data — just peeks.
    """
    try:
        source = create_source(connector_name, config, streams=[stream_name])

        records = []
        for record in source.get_records(stream_name):
            records.append(dict(record))
            if len(records) >= max_records:
                break

        if not records:
            return PreviewResult(
                success=True,
                dataframe=pd.DataFrame(),
                row_count=0,
                column_count=0,
                stream_name=stream_name,
            )

        df = pd.DataFrame(records)
        schema_info = _build_schema_info(df)

        return PreviewResult(
            success=True,
            dataframe=df,
            row_count=len(df),
            column_count=len(df.columns),
            stream_name=stream_name,
            schema_info=schema_info,
        )
    except Exception as e:
        return PreviewResult(
            success=False,
            stream_name=stream_name,
            error=str(e),
        )


def sync_to_bigquery(
    connector_name: str,
    config: dict[str, Any],
    streams: list[str],
    bq_project: str,
    bq_dataset: str,
    credentials_path: Optional[str] = None,
) -> SyncResult:
    """
    Full sync: read from source → write to BigQuery using PyAirbyte's BigQueryCache.

    Args:
        connector_name: e.g. "source-s3"
        config: source config dict
        streams: list of stream names to sync
        bq_project: BigQuery project ID
        bq_dataset: BigQuery dataset name
        credentials_path: optional path to service account JSON
    """
    import airbyte as ab
    from airbyte.caches import BigQueryCache

    start = time.time()
    try:
        # Configure BigQuery destination
        cache_kwargs = {
            "project_name": bq_project,
            "dataset_name": bq_dataset,
        }
        if credentials_path:
            cache_kwargs["credentials_path"] = credentials_path

        cache = BigQueryCache(**cache_kwargs)

        # Configure source
        source = create_source(connector_name, config, streams=streams)
        source.check()

        # Run the sync
        read_result = source.read(cache=cache)

        elapsed = time.time() - start
        total_records = 0
        synced_streams = []

        for stream_name in streams:
            try:
                dataset = read_result[stream_name]
                # Count records from the dataset
                count = len(dataset.to_pandas())
                total_records += count
                synced_streams.append(stream_name)
            except Exception as e:
                logger.warning(f"Could not count records for stream {stream_name}: {e}")
                synced_streams.append(stream_name)

        return SyncResult(
            success=True,
            streams_synced=synced_streams,
            total_records=total_records,
            elapsed_seconds=round(elapsed, 2),
            destination=f"{bq_project}.{bq_dataset}",
        )

    except Exception as e:
        return SyncResult(
            success=False,
            error=str(e),
            elapsed_seconds=round(time.time() - start, 2),
        )


def sync_to_local_cache(
    connector_name: str,
    config: dict[str, Any],
    streams: list[str],
    cache_path: Optional[str] = None,
) -> tuple[Any, SyncResult]:
    """
    Sync data to a local DuckDB cache. Returns (read_result, SyncResult).

    Useful for preview or when BigQuery is not configured.
    """
    import airbyte as ab

    start = time.time()
    try:
        if cache_path:
            cache = ab.DuckDBCache(db_path=cache_path)
        else:
            cache = ab.new_local_cache()

        source = create_source(connector_name, config, streams=streams)
        read_result = source.read(cache=cache)

        elapsed = time.time() - start
        total_records = 0
        synced_streams = []

        for stream_name in streams:
            try:
                df = read_result[stream_name].to_pandas()
                total_records += len(df)
                synced_streams.append(stream_name)
            except Exception:
                synced_streams.append(stream_name)

        result = SyncResult(
            success=True,
            streams_synced=synced_streams,
            total_records=total_records,
            elapsed_seconds=round(elapsed, 2),
            destination="local DuckDB cache",
        )
        return read_result, result

    except Exception as e:
        result = SyncResult(
            success=False,
            error=str(e),
            elapsed_seconds=round(time.time() - start, 2),
        )
        return None, result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_schema_info(df: pd.DataFrame) -> list[dict]:
    """Build column schema info from a DataFrame."""
    info = []
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
        if len(sample) > 80:
            sample = sample[:80] + "..."
        info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "nulls": nulls,
            "null_pct": round(100 * nulls / len(df), 1) if len(df) > 0 else 0,
            "sample": sample,
        })
    return info
