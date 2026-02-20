"""
Step 1: Metadata Loader
Parses FHIR StructureDefinition JSON files into structured table/column schemas
that the LLM can use to generate accurate SQL queries.

Supports two metadata sources:
  1. FHIR StructureDefinition JSON files (semantic metadata)
  2. BigQuery INFORMATION_SCHEMA (live structural metadata) — added in Step 3

Usage:
    from metadata_loader import load_metadata_from_json_dir, format_schemas_for_prompt

    schemas = load_metadata_from_json_dir("/path/to/json/metadata/")
    prompt_context = format_schemas_for_prompt(schemas)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnSchema:
    """Represents a single column in a BigQuery table."""
    name: str
    data_type: str  # string, integer, decimal, code, date, etc.
    short_description: str
    full_description: str
    is_required: bool = True
    sensitivity_label: Optional[str] = None  # UID, PHI, or None
    value_set_binding: Optional[str] = None  # Description of allowed values
    fhir_mapping: Optional[str] = None  # e.g. "Patient.identifier"
    fhir_mapping_comment: Optional[str] = None
    measurement_method: Optional[str] = None
    comment: Optional[str] = None  # Raw comment from JSON


@dataclass
class JoinLink:
    """Represents a structural link (join relationship) to another table."""
    target_profile_url: str
    target_table_name: Optional[str] = None  # Resolved after all profiles loaded


@dataclass
class TableSchema:
    """Represents a BigQuery table with all its metadata."""
    # Identity
    bq_table_name: str  # e.g. "bhs.admin.COEVAL"
    profile_id: str  # e.g. "bhs-admin-coeval"
    profile_url: str  # e.g. "http://fhir.verily.com/StructureDefinition/bhs-admin-coeval"

    # Descriptive
    title: str  # e.g. "BHS Admin — COEVAL (Cohort Eligibility)"
    description: str
    purpose: str  # e.g. "One record per participant"

    # Keys
    primary_key: str  # e.g. "USUBJID" or "athena_id + procedure_code + service_date"

    # Columns
    columns: list[ColumnSchema] = field(default_factory=list)

    # Relationships
    join_links: list[JoinLink] = field(default_factory=list)

    # Table-level metadata
    confidentiality: Optional[str] = None  # e.g. "Restricted"
    compliance_zone: Optional[str] = None  # e.g. "HIPAA Covered"
    schema_stability: Optional[str] = None  # e.g. "Stable"
    retention_years: Optional[int] = None
    domain_contact: Optional[str] = None


# ── FHIR JSON Parsing ────────────────────────────────────────────────────────

def _extract_extension_value(extensions: list[dict], url: str) -> Optional[str]:
    """Extract a single string value from a FHIR extension by URL."""
    for ext in extensions:
        if ext.get("url") == url:
            return (
                ext.get("valueString")
                or ext.get("valueCanonical")
                or ext.get("valueCode")
            )
    return None


def _extract_bq_table_name(extensions: list[dict]) -> Optional[str]:
    """Extract BigQuery table name from BigQueryTableSchemaMetadata extension."""
    for ext in extensions:
        if ext.get("url") == "http://fhir.verily.com/StructureDefinition/BigQueryTableSchemaMetadata":
            for sub_ext in ext.get("extension", []):
                if sub_ext.get("url") == "table-name":
                    return sub_ext.get("valueString")
    return None


def _extract_structural_links(extensions: list[dict]) -> list[JoinLink]:
    """Extract all verily-structural-link extensions."""
    links = []
    for ext in extensions:
        if ext.get("url") == "http://fhir.verily.com/StructureDefinition/verily-structural-link":
            target_url = ext.get("valueCanonical", "")
            links.append(JoinLink(target_profile_url=target_url))
    return links


def _extract_sensitivity_label(element: dict) -> Optional[str]:
    """Extract sensitivity label (UID, PHI) from element extensions."""
    for ext in element.get("extension", []):
        if ext.get("url") == "http://hl7.org/fhir/uv/security-label-ds4p/StructureDefinition/extension-inline-sec-label":
            coding = ext.get("valueCoding", {})
            if coding.get("system") == "http://fhir.verily.com/CodeSystem/SensitivityLabels":
                return coding.get("code")
    return None


def _extract_measurement_method(element: dict) -> Optional[str]:
    """Extract measurement method text from semantic metadata extension."""
    for ext in element.get("extension", []):
        if ext.get("url") == "http://fhir.verily.com/StructureDefinition/verily-attribute-semantic-metadata":
            for sub_ext in ext.get("extension", []):
                if sub_ext.get("url") == "measurement-method":
                    concept = sub_ext.get("valueCodeableConcept", {})
                    return concept.get("text") or (
                        concept.get("coding", [{}])[0].get("display")
                        if concept.get("coding")
                        else None
                    )
    return None


def _extract_fhir_mapping(element: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract FHIR mapping (map path + comment) from element."""
    for mapping in element.get("mapping", []):
        if mapping.get("identity") == "vfig":
            return mapping.get("map"), mapping.get("comment")
    return None, None


def _extract_binding(element: dict) -> Optional[str]:
    """Extract value set binding description."""
    binding = element.get("binding")
    if binding:
        desc = binding.get("description", "")
        vs = binding.get("valueSet", "")
        strength = binding.get("strength", "")
        parts = [p for p in [desc, f"strength={strength}" if strength else ""] if p]
        return "; ".join(parts) if parts else None
    return None


def _fhir_type_to_bq_type(fhir_type: str) -> str:
    """Map FHIR element type codes to approximate BigQuery type names."""
    mapping = {
        "string": "STRING",
        "code": "STRING",
        "integer": "INTEGER",
        "decimal": "FLOAT",
        "boolean": "BOOLEAN",
        "date": "DATE",
        "dateTime": "TIMESTAMP",
        "instant": "TIMESTAMP",
    }
    return mapping.get(fhir_type, "STRING")


def _extract_use_context(resource: dict) -> dict:
    """Extract use context metadata (compliance zone, retention, stability)."""
    result = {}
    for ctx in resource.get("useContext", []):
        code = ctx.get("code", {}).get("code", "")
        if code == "compliance-zone":
            codings = ctx.get("valueCodeableConcept", {}).get("coding", [])
            if codings:
                result["compliance_zone"] = codings[0].get("display")
        elif code == "retention-policy":
            qty = ctx.get("valueQuantity", {})
            result["retention_years"] = qty.get("value")
        elif code == "schema-stability":
            codings = ctx.get("valueCodeableConcept", {}).get("coding", [])
            if codings:
                result["schema_stability"] = codings[0].get("display")
    return result


def parse_fhir_structure_definition(json_data: dict) -> Optional[TableSchema]:
    """
    Parse a single FHIR StructureDefinition JSON into a TableSchema.

    Returns None if the JSON is not a StructureDefinition or lacks
    the BigQueryTableSchemaMetadata extension.
    """
    if json_data.get("resourceType") != "StructureDefinition":
        return None

    extensions = json_data.get("extension", [])

    # Extract BigQuery table name — skip if not present
    bq_table_name = _extract_bq_table_name(extensions)
    if not bq_table_name:
        return None

    # Extract primary identity key
    primary_key = _extract_extension_value(
        extensions,
        "http://fhir.verily.com/StructureDefinition/verily-primary-identity"
    ) or ""

    # Extract structural links
    join_links = _extract_structural_links(extensions)

    # Use context (compliance, retention, stability)
    use_ctx = _extract_use_context(json_data)

    # Confidentiality from meta.security
    confidentiality = None
    for sec in json_data.get("meta", {}).get("security", []):
        if "Confidentiality" in sec.get("system", ""):
            confidentiality = sec.get("display")

    # Domain contact
    domain_contact = None
    contacts = json_data.get("contact", [])
    if contacts:
        domain_contact = contacts[0].get("name")

    # Build table schema
    table = TableSchema(
        bq_table_name=bq_table_name,
        profile_id=json_data.get("id", ""),
        profile_url=json_data.get("url", ""),
        title=json_data.get("title", ""),
        description=json_data.get("description", ""),
        purpose=json_data.get("purpose", ""),
        primary_key=primary_key,
        join_links=join_links,
        confidentiality=confidentiality,
        compliance_zone=use_ctx.get("compliance_zone"),
        schema_stability=use_ctx.get("schema_stability"),
        retention_years=use_ctx.get("retention_years"),
        domain_contact=domain_contact,
    )

    # Parse columns from differential.element
    elements = json_data.get("differential", {}).get("element", [])
    for element in elements:
        path = element.get("path", "")

        # Skip the root element (no dot = table-level, not a column)
        if "." not in path:
            continue

        col_name = path.split(".")[-1]

        # Determine type
        types = element.get("type", [])
        fhir_type = types[0].get("code", "string") if types else "string"
        bq_type = _fhir_type_to_bq_type(fhir_type)

        # Required?
        is_required = element.get("min", 0) >= 1

        # Sensitivity
        sensitivity = _extract_sensitivity_label(element)

        # FHIR mapping
        fhir_map, fhir_map_comment = _extract_fhir_mapping(element)

        # Value set binding
        binding_desc = _extract_binding(element)

        # Measurement method
        measurement = _extract_measurement_method(element)

        col = ColumnSchema(
            name=col_name,
            data_type=bq_type,
            short_description=element.get("short", ""),
            full_description=element.get("definition", ""),
            is_required=is_required,
            sensitivity_label=sensitivity,
            value_set_binding=binding_desc,
            fhir_mapping=fhir_map,
            fhir_mapping_comment=fhir_map_comment,
            measurement_method=measurement,
            comment=element.get("comment"),
        )
        table.columns.append(col)

    return table


# ── Directory Loading ─────────────────────────────────────────────────────────

def load_metadata_from_json_dir(
    json_dir: str | Path,
    recursive: bool = True,
) -> dict[str, TableSchema]:
    """
    Load all FHIR StructureDefinition JSONs from a local directory.

    Args:
        json_dir: Path to the directory containing JSON files.
        recursive: If True, also scan subdirectories.

    Returns:
        dict mapping BigQuery table name → TableSchema
    """
    json_dir = Path(json_dir)
    schemas: dict[str, TableSchema] = {}

    # Collect all .json files
    pattern = "**/*.json" if recursive else "*.json"
    json_files = sorted(json_dir.glob(pattern))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        table = parse_fhir_structure_definition(data)
        if table:
            schemas[table.bq_table_name] = table

    # Resolve join links: map profile URLs → table names
    url_to_table = {t.profile_url: t.bq_table_name for t in schemas.values()}
    for table in schemas.values():
        for link in table.join_links:
            link.target_table_name = url_to_table.get(link.target_profile_url)

    return schemas


def load_metadata_from_gcs(
    gcs_uri: str,
) -> dict[str, TableSchema]:
    """
    Load all FHIR StructureDefinition JSONs from a GCS bucket/prefix.

    Args:
        gcs_uri: GCS URI like 'gs://bucket-name' or 'gs://bucket-name/prefix/'

    Returns:
        dict mapping BigQuery table name → TableSchema
    """
    from google.cloud import storage

    # Parse the GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    path = gcs_uri[5:]  # Remove 'gs://'
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # Ensure prefix ends with / if non-empty
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    print(f"  📦 Loading metadata from gs://{bucket_name}/{prefix}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    schemas: dict[str, TableSchema] = {}
    file_count = 0

    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue

        try:
            content = blob.download_as_text()
            data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
            print(f"  ⚠ Skipping {blob.name}: {e}")
            continue

        table = parse_fhir_structure_definition(data)
        if table:
            schemas[table.bq_table_name] = table
            file_count += 1

    print(f"  ✓ Parsed {file_count} table schemas from {gcs_uri}")

    # Resolve join links: map profile URLs → table names
    url_to_table = {t.profile_url: t.bq_table_name for t in schemas.values()}
    for table in schemas.values():
        for link in table.join_links:
            link.target_table_name = url_to_table.get(link.target_profile_url)

    return schemas


def load_metadata(source: str) -> dict[str, TableSchema]:
    """
    Load metadata from either a local directory or a GCS URI.
    Auto-detects based on whether the source starts with 'gs://'.

    Args:
        source: Local path or GCS URI (gs://bucket/prefix)

    Returns:
        dict mapping BigQuery table name → TableSchema
    """
    if source.startswith("gs://"):
        return load_metadata_from_gcs(source)
    else:
        return load_metadata_from_json_dir(source)


# ── Prompt Formatting ─────────────────────────────────────────────────────────

def format_schemas_for_prompt(schemas: dict[str, TableSchema]) -> str:
    """
    Format all table schemas into a text block suitable for injection
    into an LLM system prompt.

    This is the primary interface between the metadata loader and the
    prompt engine (Step 2).
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("AVAILABLE TABLES AND COLUMNS")
    lines.append("=" * 70)

    for table_name, table in sorted(schemas.items()):
        lines.append("")
        lines.append(f"TABLE: `{table_name}`")
        lines.append(f"  Title: {table.title}")
        lines.append(f"  Description: {table.description}")
        lines.append(f"  Grain: {table.purpose}")
        lines.append(f"  Primary Key: {table.primary_key}")

        # Join targets
        join_targets = [
            link.target_table_name or link.target_profile_url
            for link in table.join_links
        ]
        if join_targets:
            lines.append(f"  Joins To: {', '.join(join_targets)}")

        # Columns
        lines.append(f"  Columns:")
        for col in table.columns:
            required = "REQUIRED" if col.is_required else "OPTIONAL"
            sens = f" [{col.sensitivity_label}]" if col.sensitivity_label else ""
            lines.append(f"    - {col.name} ({col.data_type}, {required}{sens})")
            lines.append(f"      {col.full_description}")
            if col.value_set_binding:
                lines.append(f"      Allowed values: {col.value_set_binding}")
            if col.fhir_mapping:
                lines.append(f"      FHIR: {col.fhir_mapping}")

        lines.append("")

    return "\n".join(lines)


# ── Runtime BQ Resolution ─────────────────────────────────────────────────

def resolve_against_bigquery(
    schemas: dict[str, TableSchema],
    data_project_ids: str | list[str],
) -> dict[str, TableSchema]:
    """
    Match FHIR metadata table names against actual BigQuery tables at runtime.

    The FHIR metadata uses names like 'bhs.admin.COEVAL' (study.dataset.table)
    but BQ uses 'project.dataset.table'.  This function:
      1. Lists all real tables in each data project's datasets
      2. Strips the study prefix from each FHIR name → 'dataset.TABLE'
      3. Matches against real BQ tables
      4. Returns a new dict with only matched schemas, keyed by the
         fully-qualified BQ name (project.dataset.table)
      5. Updates join links to use resolved names too

    Supports multiple data projects (e.g. BHS data in one project, PRESCO
    in another) — tables are matched in order, first match wins.

    Args:
        schemas: Raw schemas from load_metadata_from_json_dir().
        data_project_ids: One or more GCP project IDs where BigQuery data lives.
                          Can be a single string or a list of strings.

    Returns:
        New dict mapping fully-qualified BQ table name → TableSchema.
        Only tables that exist in both metadata AND BigQuery are included.
    """
    from google.cloud import bigquery

    # Normalize to list
    if isinstance(data_project_ids, str):
        data_project_ids = [data_project_ids]

    # Step 1: Discover all real tables across all projects
    actual_tables: dict[str, str] = {}  # "dataset.TABLE" → "project.dataset.TABLE"
    for project_id in data_project_ids:
        try:
            client = bigquery.Client(project=project_id)
            datasets = list(client.list_datasets())
        except Exception as e:
            print(f"  ⚠ Could not list datasets in {project_id}: {e}")
            continue

        project_count = 0
        for ds_ref in datasets:
            ds_id = ds_ref.dataset_id
            try:
                for tbl in client.list_tables(f"{project_id}.{ds_id}"):
                    key = f"{ds_id}.{tbl.table_id}"
                    if key not in actual_tables:  # First match wins
                        fq_name = f"{project_id}.{ds_id}.{tbl.table_id}"
                        actual_tables[key] = fq_name
                        project_count += 1
            except Exception:
                continue

        print(f"  📊 Found {project_count} tables across {len(datasets)} datasets in {project_id}")

    # Step 2: Match FHIR metadata to real BQ tables
    fhir_to_bq: dict[str, str] = {}  # old FHIR name → new FQ BQ name
    resolved: dict[str, TableSchema] = {}

    for fhir_name, table in schemas.items():
        parts = fhir_name.split(".")
        if len(parts) == 3:
            # "bhs.admin.COEVAL" → lookup "admin.COEVAL"
            lookup_key = f"{parts[1]}.{parts[2]}"
        elif len(parts) == 2:
            # Already "dataset.TABLE" — try direct match
            lookup_key = fhir_name
        else:
            continue  # Unexpected format

        if lookup_key in actual_tables:
            fq_name = actual_tables[lookup_key]
            fhir_to_bq[fhir_name] = fq_name
            # Update the schema's table name to the fully-qualified BQ name
            table.bq_table_name = fq_name
            resolved[fq_name] = table
            print(f"  ✅ {fhir_name:<40} → {fq_name}")
        else:
            print(f"  ⏭  {fhir_name:<40} → not in BQ (skipped)")

    # Step 3: Update join links to use resolved names
    for table in resolved.values():
        for link in table.join_links:
            if link.target_table_name and link.target_table_name in fhir_to_bq:
                link.target_table_name = fhir_to_bq[link.target_table_name]

    print(f"  ✓ Resolved {len(resolved)}/{len(schemas)} tables with metadata + BQ data")
    return resolved


def format_table_summary(schemas: dict[str, TableSchema]) -> str:
    """
    Format a compact summary of all tables (for display in UI sidebar).
    """
    lines: list[str] = []
    for table_name, table in sorted(schemas.items()):
        col_names = [c.name for c in table.columns]
        lines.append(f"📊 {table_name}")
        lines.append(f"   {table.title}")
        lines.append(f"   Key: {table.primary_key}")
        lines.append(f"   Cols: {', '.join(col_names)}")
        lines.append("")
    return "\n".join(lines)


# ── Main (for quick testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metadata_loader.py <path_to_json_dir>")
        print("Example: python metadata_loader.py ../product_mgmnt/Metadata/Metadata\\ JSON\\ for\\ Demo/JSON\\ Metadata/")
        sys.exit(1)

    json_dir = sys.argv[1]
    print(f"Loading metadata from: {json_dir}")
    print()

    schemas = load_metadata_from_json_dir(json_dir)

    print(f"Loaded {len(schemas)} table schemas:")
    for name, table in sorted(schemas.items()):
        join_targets = [l.target_table_name or "???" for l in table.join_links]
        print(f"  ✓ {name} ({len(table.columns)} columns, PK={table.primary_key}, joins={join_targets})")

    print()
    print("=" * 70)
    print("FULL PROMPT CONTEXT:")
    print("=" * 70)
    print(format_schemas_for_prompt(schemas))

