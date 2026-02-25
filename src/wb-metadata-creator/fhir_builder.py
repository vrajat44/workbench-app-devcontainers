"""
Deterministic FHIR JSON builders — no LLM calls.

Builds all FHIR resources programmatically from profiling data and user-edited metadata:
  - StructureDefinition (schema + semantics + sensitivity)
  - ValueSet, CodeSystem
  - Terminology Bundle (CodeSystem + ValueSet + ConceptMap)
  - MeasureReport (data profile)
  - Measure definition (shared template)
  - ConceptMap (deterministic builder)
"""

from __future__ import annotations

import re
from typing import Optional

from models import BQColumnInfo, BQTableInfo, StudyConfig, ColumnProfile, TableProfile
from sensitivity import (
    build_sec_label_extension,
    sensitivity_comment,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_METRIC_SYSTEM = "http://fhir.verily.com/CodeSystem/data-profile-metric"
_LEVEL_SYSTEM = "http://fhir.verily.com/CodeSystem/data-profile-level"
_ELEMENT_SYSTEM = "http://fhir.verily.com/CodeSystem/data-profile-element"

# SNOMED ordinal codes for frequency-distribution ranking
_ORDINAL_CODES = [
    {"system": "http://snomed.info/sct", "code": "255216001", "display": "First"},
    {"system": "http://snomed.info/sct", "code": "81170007", "display": "Second"},
    {"system": "http://snomed.info/sct", "code": "70905002", "display": "Third"},
]

# BQ types that are numeric
_NUMERIC_BQ_TYPES = {"INT64", "INTEGER", "FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC"}

# BQ → FHIR type mapping
_BQ_TO_FHIR_TYPE = {
    "STRING": "string", "BYTES": "string",
    "INT64": "integer", "INTEGER": "integer",
    "FLOAT64": "decimal", "FLOAT": "decimal", "NUMERIC": "decimal", "BIGNUMERIC": "decimal",
    "BOOL": "boolean", "BOOLEAN": "boolean",
    "DATE": "date", "DATETIME": "dateTime", "TIMESTAMP": "dateTime",
    "TIME": "string", "GEOGRAPHY": "string", "JSON": "string",
}


# ── Utility Helpers ───────────────────────────────────────────────────────────

def _to_kebab(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _to_pascal(s: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[^a-zA-Z0-9]", s) if w)


def _get_iso_date() -> str:
    """Get current date in ISO format."""
    from datetime import date
    return date.today().isoformat()


def map_bq_type_to_fhir(bq_type: str) -> str:
    """Map a BigQuery data type to a FHIR element type."""
    return _BQ_TO_FHIR_TYPE.get(bq_type.upper().split("<")[0].strip(), "string")


def _format_contact_name(domain_contact: str, study_name: str) -> str:
    """
    Ensure contact name uses the 'domain:' prefix pattern per gold standard.
    Examples:
        "" → "domain:test-research"
        "test" → "domain:test"
        "domain:bhs-research" → "domain:bhs-research" (unchanged)
    """
    if not domain_contact:
        return f"domain:{study_name.lower()}-research"
    if domain_contact.startswith("domain:"):
        return domain_contact
    return f"domain:{domain_contact}"


# ── StructureDefinition Builder ───────────────────────────────────────────────

def build_structure_definition(
    table_info: BQTableInfo,
    study_config: StudyConfig,
    table_meta: dict,
    columns_meta: list[dict],
) -> dict:
    """
    Build a complete FHIR StructureDefinition JSON programmatically.
    No LLM call — all structure is code-generated.
    Semantic content comes from the earlier LLM call / user edits.
    """
    study = study_config.study_name
    dataset = table_info.dataset_id
    table = table_info.table_id

    profile_id = _to_kebab(f"{study}-{dataset}-{table}")
    # Build name: preserve study acronym casing (e.g. "BHS") and PascalCase
    # dataset/table parts — handles multi-word names (e.g. "my_dataset" → "MyDataset").
    # Gold standard: BHSAnalysisDiagnoses
    name = study + _to_pascal(dataset) + _to_pascal(table)
    url = f"http://fhir.verily.com/StructureDefinition/{profile_id}"

    conf_code = study_config.confidentiality or "R"
    conf_display = {"R": "Restricted", "N": "Normal"}.get(conf_code, conf_code)

    sd = {
        "resourceType": "StructureDefinition",
        "id": profile_id,
        "meta": {"security": [{"system": "http://terminology.hl7.org/CodeSystem/v3-Confidentiality", "code": conf_code, "display": conf_display}]},
        "extension": [
            {"url": "http://fhir.verily.com/StructureDefinition/BigQueryTableSchemaMetadata", "extension": [{"url": "table-name", "valueString": table_info.fq_name}]},
            {"url": "http://fhir.verily.com/StructureDefinition/verily-primary-identity", "valueString": table_meta.get("primary_key", "")},
        ] + [
            {"url": "http://fhir.verily.com/StructureDefinition/verily-structural-link", "valueCanonical": link}
            for link in table_meta.get("structural_links", [])
        ],
        "url": url,
        "name": name,
        "title": table_meta.get("title", f"{study} {dataset} — {table}"),
        "status": "active",
        "purpose": table_meta.get("purpose", ""),
        "description": table_meta.get("description", ""),
        "contact": [{"name": _format_contact_name(study_config.domain_contact, study), "telecom": [{"system": "url", "value": f"Organization/{study.lower()}-research"}]}],
        "useContext": [
            {"code": {"system": "http://fhir.verily.com/CodeSystem/usage-context-type", "code": "compliance-zone"}, "valueCodeableConcept": {"coding": [{"system": "http://fhir.verily.com/CodeSystem/compliance-zones", "code": study_config.compliance_zone, "display": study_config.compliance_zone}]}},
            {"code": {"system": "http://fhir.verily.com/CodeSystem/usage-context-type", "code": "retention-policy"}, "valueQuantity": {"value": study_config.retention_years, "unit": "years", "system": "http://unitsofmeasure.org", "code": "a"}},
            {"code": {"system": "http://fhir.verily.com/CodeSystem/usage-context-type", "code": "schema-stability"}, "valueCodeableConcept": {"coding": [{"system": "http://fhir.verily.com/CodeSystem/schema-stability", "code": study_config.schema_stability, "display": study_config.schema_stability.capitalize()}]}},
        ],
        "kind": "logical",
        "abstract": False,
        "type": url,
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Element",
        "derivation": "specialization",
        "mapping": [{"identity": "vfig", "uri": "http://fhir.verily.com/vfig", "name": "Verily FHIR Implementation Guide", "comment": "Semantic mappings from physical columns to the Verily common data model (VFIG)"}],
        "differential": {"element": _build_elements(name, columns_meta, profile_id, study)},
    }

    # Append L3/L4 metadata boilerplate to description (gold-standard parity)
    _L3L4_SUFFIX = (
        " This logical model describes the BigQuery table schema and carries "
        "both Level III (Asset) and Level IV (Attribute) metadata. Sensitive data "
        "elements are tagged using the HL7 DS4P inline security label extension "
        "with Verily Sensitivity Labels CodeSystem."
    )
    if sd["description"] and _L3L4_SUFFIX.strip() not in sd["description"]:
        sd["description"] = sd["description"].rstrip() + _L3L4_SUFFIX

    return sd


def _build_elements(root_name: str, columns_meta: list[dict], profile_id: str, study_name: str = "") -> list[dict]:
    """Build the differential.element array from column metadata."""
    elements = [{
        "id": root_name, "path": root_name,
        "short": "Table Row", "definition": "One row in this table.",
        "min": 0, "max": "*",
    }]

    for col in columns_meta:
        col_name = col.get("Column", "")
        if not col_name:
            continue

        fhir_type = col.get("FHIR Type", "string") or "string"
        required = str(col.get("Required", "No")).strip().lower() in ("yes", "true", "1")
        sensitivity = str(col.get("Sensitivity", "")).strip().upper()
        measurement = str(col.get("Measurement", "")).strip()
        fhir_mapping = str(col.get("FHIR Mapping", "")).strip()
        is_coded = str(col.get("Coded", "")).strip().lower() in ("yes", "true", "1")

        # New rich semantic fields
        comment_text = str(col.get("Comment", "")).strip()
        mapping_cmt = str(col.get("Mapping Comment", "")).strip()
        measurement_text = str(col.get("Measurement Text", "")).strip()
        fixed_value = str(col.get("Fixed Value", "")).strip()

        elem: dict = {
            "id": f"{root_name}.{col_name}",
            "path": f"{root_name}.{col_name}",
            "short": col.get("Short Label", "") or col_name,
            "definition": col.get("Description", "") or "",
            "min": 1 if required else 0,
            "max": "1",
            "type": [{"code": fhir_type}],
        }

        # fixedString for constant columns
        if fixed_value:
            elem["fixedString"] = fixed_value

        # Comment: use rich LLM comment when present, fall back to sensitivity comment
        elem["comment"] = comment_text if comment_text else sensitivity_comment(sensitivity)

        # Extensions
        extensions = []
        sec_label_ext = build_sec_label_extension(sensitivity)
        if sec_label_ext:
            extensions.append(sec_label_ext)

        if measurement and measurement not in ("—", ""):
            meas_code = measurement.lower().replace(" ", "-")
            meas_display = "-".join(w.capitalize() for w in meas_code.split("-"))  # "self-reported" → "Self-Reported"
            meas_text = measurement_text if measurement_text else meas_code
            extensions.append({
                "url": "http://fhir.verily.com/StructureDefinition/verily-attribute-semantic-metadata",
                "extension": [{"url": "measurement-method", "valueCodeableConcept": {
                    "coding": [{"system": "http://fhir.verily.com/CodeSystem/measurement-method", "code": meas_code, "display": meas_display}],
                    "text": meas_text,
                }}],
            })

        if extensions:
            elem["extension"] = extensions

        if fhir_mapping and fhir_mapping not in ("—", ""):
            m = {"identity": "vfig", "map": fhir_mapping}
            if mapping_cmt:
                m["comment"] = mapping_cmt
            elem["mapping"] = [m]

        if is_coded:
            # Use study-scoped ValueSet ID to match terminology bundle pattern
            # Gold standard: bhs-sex, bhs-visit-label (not bhs-analysis-diagnoses-sex)
            vs_id = _to_kebab(f"{study_name}-{col_name}") if study_name else _to_kebab(f"{profile_id}-{col_name}")
            short_label = col.get("Short Label", "") or col_name
            elem["binding"] = {
                "strength": "required",
                "valueSet": f"http://fhir.verily.com/ValueSet/{vs_id}",
                "description": f"Allowed values for {short_label}",
            }

        elements.append(elem)

    return elements


# ── ValueSet Builder ──────────────────────────────────────────────────────────

def build_value_set(profile_id: str, column_name: str, coded_values: list[str], study_name: str) -> dict:
    """Build a FHIR ValueSet JSON from actual coded values. No LLM needed."""
    vs_id = _to_kebab(f"{profile_id}-{column_name}")
    concepts = [{"code": str(v), "display": str(v)} for v in coded_values if v]
    return {
        "resourceType": "ValueSet",
        "id": vs_id,
        "url": f"http://fhir.verily.com/ValueSet/{vs_id}",
        "version": "1.0.0",
        "name": _to_pascal(vs_id),
        "title": f"{study_name} — {column_name} Values",
        "status": "active",
        "experimental": False,
        "description": f"Coded values for column {column_name}.",
        "compose": {"include": [{"system": f"http://fhir.verily.com/CodeSystem/{vs_id}", "concept": concepts}]},
    }


# ── CodeSystem Builder ────────────────────────────────────────────────────────

def build_code_system(
    profile_id: str,
    column_name: str,
    coded_values: list[str],
    study_name: str,
    column_description: str = "",
) -> dict:
    """
    Build a FHIR CodeSystem JSON for a coded column's physical values.
    No LLM needed — uses the actual distinct values from BQ profiling.
    """
    cs_id = _to_kebab(f"{study_name}-{column_name}")
    concepts = []
    for v in coded_values:
        if v:
            concepts.append({
                "code": str(v),
                "display": str(v),
                "definition": f"{column_name} value: {v}",
            })

    return {
        "resourceType": "CodeSystem",
        "id": cs_id,
        "url": f"http://fhir.verily.com/CodeSystem/{cs_id}",
        "name": _to_pascal(cs_id),
        "title": f"{study_name} — {column_name} Codes",
        "status": "active",
        "description": column_description or f"Code system for {column_name} values used in {study_name} study tables.",
        "content": "complete",
        "count": len(concepts),
        "concept": concepts,
    }


# ── Terminology Bundle Builder ────────────────────────────────────────────────

def build_terminology_bundle(
    profile_id: str,
    study_name: str,
    coded_columns: list[dict],
) -> Optional[dict]:
    """
    Build a FHIR Bundle containing CodeSystem + ValueSet for each coded column.

    Args:
        profile_id: The StructureDefinition profile id.
        study_name: Study name for naming conventions.
        coded_columns: List of dicts with keys: column_name, coded_values, description.

    Returns:
        FHIR Bundle dict, or None if no coded columns.
    """
    if not coded_columns:
        return None

    iso_date = _get_iso_date()
    entries = []

    for col in coded_columns:
        col_name = col.get("column_name", "")
        values = col.get("coded_values", [])
        description = col.get("description", "")
        if not col_name or not values:
            continue

        cs_id = _to_kebab(f"{study_name}-{col_name}")
        vs_id = cs_id

        # Build CodeSystem
        cs = build_code_system(profile_id, col_name, values, study_name, description)
        entries.append({
            "fullUrl": f"http://fhir.verily.com/CodeSystem/{cs_id}",
            "resource": cs,
        })

        # Build ValueSet referencing the CodeSystem
        vs_concepts = [{"code": str(v), "display": str(v)} for v in values if v]
        vs = {
            "resourceType": "ValueSet",
            "id": vs_id,
            "url": f"http://fhir.verily.com/ValueSet/{vs_id}",
            "name": _to_pascal(f"{vs_id}ValueSet"),
            "title": f"{study_name} — {col_name} Value Set",
            "status": "active",
            "description": f"Value set binding for the '{col_name}' column in {study_name} tables.",
            "compose": {
                "include": [{"system": f"http://fhir.verily.com/CodeSystem/{cs_id}"}],
            },
            "expansion": {
                "timestamp": f"{iso_date}T00:00:00Z",
                "total": len(vs_concepts),
                "contains": [
                    {"system": f"http://fhir.verily.com/CodeSystem/{cs_id}", "code": c["code"], "display": c["display"]}
                    for c in vs_concepts
                ],
            },
        }
        entries.append({
            "fullUrl": f"http://fhir.verily.com/ValueSet/{vs_id}",
            "resource": vs,
        })

    if not entries:
        return None

    return {
        "resourceType": "Bundle",
        "id": f"{_to_kebab(f'{study_name}-{profile_id}')}-terminology",
        "type": "collection",
        "meta": {"lastUpdated": f"{iso_date}T00:00:00Z"},
        "entry": entries,
    }


def build_terminology_bundle_with_concept_maps(
    profile_id: str,
    study_name: str,
    coded_columns: list[dict],
    concept_maps: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Build a FHIR Bundle containing CodeSystem + ValueSet + ConceptMap for each coded column.

    Same as build_terminology_bundle() but also appends any generated ConceptMaps.
    """
    bundle = build_terminology_bundle(profile_id, study_name, coded_columns)
    if bundle is None:
        return None

    # Append ConceptMaps to the bundle (strip internal metadata keys)
    if concept_maps:
        for cm in concept_maps:
            if cm and cm.get("resourceType") == "ConceptMap":
                clean_cm = {k: v for k, v in cm.items() if not k.startswith("_")}
                cm_id = clean_cm.get("id", "unknown")
                bundle["entry"].append({
                    "fullUrl": f"http://fhir.verily.com/ConceptMap/{cm_id}",
                    "resource": clean_cm,
                })

    return bundle


# ── ConceptMap Builder (deterministic) ────────────────────────────────────────

def build_concept_map(
    study_name: str,
    column_name: str,
    source_codes: list[dict],
    target_system: str,
    target_valueset: str,
    target_codes: list[dict],
    description: str = "",
) -> dict:
    """
    Build a FHIR ConceptMap JSON programmatically (no LLM needed).

    Args:
        study_name: Study name for naming.
        column_name: Source column name.
        source_codes: List of {"code": ..., "display": ...} for source concepts.
        target_system: Target CodeSystem URL.
        target_valueset: Target ValueSet URL.
        target_codes: List of {"source_code": ..., "target_code": ..., "target_display": ..., "equivalence": ...}.
        description: Optional description.
    """
    cm_id = _to_kebab(f"{study_name}-{column_name}-to-{_to_kebab(target_system.split('/')[-1])}")
    source_cs_id = _to_kebab(f"{study_name}-{column_name}")

    elements = []
    for mapping in target_codes:
        source_code = mapping.get("source_code", "")
        source_display = next((c["display"] for c in source_codes if c["code"] == source_code), source_code)
        elements.append({
            "code": source_code,
            "display": source_display,
            "target": [{
                "code": mapping.get("target_code", ""),
                "display": mapping.get("target_display", ""),
                "equivalence": mapping.get("equivalence", "equivalent"),
            }],
        })

    return {
        "resourceType": "ConceptMap",
        "id": cm_id,
        "url": f"http://fhir.verily.com/ConceptMap/{cm_id}",
        "name": _to_pascal(cm_id),
        "title": f"{study_name} {column_name} → {target_system.split('/')[-1]}",
        "status": "active",
        "description": description or f"Maps the physical {column_name} codes in {study_name} data to {target_system.split('/')[-1]}.",
        "sourceUri": f"http://fhir.verily.com/ValueSet/{source_cs_id}",
        "targetUri": target_valueset,
        "group": [{
            "source": f"http://fhir.verily.com/CodeSystem/{source_cs_id}",
            "target": target_system,
            "element": elements,
        }],
    }


# ── Data Profile (MeasureReport) Builder ──────────────────────────────────────

def _metric_group(metric_id: str, metric_code: str, metric_display: str, value, unit: str = None) -> dict:
    """Helper to build a single metric group with measureScore."""
    score: dict = {"value": value}
    if unit:
        score["unit"] = unit
        if unit == "percentage":
            score["system"] = "http://unitsofmeasure.org"
            score["code"] = "%"
        elif unit == "bytes":
            score["system"] = "http://unitsofmeasure.org"
            score["code"] = "By"
    return {
        "id": metric_id,
        "code": {"coding": [{"system": _METRIC_SYSTEM, "code": metric_code, "display": metric_display}]},
        "measureScore": score,
    }


def _freq_distribution_group(
    element_id: str,
    top_values: list[str],
    total_rows: int,
    value_counts: Optional[dict[str, int]] = None,
) -> Optional[dict]:
    """Build a frequency-distribution group with stratifier.stratum entries."""
    if not top_values or total_rows == 0:
        return None

    strata = []
    for i, val in enumerate(top_values[:3]):
        if i >= len(_ORDINAL_CODES):
            break
        if value_counts and val in value_counts:
            pct = round(100.0 * value_counts[val] / total_rows, 1)
        else:
            pct = round(100.0 / len(top_values), 1)

        strata.append({
            "value": {"text": str(val)},
            "component": [{"code": {"coding": [_ORDINAL_CODES[i]]}}],
            "measureScore": {
                "value": pct,
                "unit": "percentage",
                "system": "http://unitsofmeasure.org",
                "code": "%",
            },
        })

    if not strata:
        return None

    return {
        "id": f"{element_id}-freq-distribution",
        "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "freq-distribution", "display": "Frequency Distribution"}]},
        "stratifier": [{"code": [{"text": "mode"}], "stratum": strata}],
    }


def _build_element_profile(
    col: BQColumnInfo,
    col_profile: ColumnProfile,
    total_rows: int,
    columns_meta: list[dict],
) -> dict:
    """
    Build the per-column element group with type-appropriate metrics.

    Metrics produced:
      - All columns: completeness, cardinality
      - String columns: min-length, max-length, avg-length, freq-distribution (if not UID)
      - Numeric columns: min-value, max-value, median, stddev
      - Coded columns: min-length, max-length, avg-length, freq-distribution
    """
    col_name = col.column_name
    short_label = col_name

    # Look up short label and sensitivity from column metadata
    sensitivity = "NONE"
    for cm in columns_meta:
        if cm.get("Column") == col_name:
            short_label = cm.get("Short Label", col_name) or col_name
            sensitivity = str(cm.get("Sensitivity", "")).strip().upper()
            break

    is_uid = sensitivity == "UID"
    bq_type_upper = col.data_type.upper().split("<")[0].strip()
    is_numeric = bq_type_upper in _NUMERIC_BQ_TYPES

    completeness = round(100.0 - col_profile.null_percent, 1) if col_profile.null_percent is not None else 100.0
    cardinality = round(100.0 * col_profile.distinct_count / total_rows, 2) if total_rows > 0 else 0.0

    groups = []
    groups.append(_metric_group(f"{col_name}-completeness", "completeness", "Completeness", completeness, "percentage"))
    groups.append(_metric_group(f"{col_name}-cardinality", "cardinality", "Cardinality", cardinality, "percentage"))

    if is_numeric:
        if col_profile.min_value is not None:
            groups.append(_metric_group(f"{col_name}-min-value", "min-value", "Minimum Value", col_profile.min_value))
        if col_profile.max_value is not None:
            groups.append(_metric_group(f"{col_name}-max-value", "max-value", "Maximum Value", col_profile.max_value))
        if col_profile.median is not None:
            groups.append(_metric_group(f"{col_name}-median", "median", "Median", col_profile.median))
        if col_profile.stddev is not None:
            groups.append(_metric_group(f"{col_name}-stddev", "stddev", "Standard Deviation", col_profile.stddev))
    else:
        if col_profile.min_length is not None:
            groups.append(_metric_group(f"{col_name}-min-length", "min-length", "Minimum Length",
                                        col_profile.min_length, "characters"))
        if col_profile.max_length is not None:
            groups.append(_metric_group(f"{col_name}-max-length", "max-length", "Maximum Length",
                                        col_profile.max_length, "characters"))
        if col_profile.avg_length is not None:
            groups.append(_metric_group(f"{col_name}-avg-length", "avg-length", "Average Length",
                                        col_profile.avg_length, "characters"))

    # Frequency distribution — skip for UID-tagged columns (de-id safety)
    if not is_uid and col_profile.top_values:
        freq_group = _freq_distribution_group(
            col_name, col_profile.top_values, total_rows,
            value_counts=col_profile.value_counts,
        )
        if freq_group:
            groups.append(freq_group)

    return {
        "id": col_name,
        "code": {"coding": [{"system": _ELEMENT_SYSTEM, "code": col_name, "display": short_label}]},
        "group": groups,
    }


def generate_data_profile(
    table_info: BQTableInfo,
    profile_id: str,
    table_profile: Optional[TableProfile] = None,
    columns_meta: Optional[list[dict]] = None,
    sd_title: str = "",
) -> Optional[dict]:
    """
    Generate a FHIR MeasureReport (data profile) JSON.

    If table_profile is provided, produces rich per-column metrics matching the
    gold-standard format. Otherwise, falls back to table-level only.

    No LLM needed — all code-generated.
    """
    if table_info.row_count is None and table_info.size_bytes is None and table_profile is None:
        return None

    iso_date = _get_iso_date()
    row_count = table_profile.total_rows if table_profile else (table_info.row_count or 0)

    mr: dict = {
        "resourceType": "MeasureReport",
        "id": f"{profile_id}-data-profile",
        "meta": {
            "profile": ["http://fhir.verily.com/StructureDefinition/verily-data-profile"],
        },
        "status": "complete",
        "type": "summary",
        "measure": "http://fhir.verily.com/Measure/data-profile-tabular",
        "date": f"{iso_date}T00:00:00Z",
        "subject": {
            "reference": f"StructureDefinition/{profile_id}",
        },
        "period": {
            "start": iso_date,
            "end": iso_date,
        },
        "group": [],
    }

    if sd_title:
        mr["subject"]["display"] = sd_title

    # ── Table-level metrics group ──
    table_metrics: list[dict] = []

    if row_count:
        table_metrics.append(_metric_group("row-count", "row-count", "Row Count", row_count))

    if table_info.size_bytes is not None:
        table_metrics.append(_metric_group("physical-size", "physical-size", "Physical Size",
                                           table_info.size_bytes, "bytes"))

    mr["group"].append({
        "id": "table-metrics",
        "code": {"coding": [{"system": _LEVEL_SYSTEM, "code": "table", "display": "Table Level Metrics"}]},
        "group": table_metrics,
    })

    # ── Element-level metrics group (if profile data available) ──
    if table_profile and table_profile.columns:
        element_groups = []
        for col in table_info.columns:
            cp = table_profile.columns.get(col.column_name)
            if cp:
                element_groups.append(
                    _build_element_profile(col, cp, row_count, columns_meta or [])
                )

        if element_groups:
            mr["group"].append({
                "id": "element-metrics",
                "code": {"coding": [{"system": _LEVEL_SYSTEM, "code": "element", "display": "Element Level Metrics"}]},
                "group": element_groups,
            })

    return mr


# ── Measure Definition (reusable template) ────────────────────────────────────

def build_measure_definition() -> dict:
    """
    Build the shared FHIR Measure resource that defines data profiling metrics.

    This is a static resource — identical for every table. It defines the
    metric vocabulary used by MeasureReport data profiles.

    Matches the gold-standard: data_profile_measure.json
    """
    return {
        "resourceType": "Measure",
        "id": "data-profile-tabular",
        "url": "http://fhir.verily.com/Measure/data-profile-tabular",
        "version": "1.0.0",
        "name": "DataProfileTabular",
        "title": "Data Profile Metrics for Tabular Data",
        "status": "active",
        "date": "2024-01-01",
        "publisher": "Verily",
        "description": (
            "Measure defining data profiling metrics for tabular data including "
            "set metrics (row count, size) and element metrics (completeness, "
            "cardinality, frequency distribution, min/max, etc.)"
        ),
        "group": [
            {
                "id": "table-metrics",
                "code": {"coding": [{"system": _LEVEL_SYSTEM, "code": "table", "display": "Table Level Metrics"}]},
                "group": [
                    {"id": "row-count", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "row-count", "display": "Row Count"}]}, "description": "Total number of rows in the dataset"},
                    {"id": "physical-size", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "physical-size", "display": "Physical Size"}]}, "description": "Size in the current platform storage format"},
                    {"id": "raw-bytes", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "raw-bytes", "display": "Raw Bytes"}]}, "description": "Uncompressed size"},
                ],
            },
            {
                "id": "element-metrics",
                "code": {"coding": [{"system": _LEVEL_SYSTEM, "code": "element", "display": "Element Level Metrics"}]},
                "group": [
                    {"id": "completeness", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "completeness", "display": "Completeness"}]}, "description": "Percentage of non-NULL values for element"},
                    {"id": "cardinality", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "cardinality", "display": "Cardinality"}]}, "description": "Percentage of unique values (PK = 1.0)"},
                    {
                        "id": "freq-distribution",
                        "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "freq-distribution", "display": "Frequency Distribution"}]},
                        "description": "Top X values and percentage (only for adequate sample size to preserve de-id)",
                        "stratifier": [{
                            "code": {"text": "mode"},
                            "description": "Ordinal position of top values (first, second, third)",
                            "component": [
                                {"code": {"coding": [_ORDINAL_CODES[0]]}, "description": "First most frequent value"},
                                {"code": {"coding": [_ORDINAL_CODES[1]]}, "description": "Second most frequent value"},
                                {"code": {"coding": [_ORDINAL_CODES[2]]}, "description": "Third most frequent value"},
                            ],
                        }],
                    },
                    {"id": "min-value", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "min-value", "display": "Minimum Value"}]}, "description": "Minimum value (numeric/date fields only)"},
                    {"id": "max-value", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "max-value", "display": "Maximum Value"}]}, "description": "Maximum value (numeric/date fields only)"},
                    {"id": "median", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "median", "display": "Median"}]}, "description": "Median value (numeric fields only)"},
                    {"id": "stddev", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "stddev", "display": "Standard Deviation"}]}, "description": "Standard deviation (numeric fields only)"},
                    {"id": "min-length", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "min-length", "display": "Minimum Length"}]}, "description": "Minimum string length (string fields only)"},
                    {"id": "max-length", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "max-length", "display": "Maximum Length"}]}, "description": "Maximum string length (string fields only)"},
                    {"id": "avg-length", "code": {"coding": [{"system": _METRIC_SYSTEM, "code": "avg-length", "display": "Average Length"}]}, "description": "Average string length (string fields only)"},
                ],
            },
        ],
    }
