"""
LLM-based FHIR metadata generation.

All functions in this module make LLM calls (Gemini via Vertex AI).
Separated from deterministic builders for clear LLM boundary.

Handles:
  - StructureDefinition generation (full LLM)
  - ValueSet generation (LLM)
  - Column definitions generation (LLM)
  - ConceptMap generation (LLM)
  - Batch generation pipeline
"""

from __future__ import annotations

import json
from typing import Optional

from models import BQTableInfo, StudyConfig, GenerationResult, TableProfile
from bq_profiler import format_bq_schema_for_prompt
from fhir_builder import generate_data_profile
from prompt_engine import call_gemini_fast, extract_json_from_response, extract_multiple_jsons_from_response
from sensitivity import sensitivity_vocabulary_for_prompt


# ══════════════════════════════════════════════════════════════════════════════
# StructureDefinition Generation (full LLM)
# ══════════════════════════════════════════════════════════════════════════════

_STRUCTURE_DEFINITION_SYSTEM_PROMPT = """You are a FHIR metadata specialist. Your task is to generate
a FHIR R4 StructureDefinition JSON (kind: "logical") that describes a BigQuery table.

You will be given:
1. The BigQuery table schema (columns, types, descriptions)
2. Study configuration (study name, compliance zone, etc.)
3. Optionally: a data dictionary with richer column descriptions
4. Optionally: existing FHIR metadata for cross-file reference

## OUTPUT FORMAT
Return EXACTLY ONE valid JSON object in a ```json code block.
The JSON must be a complete FHIR StructureDefinition.

## STRUCTURAL RULES — Every JSON MUST include these elements:

1. **`resourceType`**: `"StructureDefinition"`
2. **`id`**: kebab-case derived from study + dataset + table (e.g., `"bhs-admin-coeval"`)
3. **`meta.security`**: confidentiality code
4. **`extension`** array with:
   - `BigQueryTableSchemaMetadata` → `table-name` as `{study}.{dataset}.{TABLE}`
   - `verily-primary-identity` → the primary key column(s)
   - `verily-structural-link` entries to related tables (if references provided)
5. **`url`**: `"http://fhir.verily.com/StructureDefinition/{id}"`
6. **`name`**: PascalCase (e.g., `"BHSAdminCoeval"`)
7. **`title`**: Human-readable (e.g., `"BHS Admin — COEVAL (Cohort Eligibility)"`)
8. **`status`**: `"active"`
9. **`purpose`**: Granularity — what one record represents
10. **`description`**: Rich description of the table's contents and purpose
11. **`contact`**: domain reference
12. **`useContext`**: 3 entries — compliance-zone, retention-policy, schema-stability
13. **`kind`**: `"logical"`, **`abstract`**: `false`
14. **`type`**: same as `url`
15. **`baseDefinition`**: `"http://hl7.org/fhir/StructureDefinition/Element"`
16. **`derivation`**: `"specialization"`
17. **`mapping`**: vfig identity block
18. **`differential.element`** array — one root element + one per column

## COLUMN ELEMENT RULES — Each column element MUST include:
- `id`: `"{Name}.{column}"` (e.g., `"BHSAdminCoeval.STUDYID"`)
- `path`: same as id
- `short`: brief label
- `definition`: detailed description — rich clinical/business context
- `comment`: include sensitivity note (e.g., "SENSITIVE — PHI" or "NOT SENSITIVE")
- `min`/`max`: `1`/`"1"` for required (NOT NULL), `0`/`"1"` for nullable
- `type`: mapped from BQ type:
  - STRING → `"string"` (or `"code"` if categorical/coded)
  - INTEGER/INT64 → `"integer"`
  - FLOAT/FLOAT64/NUMERIC → `"decimal"`
  - BOOLEAN/BOOL → `"boolean"`
  - DATE → `"date"`
  - TIMESTAMP/DATETIME → `"dateTime"`
- `mapping` to FHIR where applicable (identity: "vfig"):
  - Subject IDs → `Patient.identifier`
  - Visit fields → `Encounter.period` or `Encounter.type`
  - Diagnoses → `Condition.code`
  - Lab results → `Observation.valueQuantity`
  - Demographics → `Patient.gender`, `Patient.birthDate`
  - Medications → `MedicationStatement.medication`
  - Scores/Measures → `Observation.valueInteger` or `Observation.valueQuantity`
- `extension` with `inline-sec-label` for sensitive columns — use the MOST SPECIFIC code from the Verily SensitivityLabels CodeSystem:
  - `UID` for study-assigned subject/participant identifiers
  - `P_BIRTHSEX` for sex-at-birth columns
  - `P_RACEETHNICITY` for race/ethnicity columns
  - `P_DOB` for date of birth, `P_DOD` for date of death
  - `P_MRN` for medical record numbers
  - `P_PNAME` for patient names
  - `P_SSN` for Social Security Numbers
  - `P_STREETADDR`/`P_POSTALCODE`/`P_GEOREGION` for address components
  - `P_EMAIL`/`P_PHONE`/`P_FAX` for contact info
  - `PHI` ONLY as fallback for health data with no more specific code (scores, diagnoses, measurements)
  - `PII` for general personally identifiable info
  - `FINANCIAL` for monetary amounts, billing totals
  - `FREETEXT` for free-text/unstructured text columns
  - No label for non-sensitive metadata (study IDs, site IDs, visit labels)
- `extension` with `verily-attribute-semantic-metadata` for measurement method:
  - `self-reported` for questionnaire responses
  - `calculated` for derived/computed values
  - `laboratory-measured` for lab results
  - `clinician-observed` for vital signs, physical exam
  - `device-collected` for device/wearable data
  - `extracted-from-ehr` for EHR-extracted records
  - `administrative` for system/protocol metadata
- For coded/categorical columns: add `binding` with `strength` and `valueSet` URL

## REFERENCE EXAMPLE — Template Structure:
```json
{
  "resourceType": "StructureDefinition",
  "id": "study-dataset-table",
  "meta": {
    "security": [{"system": "http://terminology.hl7.org/CodeSystem/v3-Confidentiality", "code": "R", "display": "Restricted"}]
  },
  "extension": [
    {
      "url": "http://fhir.verily.com/StructureDefinition/BigQueryTableSchemaMetadata",
      "extension": [{"url": "table-name", "valueString": "study.dataset.TABLE"}]
    },
    {"url": "http://fhir.verily.com/StructureDefinition/verily-primary-identity", "valueString": "PRIMARY_KEY"}
  ],
  "url": "http://fhir.verily.com/StructureDefinition/study-dataset-table",
  "name": "StudyDatasetTable",
  "title": "Study Dataset — TABLE (Description)",
  "status": "active",
  "purpose": "One record per ...",
  "description": "Rich description...",
  "contact": [{"name": "domain:study-domain", "telecom": [{"system": "url", "value": "Organization/study-domain"}]}],
  "useContext": [
    {"code": {"system": "http://fhir.verily.com/CodeSystem/usage-context-type", "code": "compliance-zone"}, "valueCodeableConcept": {"coding": [{"system": "http://fhir.verily.com/CodeSystem/compliance-zones", "code": "HIPAA-covered", "display": "HIPAA Covered"}]}},
    {"code": {"system": "http://fhir.verily.com/CodeSystem/usage-context-type", "code": "retention-policy"}, "valueQuantity": {"value": 7, "unit": "years", "system": "http://unitsofmeasure.org", "code": "a"}},
    {"code": {"system": "http://fhir.verily.com/CodeSystem/usage-context-type", "code": "schema-stability"}, "valueCodeableConcept": {"coding": [{"system": "http://fhir.verily.com/CodeSystem/schema-stability", "code": "stable", "display": "Stable"}]}}
  ],
  "kind": "logical",
  "abstract": false,
  "type": "http://fhir.verily.com/StructureDefinition/study-dataset-table",
  "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Element",
  "derivation": "specialization",
  "mapping": [{"identity": "vfig", "uri": "http://fhir.verily.com/vfig", "name": "Verily FHIR Implementation Guide"}],
  "differential": {
    "element": [
      {"id": "StudyDatasetTable", "path": "StudyDatasetTable", "short": "Table Row", "definition": "One row represents...", "min": 0, "max": "*"},
      {
        "id": "StudyDatasetTable.column_name",
        "path": "StudyDatasetTable.column_name",
        "short": "Column label",
        "definition": "Detailed description...",
        "comment": "SENSITIVE — PHI",
        "min": 1, "max": "1",
        "type": [{"code": "string"}],
        "mapping": [{"identity": "vfig", "map": "Resource.field", "comment": "FHIR mapping note"}],
        "extension": [
          {"url": "http://hl7.org/fhir/uv/security-label-ds4p/StructureDefinition/extension-inline-sec-label", "valueCoding": {"system": "http://fhir.verily.com/CodeSystem/SensitivityLabels", "code": "PHI", "display": "Protected Health Information"}},
          {"url": "http://fhir.verily.com/StructureDefinition/verily-attribute-semantic-metadata", "extension": [{"url": "measurement-method", "valueCodeableConcept": {"coding": [{"system": "http://fhir.verily.com/CodeSystem/measurement-method", "code": "self-reported", "display": "Self-Reported"}], "text": "Description of measurement method"}}]}
        ]
      }
    ]
  }
}
```

## IMPORTANT:
- Generate descriptions that are RICH with clinical/business context, not just restatements of column names
- Infer the primary key from column names (look for ID columns, unique identifiers)
- Infer granularity from the table structure (one row per participant? per visit? per observation?)
- For coded columns (those with a small set of possible values), use type "code" and add binding
- Map EVERY column — do not skip any
- If a data dictionary is provided, use it to enrich descriptions and determine coded values
"""

_STRUCTURE_DEFINITION_USER_TEMPLATE = """Generate a FHIR StructureDefinition JSON for the following BigQuery table.

## Study Configuration:
- Study Name: {study_name}
- Compliance Zone: {compliance_zone}
- Retention Policy: {retention_years} years
- Schema Stability: {schema_stability}
- Domain Contact: {domain_contact}
- Confidentiality: {confidentiality}

## BigQuery Table Schema:
{bq_schema}

{data_dict_section}

{existing_metadata_section}

{structural_links_section}

Generate the complete FHIR StructureDefinition JSON now. Return it in a ```json code block.
"""


def generate_structure_definition(
    table_info: BQTableInfo,
    study_config: StudyConfig,
    project_id: Optional[str] = None,
    data_dict_text: str = "",
    existing_profiles: list[dict] = None,
) -> GenerationResult:
    """
    Generate a FHIR StructureDefinition JSON for a BigQuery table.
    Uses the fast LLM model for generation.
    """
    bq_schema = format_bq_schema_for_prompt(table_info)

    data_dict_section = ""
    if data_dict_text:
        data_dict_section = f"""## Data Dictionary (for enriching descriptions):
{data_dict_text}
"""

    existing_metadata_section = ""
    structural_links_section = ""
    if existing_profiles:
        summaries = []
        link_urls = []
        for profile in existing_profiles:
            pid = profile.get("id", "unknown")
            url = profile.get("url", "")
            title = profile.get("title", "")
            bq_name = ""
            for ext in profile.get("extension", []):
                if "BigQueryTableSchemaMetadata" in ext.get("url", ""):
                    for sub in ext.get("extension", []):
                        if sub.get("url") == "table-name":
                            bq_name = sub.get("valueString", "")
            summaries.append(f"  - {pid}: {title} (table: {bq_name})")
            if url:
                link_urls.append(url)

        existing_metadata_section = f"""## Existing FHIR Metadata (for cross-file consistency):
{chr(10).join(summaries)}
"""
        if link_urls:
            structural_links_section = f"""## Available Structural Link URLs (add relevant links):
{chr(10).join(f'  - {u}' for u in link_urls)}
"""

    user_message = _STRUCTURE_DEFINITION_USER_TEMPLATE.format(
        study_name=study_config.study_name,
        compliance_zone=study_config.compliance_zone,
        retention_years=study_config.retention_years,
        schema_stability=study_config.schema_stability,
        domain_contact=study_config.domain_contact or f"domain:{study_config.study_name.lower()}-research",
        confidentiality=study_config.confidentiality,
        bq_schema=bq_schema,
        data_dict_section=data_dict_section,
        existing_metadata_section=existing_metadata_section,
        structural_links_section=structural_links_section,
    )

    try:
        response = call_gemini_fast(
            system_prompt=_STRUCTURE_DEFINITION_SYSTEM_PROMPT,
            user_message=user_message,
            project_id=project_id,
        )

        json_obj = extract_json_from_response(response)
        if json_obj:
            return GenerationResult(
                table_name=table_info.fq_name,
                structure_definition=json_obj,
            )
        else:
            return GenerationResult(
                table_name=table_info.fq_name,
                error=f"Could not parse JSON from LLM response. Response preview: {response[:500]}",
            )

    except Exception as e:
        return GenerationResult(
            table_name=table_info.fq_name,
            error=f"LLM call failed: {str(e)}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ValueSet Generation (LLM)
# ══════════════════════════════════════════════════════════════════════════════

_VALUE_SET_SYSTEM_PROMPT = """You are a FHIR terminology specialist. Generate FHIR R4 ValueSet JSON files
for coded columns in a StructureDefinition.

For each column that has a `binding` with a `valueSet` URL, create a corresponding ValueSet JSON.

## OUTPUT FORMAT
Return one or more valid JSON objects, each in a separate ```json code block.
Each JSON must be a complete FHIR ValueSet.

## ValueSet Template:
```json
{
  "resourceType": "ValueSet",
  "id": "{kebab-case-id}",
  "url": "http://fhir.verily.com/ValueSet/{kebab-case-id}",
  "version": "1.0.0",
  "name": "{PascalCaseName}",
  "title": "{Human Readable Title}",
  "status": "active",
  "experimental": false,
  "description": "{What this value set represents}",
  "compose": {
    "include": [
      {
        "system": "http://fhir.verily.com/CodeSystem/{kebab-case-id}",
        "concept": [
          {"code": "VALUE_FROM_DATA", "display": "Human Readable Label"}
        ]
      }
    ]
  }
}
```

## RULES:
1. The ValueSet id and URL must match what's referenced in the StructureDefinition
2. Include ALL known values for each coded column
3. If exact values aren't known, provide reasonable defaults based on the column description
4. Use the `code` field for the exact value stored in BigQuery
5. Use the `display` field for a human-readable label
"""

_VALUE_SET_USER_TEMPLATE = """Generate ValueSet JSON files for the coded columns in this StructureDefinition.

## StructureDefinition:
```json
{structure_definition_json}
```

{data_dict_section}

Generate a separate ValueSet JSON for each column that has a binding.valueSet reference.
Return each ValueSet in its own ```json code block.
"""


def generate_value_sets(
    structure_definition: dict,
    project_id: Optional[str] = None,
    data_dict_text: str = "",
) -> list[dict]:
    """Generate FHIR ValueSet JSONs for coded columns in a StructureDefinition."""
    has_bindings = False
    for element in structure_definition.get("differential", {}).get("element", []):
        if element.get("binding", {}).get("valueSet"):
            has_bindings = True
            break

    if not has_bindings:
        return []

    data_dict_section = ""
    if data_dict_text:
        data_dict_section = f"""## Data Dictionary (for determining coded values):
{data_dict_text}
"""

    user_message = _VALUE_SET_USER_TEMPLATE.format(
        structure_definition_json=json.dumps(structure_definition, indent=2),
        data_dict_section=data_dict_section,
    )

    try:
        response = call_gemini_fast(
            system_prompt=_VALUE_SET_SYSTEM_PROMPT,
            user_message=user_message,
            project_id=project_id,
        )
        return extract_multiple_jsons_from_response(response)
    except Exception as e:
        print(f"⚠ ValueSet generation failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Column Definitions Generation (LLM)
# ══════════════════════════════════════════════════════════════════════════════

_COLUMN_DEFS_SYSTEM_PROMPT = """You are a FHIR metadata specialist. Analyze a BigQuery table and generate
column-level metadata and table-level summary.

For the TABLE, provide:
- title: Human-readable title (e.g., "CDC Natality — Birth Records 2023")
- description: Rich description of what this table contains (2-4 sentences)
- purpose: What one record represents (e.g., "One record per live birth occurrence")
- primary_key: The column(s) that uniquely identify a row. Format: "COL" (single) or "COL1 + COL2" (composite, space-plus-space separator)

For each COLUMN, provide:
- column_name: exact column name (must match input exactly)
- fhir_type: FHIR R4 type — one of: string, code, integer, decimal, boolean, date, dateTime
  Use "code" for categorical/coded columns (low distinct count or known coded values)
- short_label: Brief 2-5 word human-readable label
- description: Rich clinical/business description (1-3 sentences). Don't just restate the column name.
- sensitivity: A code from the Verily SensitivityLabels CodeSystem (see vocabulary below), or "NONE"
- measurement_method: one of: self-reported, calculated, laboratory-measured, clinician-observed, device-collected, extracted-from-ehr, administrative
- fhir_mapping: Semantic FHIR R4 resource.field path. This is a documentation mapping showing which FHIR resource field this column's concept corresponds to. Use the guidance table below. Only leave empty for fixed-value constants.
- comment: A rich element comment (1-3 sentences). Include:
  * For coded columns: whether this is a "Fixed Concept Binding" (same concept always) or "Code System Binding" (different codes per row)
  * For PK columns: mention the primary key
  * Sensitivity justification (why this IS or IS NOT sensitive)
  * Any important data semantics (binary indicator, fixed value, etc.)
- mapping_comment: A one-sentence explanation of the FHIR mapping (e.g., "Maps to FHIR Patient.identifier — study-specific subject ID"). Empty string if no fhir_mapping.
- measurement_text: A human-readable description of the measurement method (e.g., "Self-reported by participant during enrollment"). More specific than the code alone. Empty string if no measurement_method.
- fixed_value: If this column has exactly one distinct value (constant across all rows), put that value here (e.g., "BL001"). Otherwise empty string.

{sensitivity_vocabulary}

Rules:
- Use profiling data (null %, distinct count, top values) to decide if a column is coded
- Columns with low distinct count and known value sets are coded
- **NOT coded**: Temporal columns (dates, years, timestamps) are NEVER coded even if they have low distinct counts. Years are ordinal time periods, not categorical codes. Use fhir_type "date", "dateTime", or "string" — NOT "code".
- **NOT coded**: Numeric columns (counts, averages, measurements) are NEVER coded. Use fhir_type "integer" or "decimal".
- **NOT coded**: High-cardinality identifier columns (FIPS codes, ZIP codes, IDs with many distinct values) are NOT coded — they are identifiers, not category codes.
- ALWAYS use the MOST SPECIFIC sensitivity code — e.g., P_BIRTHSEX for sex-at-birth, P_DOB for date of birth, P_RACEETHNICITY for race/ethnicity, P_MRN for medical record numbers
- Use UID for study-assigned subject/participant identifiers
- Use PHI only as a fallback for clinical/health data with no more specific code
- Use FREETEXT for free-text columns (clinical notes, open survey responses)
- Use FINANCIAL for monetary/billing amounts
- Use NONE for non-sensitive columns (administrative flags, reporting flags, study metadata)
- Reporting-flag columns (F_*) are administrative → NONE
- If a data dictionary is provided, use it heavily to enrich descriptions
- **AGGREGATED DATA — SENSITIVITY**: Sensitivity labels ONLY apply when each row represents an individual person (patient-level or subject-level data). If the table is aggregated (e.g., one row per county per year, population statistics, summary counts/averages), use sensitivity "NONE" for ALL columns. Geographic identifiers in aggregated data (county names, FIPS codes) are NOT sensitive — they identify places, not people. Do NOT apply P_GEOREGION, P_DOB, P_BIRTHSEX, or any other sensitivity label to aggregated data.
- **measurement_method**: MUST be exactly one of the 7 allowed values listed above. Do NOT use FHIR resource paths, data locations, or any other values.
- **description**: MUST be a rich 1-3 sentence semantic description. Do NOT use single words, column metadata values ("Yes"/"No"/"calculated"), or BQ type echoes.
- **comment**: MUST include binding type context for coded columns (Fixed Concept Binding vs Code System Binding)
- **comment**: MUST include sensitivity justification (why labeled or not)
- **mapping_comment**: Only provide when fhir_mapping is non-empty
- **fixed_value**: Use profiling data — if distinct count is 1, the fixed value is the single top value
- **fhir_mapping**: Map EVERY column where a reasonable FHIR equivalent exists. These are purely semantic/documentation mappings (not ETL transforms). Use this guidance:
  * Subject/participant IDs → Patient.identifier
  * Site/location IDs → Location.identifier
  * Location/county/region names → Location.name
  * Geographic codes (FIPS, ZIP, postal) → Location.identifier
  * Visit labels/types → Encounter.type
  * Visit numbers, visit dates → Encounter.period
  * Temporal periods, reporting years → Observation.effectivePeriod
  * Sex/gender → Patient.gender
  * Age, weight, BMI, lab values → Observation.valueQuantity
  * Counts (births, events, occurrences) → Observation.valueInteger
  * Diagnoses/conditions → Condition.code
  * Medications → MedicationStatement.medication
  * Scores, computed statistics, averages → Observation.valueQuantity
  * Binary indicators (yes/no flags) → Observation.valueBoolean
  * Free-text clinical notes → DocumentReference.content
  * Only leave fhir_mapping empty for fixed-value constants (e.g., a study ID that never varies)
- **fhir_mapping for aggregate data**: For population-level statistics (averages, rates, counts), map to Observation.valueQuantity or Observation.valueInteger. Geographic identifiers map to Location.identifier or Location.name.

Return ONLY a JSON object in a ```json block:
```json
{{
  "table": {{"title": "...", "description": "...", "purpose": "...", "primary_key": "COL1 + COL2"}},
  "columns": [
    {{
      "column_name": "SITEID",
      "fhir_type": "string",
      "short_label": "Site ID",
      "description": "Clinical site identifier — a code identifying where the participant was enrolled.",
      "sensitivity": "NONE",
      "measurement_method": "administrative",
      "fhir_mapping": "Location.identifier",
      "comment": "NOT SENSITIVE — site codes alone do not identify individuals. No security label applied.",
      "mapping_comment": "Maps to FHIR Location.identifier — the clinical site where the participant was enrolled",
      "measurement_text": "Extracted from administrative enrollment records",
      "fixed_value": ""
    }}
  ]
}}
```
""".format(sensitivity_vocabulary=sensitivity_vocabulary_for_prompt())

_COLUMN_DEFS_USER_TEMPLATE = """Generate metadata for this BigQuery table.

## Study: {study_name}

## Table Schema:
{bq_schema}

## Data Profile:
Total rows: {total_rows:,}
{profile_summary}

{data_dict_section}

{data_level_context}

Return the JSON object with "table" and "columns" keys. Include ALL {col_count} columns.
"""


def generate_column_definitions(
    table_info: BQTableInfo,
    table_profile: TableProfile,
    study_config: StudyConfig,
    project_id: Optional[str] = None,
    data_dict_text: str = "",
) -> Optional[dict]:
    """
    Use LLM to generate column-level metadata (descriptions, sensitivity, etc.).
    Returns dict with "table" and "columns" keys, or None on failure.
    """
    bq_schema = format_bq_schema_for_prompt(table_info)

    prof_lines = []
    for col in table_info.columns:
        cp = table_profile.columns.get(col.column_name)
        if not cp:
            continue
        coded_note = ""
        if cp.top_values:
            vals_preview = ", ".join(cp.top_values[:10])
            if len(cp.top_values) > 10:
                vals_preview += f" ... ({len(cp.top_values)} total)"
            coded_note = f"  Values: [{vals_preview}]"
        prof_lines.append(
            f"  {col.column_name}: null={cp.null_percent}%, distinct={cp.distinct_count}{coded_note}"
        )

    data_dict_section = ""
    if data_dict_text:
        data_dict_section = f"## Data Dictionary:\n{data_dict_text}"

    # Detect aggregate data — look for telltale patterns in column names and schema
    # Use word-boundary split to avoid false positives (e.g., "county" ≠ "count")
    import re as _re
    _agg_indicators = {"avg", "ave", "average", "mean", "count", "sum", "total", "rate", "percent", "pct"}

    def _has_agg_token(col_name: str) -> bool:
        tokens = set(_re.split(r"[^a-z0-9]+", col_name.lower()))
        return bool(tokens & _agg_indicators)

    agg_col_count = sum(1 for col in table_info.columns if _has_agg_token(col.column_name))
    is_likely_aggregate = agg_col_count >= 2

    data_level_context = ""
    if is_likely_aggregate:
        data_level_context = (
            "## ⚠ Data Level Context\n"
            "This table appears to contain **AGGREGATED / population-level data** "
            "(multiple columns with averages, counts, or rates). "
            "It is NOT individual patient/subject records. "
            "Apply sensitivity labels ONLY if the aggregation granularity is fine "
            "enough to risk re-identification. Be consistent across all columns at "
            "the same granularity level."
        )

    user_msg = _COLUMN_DEFS_USER_TEMPLATE.format(
        study_name=study_config.study_name,
        bq_schema=bq_schema,
        total_rows=table_profile.total_rows,
        profile_summary="\n".join(prof_lines),
        data_dict_section=data_dict_section,
        data_level_context=data_level_context,
        col_count=len(table_info.columns),
    )

    try:
        response = call_gemini_fast(
            system_prompt=_COLUMN_DEFS_SYSTEM_PROMPT,
            user_message=user_msg,
            project_id=project_id,
        )
        result = extract_json_from_response(response)
        if result and "columns" in result:
            return result
        print(f"⚠ LLM response missing 'columns' key. Preview: {response[:300]}")
        return None
    except Exception as e:
        print(f"⚠ Column definition generation failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ConceptMap Generation (LLM)
# ══════════════════════════════════════════════════════════════════════════════

_CONCEPT_MAP_SYSTEM_PROMPT = """You are a FHIR terminology specialist. Your task is to map physical coded values from a study database to standard terminologies.

For each coded column, determine:
1. Whether a suitable standard terminology exists
2. The exact target CodeSystem URL and ValueSet URL
3. The mapping from each physical code to the standard code
4. A confidence score (high / medium / low) for the overall mapping

## Standard Terminology Priority (prefer in this order):
- **SNOMED CT** (`http://snomed.info/sct`) — Clinical findings, body structures, organisms, procedures, substances. Use when values represent clinical concepts (e.g., diagnoses, symptoms, anatomical sites).
- **LOINC** (`http://loinc.org`) — Laboratory observations, vital signs, survey instruments, clinical documents. Use when values represent observation types or lab test names.
- **ICD-10-CM** (`http://hl7.org/fhir/sid/icd-10-cm`) — Diagnosis codes. Use when values look like ICD codes or diagnosis classifications.
- **HL7 FHIR CodeSystems** — For administrative/demographic concepts:
  - `http://hl7.org/fhir/administrative-gender` for sex/gender (male, female, other, unknown)
  - `http://terminology.hl7.org/CodeSystem/v2-0136` for yes/no
  - `http://hl7.org/fhir/encounter-status` for visit/encounter status
  - `http://terminology.hl7.org/CodeSystem/v3-MaritalStatus` for marital status
- **CDC Race & Ethnicity** (`urn:oid:2.16.840.1.113883.6.238`) — race/ethnicity codes
- **CPT** (`http://www.ama-assn.org/go/cpt`) — procedure codes
- **RxNorm** (`http://www.nlm.nih.gov/research/umls/rxnorm`) — medication codes

## Rules:
- Only produce a ConceptMap when a clear standard terminology mapping exists
- Use equivalence values: "equivalent", "wider", "narrower", "inexact", or "unmatched"
- If no standard terminology applies, return null for that column
- Use human-readable display values for both source and target concepts
- For SNOMED codes, always include the numeric SNOMED CT code (e.g., "263495000" for "Female")
- For LOINC codes, include the LOINC number (e.g., "8480-6" for "Systolic blood pressure")
- Set confidence to "high" when the mapping is well-established (gender, yes/no, ICD codes)
- Set confidence to "medium" when the mapping is likely but values are abbreviated or ambiguous
- Set confidence to "low" when the mapping is a best guess

Return a JSON array of objects (one per column that has a mapping).
Return an empty array [] if no columns have standard terminology mappings.

```json
[
  {
    "column_name": "sex",
    "confidence": "high",
    "target_system_name": "HL7 Administrative Gender",
    "concept_map": {
      "resourceType": "ConceptMap",
      "id": "...",
      "url": "...",
      "name": "...",
      "title": "...",
      "status": "active",
      "description": "...",
      "sourceUri": "...",
      "targetUri": "...",
      "group": [
        {
          "source": "...",
          "target": "...",
          "element": [
            {
              "code": "F",
              "display": "Female",
              "target": [
                {
                  "code": "female",
                  "display": "Female",
                  "equivalence": "equivalent"
                }
              ]
            }
          ]
        }
      ]
    }
  }
]
```
"""

_CONCEPT_MAP_USER_TEMPLATE = """Map the following coded columns from the {study_name} study to standard terminologies.

## Study: {study_name}

## Coded Columns:
{columns_section}

For each column, identify the best standard terminology and map each physical code to the corresponding standard code.
Return a JSON array of objects with "column_name" and "concept_map" keys.
"""


def generate_concept_maps(
    study_name: str,
    coded_columns: list[dict],
    project_id: Optional[str] = None,
) -> list[dict]:
    """
    Use LLM to generate ConceptMaps for coded columns that map to standard terminologies.

    Args:
        study_name: Study name for naming.
        coded_columns: List of dicts with keys: column_name, coded_values, description.
        project_id: GCP project for Vertex AI billing.

    Returns:
        List of ConceptMap dicts (may be empty if no standard mappings found).
    """
    if not coded_columns:
        return []

    columns_section_parts = []
    for col in coded_columns:
        col_name = col.get("column_name", "")
        values = col.get("coded_values", [])
        desc = col.get("description", "")
        values_str = ", ".join(f'"{v}"' for v in values[:25])
        columns_section_parts.append(
            f"### {col_name}\n"
            f"- Description: {desc}\n"
            f"- Values ({len(values)} distinct): [{values_str}]"
        )

    user_message = _CONCEPT_MAP_USER_TEMPLATE.format(
        study_name=study_name,
        columns_section="\n\n".join(columns_section_parts),
    )

    try:
        response = call_gemini_fast(
            system_prompt=_CONCEPT_MAP_SYSTEM_PROMPT,
            user_message=user_message,
            project_id=project_id,
        )
        return _parse_concept_map_response(response)
    except Exception as e:
        print(f"⚠ ConceptMap generation failed: {e}")
        return []


def _parse_concept_map_response(response: str) -> list[dict]:
    """
    Parse the LLM response into a list of enriched ConceptMap dicts.

    Each returned dict has keys:
      - All standard ConceptMap fields (resourceType, id, group, etc.)
      - _column_name: source column name
      - _confidence: "high" / "medium" / "low"
      - _target_system_name: human-readable target system name
    """
    raw = extract_json_from_response(response)
    if raw is None:
        return []

    if isinstance(raw, list):
        result = []
        for item in raw:
            cm = item.get("concept_map") if isinstance(item, dict) else None
            if cm and cm.get("resourceType") == "ConceptMap":
                cm["_column_name"] = item.get("column_name", "")
                cm["_confidence"] = item.get("confidence", "medium")
                cm["_target_system_name"] = item.get("target_system_name", "")
                result.append(cm)
        return result

    if isinstance(raw, dict) and raw.get("resourceType") == "ConceptMap":
        return [raw]

    return []


# ══════════════════════════════════════════════════════════════════════════════
# Batch Generation Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _generate_single_table(
    table_info: BQTableInfo,
    study_config: StudyConfig,
    project_id: Optional[str],
    data_dict_text: str,
    existing_profiles: list[dict],
) -> GenerationResult:
    """Generate all metadata for a single table (called in parallel)."""
    import time
    start = time.time()
    table_name = table_info.fq_name
    print(f"  ⏳ [{table_name}] Starting StructureDefinition generation...")

    result = generate_structure_definition(
        table_info=table_info,
        study_config=study_config,
        project_id=project_id,
        data_dict_text=data_dict_text,
        existing_profiles=existing_profiles,
    )

    sd_time = time.time() - start
    if result.success:
        print(f"  ✅ [{table_name}] StructureDefinition done in {sd_time:.1f}s — generating ValueSets...")

        result.value_sets = generate_value_sets(
            structure_definition=result.structure_definition,
            project_id=project_id,
            data_dict_text=data_dict_text,
        )

        profile_id = result.structure_definition.get("id", "")
        result.data_profile = generate_data_profile(table_info, profile_id)

        total_time = time.time() - start
        print(f"  ✅ [{table_name}] All done in {total_time:.1f}s ({len(result.value_sets)} ValueSets)")
    else:
        print(f"  ❌ [{table_name}] Failed after {sd_time:.1f}s: {result.error}")

    return result


def generate_all_metadata(
    tables: list[BQTableInfo],
    study_config: StudyConfig,
    project_id: Optional[str] = None,
    data_dict_text: str = "",
    existing_profiles: list[dict] = None,
    progress_callback=None,
    max_workers: int = 4,
) -> list[GenerationResult]:
    """Generate FHIR metadata for multiple tables (parallel batch mode)."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(tables)
    all_profiles = list(existing_profiles or [])

    print(f"\n🚀 Generating metadata for {total} table(s) with up to {max_workers} parallel workers...\n")
    start = time.time()

    results_map: dict[str, GenerationResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_table = {
            executor.submit(
                _generate_single_table,
                table_info,
                study_config,
                project_id,
                data_dict_text,
                all_profiles,
            ): table_info
            for table_info in tables
        }

        completed = 0
        for future in as_completed(future_to_table):
            table_info = future_to_table[future]
            completed += 1
            try:
                result = future.result()
                results_map[table_info.fq_name] = result
            except Exception as e:
                results_map[table_info.fq_name] = GenerationResult(
                    table_name=table_info.fq_name,
                    error=f"Unexpected error: {str(e)}",
                )
            if progress_callback:
                status = "done" if results_map[table_info.fq_name].success else "error"
                progress_callback(completed, total, table_info.fq_name, status)

    results = [results_map[t.fq_name] for t in tables]

    elapsed = time.time() - start
    success_count = sum(1 for r in results if r.success)
    print(f"\n✅ Generation complete: {success_count}/{total} succeeded in {elapsed:.1f}s total\n")

    return results


def summarize_generation_results(results: list[GenerationResult]) -> str:
    """Format generation results for display."""
    lines = []
    success_count = sum(1 for r in results if r.success)
    lines.append(f"Generated {success_count}/{len(results)} StructureDefinitions")
    lines.append("")

    for r in results:
        if r.success:
            sd = r.structure_definition
            col_count = len([
                e for e in sd.get("differential", {}).get("element", [])
                if "." in e.get("path", "")
            ])
            vs_count = len(r.value_sets)
            dp = "✓" if r.data_profile else "—"
            lines.append(f"  ✅ {r.table_name}: {col_count} columns, {vs_count} ValueSets, DataProfile: {dp}")
        else:
            lines.append(f"  ❌ {r.table_name}: {r.error}")

    return "\n".join(lines)
