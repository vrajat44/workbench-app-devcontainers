"""
Semantic Profiler — LLM-driven field definitions, terminology bindings,
PHI/PII classification, join paths, confidence scoring, primary key detection,
unit of measure, granularity definition, and two-tier semantic domain.

Consumes TechTableProfile output + optional context signals.
Includes self-validation via LLM-as-Judge and cross-check.
Non-applicable semantics are surfaced as warnings, not failures.
"""

from __future__ import annotations

from typing import Optional

from verily_profiler.models import (
    ALLOWED_SENSITIVITY_CODES,
    SENSITIVITY_CODES,
    STANDARD_SYSTEMS,
    PrimaryKeyInfo,
    SemanticColumnProfile,
    SemanticDomain,
    SemanticTableProfile,
    SemanticValidationResult,
    TechTableProfile,
    TermEntry,
    TerminologyBinding,
    TerminologyRegistry,
    _slugify,
)
from verily_profiler.llm import call_gemini, extract_json_from_response
from verily_profiler.terminology_domains import build_terminology_prompt_section


# ── Fixed domain taxonomy ─────────────────────────────────────────────────────

DOMAIN_TAXONOMY = [
    "Clinical / EHR",
    "Genomics / Omics",
    "Claims / Billing",
    "Demographics",
    "Social Determinants of Health",
    "Research / Clinical Trials",
    "Administrative / Operations",
    "Imaging / Radiology",
    "Public Health / Epidemiology",
    "Geospatial",
    "Financial",
    "IoT / Wearables / Device",
    "Pharmacy / Medication",
    "Laboratory",
    "Survey / Patient-Reported",
    "General / Other",
]

_TAXONOMY_LIST = "\n".join(f"  - {d}" for d in DOMAIN_TAXONOMY)


# ── System Prompts ────────────────────────────────────────────────────────────

def _build_system_prompt(terminology_domains: list[str] | None = None) -> str:
    """Build the semantic profiling system prompt with dynamic terminology rules."""
    term_section = build_terminology_prompt_section(terminology_domains)
    return f"""\
You are a data governance and metadata specialist. You analyze BigQuery table
schemas and technical profiling data to generate rich semantic metadata.

Return a single JSON object with these top-level fields:

1. "business_name": a short, human-friendly name for the table
2. "table_definition": 2-3 sentence plain-language description of what this table
   contains, its purpose, and how it relates to the broader dataset
3. "primary_key": an object describing the table's primary key:
   - "columns": array of column name(s) forming the primary key
   - "type": "single" (one column), "composite" (multiple columns), or "none"
   - "confidence": "high", "medium", or "low"
   Use the technical stats to identify PKs: columns with distinct_count equal to
   row_count and 0% nulls are strong single-key candidates (especially if flagged
   as unique_key_candidate). For composite keys, reason about column naming
   conventions (e.g. patient_id + visit_date) and business context. If no PK can
   be determined, set type to "none" and columns to [].
4. "granularity": a plain-English sentence describing what a single record in
   this table represents (e.g. "One observation per patient per day",
   "One row per insurance claim", "One record per gene variant").
5. "semantic_domain": an object with:
   - "primary": EXACTLY ONE value from this fixed taxonomy:
{_TAXONOMY_LIST}
   - "sub_domain": a free-text string providing more specificity
     (e.g. "Oncology Pathology Reports", "Blood Chemistry Panels")
6. "columns": a JSON array where each element has:
   - "column_name": exact column name (must match input)
   - "definition": plain-language description (2-3 sentences)
   - "terminology_bindings": array of objects with "system", "code", "display"

{term_section}

   - "sensitivity": a code from the SENSITIVITY VALUE SET below, or "" if not sensitive.
       Pick the MOST SPECIFIC code that applies. Use "" for columns with no privacy concern.

       SENSITIVITY VALUE SET (system: urn:verily:sensitivity):
""" + "\n".join(f'       {code}: {info["display"]} — {info["definition"]}' for code, info in SENSITIVITY_CODES.items()) + """
   - "join_paths": array of "likely_table.likely_column" strings, or []
   - "confidence": "high", "medium", or "low"
   - "unit_of_measure": the measurement unit for this column if applicable
       (e.g. "mg/dL", "kg", "years", "USD", "count", "percentage").
       Set to "unitless" for true dimensionless quantities: ratios, log-scale
       values, indices, scores, boolean flags, or counts that are not
       measurements (e.g. a "rule of 5 violations" count is unitless; a
       "white blood cell count per µL" has a unit).
       Set to "" for non-measurement columns (identifiers, names, codes,
       timestamps). Do NOT set "unitless" on columns that have real physical
       units. Do NOT flag missing units as an error.
   - "measurement_method": how the data in this column was captured or derived.
       Use EXACTLY ONE of these values:
         "self-reported"       — patient/participant-reported data (surveys, questionnaires, medical history)
         "clinician-reported"  — recorded by a clinician during a visit or assessment
         "lab-measured"        — from a laboratory test or assay
         "device-collected"    — from a wearable, sensor, or medical device
         "derived"             — calculated or derived from other columns (flags, scores, aggregations)
         "administrative"      — system-generated identifiers, timestamps, or operational data
       Set to "" (empty string) if the capture method cannot be determined.
       IMPORTANT: Look at column naming patterns for clues:
         - "der_" or "derived_" prefix → "derived"
         - "mh_" or "medical_history" prefix → "self-reported" (medical history is typically patient-reported)
         - score columns computed from subscales → "derived"
         - raw survey/questionnaire items → "self-reported"

IMPORTANT: Fields that are not applicable should be set to their empty/default
values (empty string, empty array). This is expected behavior, not an error.

Return ONLY a JSON object (no markdown, no explanation outside the JSON).
"""

_JUDGE_SYSTEM_PROMPT = """\
You are a metadata quality reviewer. You check AI-generated semantic metadata
for plausibility and correctness.

For each column, assess:
1. Is the definition accurate and specific enough?
2. Are terminology bindings appropriate (correct system, code, display)?
3. Is the sensitivity classification correct?
4. Are join path suggestions reasonable?
5. Is the confidence rating appropriate?
6. Is the unit_of_measure correct for measurement columns?

Also assess table-level metadata:
7. Is the primary_key identification reasonable given the stats?
8. Is the granularity description accurate?
9. Is the semantic_domain assignment appropriate?

Return a JSON object with:
- "status": "pass", "warning", or "fail"
    - "pass": all metadata is correct and complete
    - "warning": metadata is mostly correct but some fields could not be
      determined or are generic (e.g. "General / Other" domain, no PK found).
      This is NOT a failure — it means the data doesn't have those properties.
    - "fail": there are actual errors in the metadata
- "issues": array of strings describing real errors found (empty if none)
- "warnings": array of strings describing non-applicable or generic fields
    (e.g. "No primary key could be determined", "Unit of measure not applicable
    for most columns"). These are informational, not errors.
"""


# ── Main Entry Point ──────────────────────────────────────────────────────────

def profile_semantic(
    tech_profile: TechTableProfile,
    model: str,
    project_id: Optional[str] = None,
    context_text: Optional[str] = None,
    run_judge: bool = False,
    registry: Optional[TerminologyRegistry] = None,
    terminology_domains: Optional[list[str]] = None,
) -> tuple[SemanticTableProfile, list[TermEntry]]:
    """
    Generate semantic profiles for all columns in a table.

    Args:
        tech_profile: Technical profile output for the table.
        model: Gemini model name to use.
        project_id: GCP project for Vertex AI billing.
        context_text: Optional context (data dictionary, schema.yml, etc.).
        run_judge: Whether to run the LLM-as-Judge validation pass.
        registry: Existing terminology registry for reuse. If None, no
                  registry context is injected.
        terminology_domains: List of domain keys (e.g. ["clinical", "life_sciences"])
                  to include in the prompt. If None, all domains are used.

    Returns:
        Tuple of (SemanticTableProfile, list of new TermEntry objects to
        upsert into the registry).
    """
    sem_profile = SemanticTableProfile(
        table_name=tech_profile.table_name,
        model_used=model,
    )

    user_msg = _build_profiling_prompt(tech_profile, context_text, registry)
    system_prompt = _build_system_prompt(terminology_domains)

    try:
        response = call_gemini(
            system_prompt=system_prompt,
            user_message=user_msg,
            model_name=model,
            project_id=project_id,
            temperature=0.1,
        )
        raw = extract_json_from_response(response)
    except Exception as e:
        print(f"  Semantic profiling LLM call failed: {e}")
        for cp in tech_profile.columns:
            sem_profile.columns.append(SemanticColumnProfile(
                column_name=cp.column_name,
                definition=f"[LLM error: {e}]",
                confidence="low",
            ))
        sem_profile.validation = SemanticValidationResult(
            status="fail", issues=[f"LLM call failed: {e}"]
        )
        return sem_profile, []

    table_meta, col_map = _parse_llm_output(raw, tech_profile)

    sem_profile.business_name = table_meta.get("business_name", "")
    sem_profile.table_definition = table_meta.get("table_definition", "")
    sem_profile.granularity = table_meta.get("granularity", "")
    sem_profile.primary_key = _parse_primary_key(table_meta.get("primary_key"))
    sem_profile.semantic_domain = _parse_semantic_domain(table_meta.get("semantic_domain"))

    tech_col_names = {cp.column_name for cp in tech_profile.columns}
    for cp in tech_profile.columns:
        llm_data = col_map.get(cp.column_name, {})
        sem_col = _build_semantic_column(cp.column_name, llm_data)
        sem_profile.columns.append(sem_col)

    cross_issues: list[str] = []
    for name in col_map:
        if name not in tech_col_names:
            cross_issues.append(f"LLM referenced non-existent column: {name}")

    applicability_warnings = _collect_applicability_warnings(sem_profile, tech_profile)

    if run_judge:
        judge_result = _run_judge_validation(sem_profile, tech_profile, model, project_id)
        all_issues = cross_issues + judge_result.issues
        all_warnings = judge_result.warnings + applicability_warnings
    else:
        all_issues = cross_issues
        all_warnings = applicability_warnings

    if all_issues:
        status = "fail"
    elif all_warnings:
        status = "warning"
    else:
        status = "pass"

    sem_profile.validation = SemanticValidationResult(
        status=status,
        issues=all_issues,
        warnings=all_warnings,
    )

    new_entries = _extract_term_entries(sem_profile)
    return sem_profile, new_entries


def revalidate_semantic(
    sem_profile: SemanticTableProfile,
    tech_profile: TechTableProfile,
    model_name: str,
    project_id: Optional[str] = None,
) -> SemanticValidationResult:
    """Re-run LLM-as-Judge validation on an edited semantic profile."""
    return _run_judge_validation(sem_profile, tech_profile, model_name, project_id)


# ── Prompt Construction ───────────────────────────────────────────────────────

def _build_profiling_prompt(
    tech_profile: TechTableProfile,
    context_text: Optional[str],
    registry: Optional[TerminologyRegistry] = None,
) -> str:
    lines = [
        f"Table: {tech_profile.table_name}",
        f"Row count: {tech_profile.row_count:,}",
        "",
        "Column details (from technical profiling):",
    ]

    for cp in tech_profile.columns:
        parts = [f"  - {cp.column_name} ({cp.data_type})"]
        parts.append(f"nulls={cp.null_percent}%")
        parts.append(f"distinct={cp.distinct_count}")
        if cp.top_values:
            preview = ", ".join(cp.top_values[:5])
            parts.append(f"top_values=[{preview}]")
        if cp.detected_pattern:
            parts.append(f"pattern={cp.detected_pattern}")
        if cp.min_value is not None:
            parts.append(f"range=[{cp.min_value}, {cp.max_value}]")
        if cp.min_length is not None:
            parts.append(f"strlen=[{cp.min_length}, {cp.max_length}]")
        if cp.anomalies:
            parts.append(f"anomalies={cp.anomalies}")
        lines.append(" | ".join(parts))

    if registry and registry.entries:
        lines.append("")
        lines.append(registry.format_for_prompt(max_entries=200))

    if context_text:
        lines.append("")
        lines.append("Additional context (data dictionary / schema / protocol):")
        lines.append(context_text[:8000])

    return "\n".join(lines)


# ── LLM Output Parsing ───────────────────────────────────────────────────────

def _parse_llm_output(raw, tech_profile):
    if raw is None:
        return {}, {}
    table_meta = {}
    if isinstance(raw, dict) and "columns" in raw:
        table_meta = {
            "business_name": raw.get("business_name", ""),
            "table_definition": raw.get("table_definition", ""),
            "granularity": raw.get("granularity", ""),
            "primary_key": raw.get("primary_key"),
            "semantic_domain": raw.get("semantic_domain"),
        }
        items = raw["columns"] if isinstance(raw["columns"], list) else []
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    col_map: dict[str, dict] = {}
    for item in items:
        if isinstance(item, dict) and "column_name" in item:
            col_map[item["column_name"]] = item
    return table_meta, col_map


def _parse_primary_key(raw):
    if not isinstance(raw, dict):
        return PrimaryKeyInfo()
    columns = raw.get("columns", [])
    if not isinstance(columns, list):
        columns = []
    pk_type = str(raw.get("type", "none")).strip().lower()
    if pk_type not in ("single", "composite", "none"):
        pk_type = "none"
    confidence = str(raw.get("confidence", "low")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return PrimaryKeyInfo(columns=[str(c) for c in columns], pk_type=pk_type, confidence=confidence)


def _parse_semantic_domain(raw):
    if not isinstance(raw, dict):
        return SemanticDomain()
    primary = str(raw.get("primary", "")).strip()
    if primary not in DOMAIN_TAXONOMY:
        primary = "General / Other"
    sub_domain = str(raw.get("sub_domain", "")).strip()
    return SemanticDomain(primary=primary, sub_domain=sub_domain)


def _build_semantic_column(column_name, llm_data):
    bindings = []
    for tb in llm_data.get("terminology_bindings", []):
        if isinstance(tb, dict) and "system" in tb and "code" in tb:
            bindings.append(TerminologyBinding(system=tb["system"], code=str(tb["code"]), display=tb.get("display", "")))
    sensitivity = str(llm_data.get("sensitivity", "")).strip().upper()
    if sensitivity not in ALLOWED_SENSITIVITY_CODES:
        sensitivity = ""
    confidence = str(llm_data.get("confidence", "medium")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"
    join_paths = llm_data.get("join_paths", [])
    if not isinstance(join_paths, list):
        join_paths = []
    unit_of_measure = str(llm_data.get("unit_of_measure", "")).strip()
    valid_methods = {"self-reported", "clinician-reported", "lab-measured", "device-collected", "derived", "administrative"}
    measurement_method = str(llm_data.get("measurement_method", "")).strip().lower()
    if measurement_method not in valid_methods:
        measurement_method = ""
    return SemanticColumnProfile(
        column_name=column_name,
        definition=str(llm_data.get("definition", "")).strip(),
        terminology_bindings=bindings,
        sensitivity=sensitivity,
        join_paths=[str(jp) for jp in join_paths],
        confidence=confidence,
        unit_of_measure=unit_of_measure,
        measurement_method=measurement_method,
    )


# ── Term entry extraction ────────────────────────────────────────────────────

def _extract_term_entries(sem_profile: SemanticTableProfile) -> list[TermEntry]:
    """Extract TermEntry objects from a completed semantic profile."""
    entries: list[TermEntry] = []
    table_name = sem_profile.table_name
    for sc in sem_profile.columns:
        fq_col = f"{table_name}.{sc.column_name}"
        for tb in sc.terminology_bindings:
            entry = TermEntry(
                system=tb.system,
                code=tb.code,
                display=tb.display,
                source_columns=[fq_col],
            )
            entries.append(entry)
    return entries


# ── Applicability warnings ───────────────────────────────────────────────────

def _collect_applicability_warnings(sem_profile, tech_profile):
    warnings: list[str] = []
    if not sem_profile.primary_key.columns or sem_profile.primary_key.pk_type == "none":
        warnings.append("No primary key identified — manual review recommended")
    if sem_profile.semantic_domain.primary == "General / Other":
        warnings.append("Generic domain assigned — consider refining")
    if not sem_profile.granularity:
        warnings.append("Granularity could not be determined")
    numeric_cols_without_uom = []
    pk_cols = set(sem_profile.primary_key.columns) if sem_profile.primary_key else set()
    for sc in sem_profile.columns:
        tc = tech_profile.get_column(sc.column_name)
        if (tc and tc.min_value is not None
                and not sc.unit_of_measure
                and sc.measurement_method != "administrative"
                and sc.column_name not in pk_cols):
            numeric_cols_without_uom.append(sc.column_name)
    if numeric_cols_without_uom:
        cols = ", ".join(numeric_cols_without_uom[:5])
        suffix = f" and {len(numeric_cols_without_uom) - 5} more" if len(numeric_cols_without_uom) > 5 else ""
        warnings.append(f"Unit of measure not determined for numeric columns: {cols}{suffix}")
    return warnings


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

def _run_judge_validation(sem_profile, tech_profile, model_name, project_id=None):
    summary_lines = [
        f"Table: {sem_profile.table_name}",
        f"Business name: {sem_profile.business_name}",
        f"Granularity: {sem_profile.granularity}",
        f"Domain: {sem_profile.semantic_domain.primary} / {sem_profile.semantic_domain.sub_domain}",
        f"Primary key: {sem_profile.primary_key.columns} (type={sem_profile.primary_key.pk_type}, "
        f"confidence={sem_profile.primary_key.confidence})",
        "",
    ]
    for sc in sem_profile.columns:
        tc = tech_profile.get_column(sc.column_name)
        tc_info = ""
        if tc:
            tc_info = f" (type={tc.data_type}, nulls={tc.null_percent}%, distinct={tc.distinct_count})"
        bindings_str = ", ".join(f"{tb.system}|{tb.code}" for tb in sc.terminology_bindings)
        summary_lines.append(
            f"  {sc.column_name}{tc_info}: "
            f'def="{sc.definition}" | sens={sc.sensitivity} | '
            f"bindings=[{bindings_str}] | conf={sc.confidence} | "
            f"uom={sc.unit_of_measure or '(none)'}"
        )
    user_msg = "\n".join(summary_lines)
    try:
        response = call_gemini(
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_message=user_msg,
            model_name=model_name,
            project_id=project_id,
            temperature=0.0,
        )
        result = extract_json_from_response(response)
        if isinstance(result, dict):
            status = str(result.get("status", "pass")).strip().lower()
            if status not in ("pass", "warning", "fail"):
                status = "pass"
            return SemanticValidationResult(status=status, issues=result.get("issues", []), warnings=result.get("warnings", []))
    except Exception as e:
        return SemanticValidationResult(status="pass", issues=[], warnings=[f"Judge call failed (non-blocking): {e}"])
    return SemanticValidationResult(status="pass")
