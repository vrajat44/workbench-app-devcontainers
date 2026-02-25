"""
FHIR Metadata Validator for WB Metadata Creator.

Implements the LLM-as-a-Judge validation pipeline with 8 checks.
Each check runs independently via the Pro model and returns structured results.

Checks:
  1. Column Coverage — are all BQ columns present in the StructureDefinition?
  2. Data Type Accuracy — are FHIR types semantically correct?
  3. VFIG Mapping Accuracy — are FHIR resource mappings plausible?
  4. Security Label Accuracy — are sensitivity labels correct?
  5. Measurement Method Accuracy — are measurement methods plausible?
  6. Cross-File Consistency — are shared columns consistent across files?
  7. L3 Metadata Completeness — are all table-level fields present?
  8. ValueSet Binding Completeness — are coded columns bound to ValueSets?

Usage:
    from fhir_validator import run_validation_pipeline, CheckResult

    results = run_validation_pipeline(
        generated_jsons, bq_tables, project_id,
        progress_callback=lambda check, status: print(f"{check}: {status}")
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Callable

from fhir_generator import BQTableInfo, format_bq_schema_for_prompt
from prompt_engine import call_gemini_pro, extract_json_from_response


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """A single validation issue found by a check."""
    severity: str  # "critical", "warning", "info"
    check_name: str
    table_name: str
    column_name: Optional[str]  # None for table-level issues
    description: str
    current_value: str = ""
    expected_value: str = ""
    fix_suggestion: str = ""
    applied: bool = False
    skipped: bool = False


@dataclass
class CheckResult:
    """Result of a single validation check."""
    check_number: int
    check_name: str
    status: str  # "pass", "warnings", "fail"
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")


@dataclass
class ValidationReport:
    """Complete validation report across all checks."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "FAIL"
        if any(c.status == "warnings" for c in self.checks):
            return "PASS WITH WARNINGS"
        return "PASS"

    @property
    def all_issues(self) -> list[ValidationIssue]:
        issues = []
        for check in self.checks:
            issues.extend(check.issues)
        return issues

    @property
    def critical_count(self) -> int:
        return sum(c.critical_count for c in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(c.warning_count for c in self.checks)

    @property
    def info_count(self) -> int:
        return sum(c.info_count for c in self.checks)


# ── Validation Check Prompts ──────────────────────────────────────────────────

_VALIDATOR_SYSTEM_PROMPT = """You are a senior FHIR metadata auditor. You will validate
FHIR StructureDefinition JSONs against BigQuery schemas.

Respond in JSON format with this structure:
```json
{
  "status": "pass" | "warnings" | "fail",
  "summary": "Brief summary of findings",
  "issues": [
    {
      "severity": "critical" | "warning" | "info",
      "table_name": "table name",
      "column_name": "column name or null for table-level",
      "description": "What's wrong",
      "current_value": "What the JSON currently has",
      "expected_value": "What it should be",
      "fix_suggestion": "How to fix it"
    }
  ]
}
```

Be thorough but accurate. Do not report false positives. Only report genuine issues.
"""

_CHECK_PROMPTS = {
    1: {
        "name": "Column Coverage",
        "severity_default": "critical",
        "prompt": """CHECK 1 — Column Coverage (Critical)

For each table, compare the BigQuery schema columns against the StructureDefinition elements:
- Count columns in BQ schema vs. elements in JSON (exclude the root element)
- List any MISSING columns (in BQ but not in JSON)
- List any EXTRA columns (in JSON but not in BQ)
- Flag column name mismatches (case sensitivity, typos)
- Expected result: 1:1 match between BQ columns and JSON elements

BQ Schemas:
{bq_schemas}

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    2: {
        "name": "Data Type Accuracy",
        "severity_default": "critical",
        "prompt": """CHECK 2 — Data Type Accuracy (Critical)

For each column, verify the FHIR type[0].code is semantically correct given the BQ data type:
- BQ STRING → FHIR "string" (or "code" for categorical/coded columns)
- BQ INTEGER/INT64 → FHIR "integer"
- BQ FLOAT/FLOAT64/NUMERIC → FHIR "decimal"
- BQ BOOLEAN/BOOL → FHIR "boolean"
- BQ DATE → FHIR "date"
- BQ TIMESTAMP/DATETIME → FHIR "dateTime"

Special cases:
- Columns that store binary 0/1 values (even if BQ type is FLOAT) → should be "integer", not "decimal"
- Columns with controlled vocabularies → should be "code", not "string"
- Fixed value columns → should have fixedString

BQ Schemas:
{bq_schemas}

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    3: {
        "name": "VFIG Mapping Accuracy",
        "severity_default": "warning",
        "prompt": """CHECK 3 — VFIG Mapping Semantic Accuracy (Warning)

Validate that each column's VFIG mapping (identity: "vfig") is semantically plausible:
- Subject identifiers → Patient.identifier
- Site identifiers → Location.identifier
- Visit labels → Encounter.type
- Visit numbers → Encounter.period
- Diagnoses/conditions → Condition.code
- Medications → MedicationStatement.medication
- Lab results → Observation.valueQuantity or Observation.interpretation
- Risk scores → RiskAssessment.prediction
- Questionnaire scores → Observation.valueInteger or Observation.valueQuantity
- Demographics → Patient.gender, Patient.birthDate, Patient.extension:race
- Availability flags → DiagnosticReport.status
- Smoking → Observation.valueCodeableConcept

Flag any mapping that doesn't match these expected patterns.

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    4: {
        "name": "Security Label Accuracy",
        "severity_default": "warning",
        "prompt": """CHECK 4 — Security Label Accuracy (Warning)

Verify security labels are semantically correct:
- Subject identifiers → UID (Unique Identifier)
- Individual-level clinical conditions, individual diagnoses, individual medical history → PHI
- Names → PII (Personally Identifiable Information)
- Financial amounts → FINANCIAL
- Business identifiers → PII_BUSINESS
- Non-sensitive columns (study IDs, site IDs, visit labels, visit numbers) → should NOT have a security label
- Aggregated/statistical columns (averages, counts, rates, percentages, totals) → should NOT have a security label

CRITICAL DISTINCTION — Individual-level vs. Aggregated data:
- PHI applies ONLY to **individual-level** health data tied to a specific person.
- PHI does NOT apply to **aggregated / population-level statistics** such as averages, counts,
  rates, percentages, medians, or any statistic computed across a group.
  Examples of NON-PHI: "average birth weight", "average maternal age", "fertility rate",
  "percent low birth weight", "average BMI", "infant mortality rate".
  These are public health / summary statistics, not protected health information.

Flag:
- Individual-level clinical data that LACK a security label
- Aggregated/statistical columns that HAVE a PHI/UID/PII security label (they should not)
- Non-sensitive data that HAVE a security label
- Semantically wrong labels (e.g., a name labeled as UID instead of PII)

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    5: {
        "name": "Measurement Method Accuracy",
        "severity_default": "info",
        "prompt": """CHECK 5 — Measurement Method Accuracy (Info)

Verify measurement method codes are plausible:
- Self-reported questionnaire responses → self-reported
- Calculated/derived scores → calculated
- Lab test results → laboratory-measured
- Vital signs measured by staff → clinician-observed
- Device-collected data → device-collected
- EHR-extracted records → extracted-from-ehr
- System/protocol metadata → administrative

Flag any measurement method that doesn't match the expected category.

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    6: {
        "name": "Cross-File Consistency",
        "severity_default": "warning",
        "prompt": """CHECK 6 — Cross-File Consistency (Warning)

For columns that appear in multiple StructureDefinitions (shared columns like study IDs,
subject IDs, visit fields), verify consistency:
- Same FHIR type across all files
- Same definition/description theme
- Same VFIG mapping
- Same security label
- Same measurement method

Flag any inconsistency between files for the same column name.

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    7: {
        "name": "L3 Metadata Completeness",
        "severity_default": "critical",
        "prompt": """CHECK 7 — L3 Metadata Completeness (Critical)

For each StructureDefinition, verify all 9 Level 3 (table-level) fields are present and valid:
1. title AND description are non-empty and informative
2. purpose describes granularity (what one record represents)
3. contact references an Organization
4. status is set (active, draft, or retired)
5. useContext has compliance-zone entry
6. useContext has retention-policy entry
7. useContext has schema-stability entry
8. verily-primary-identity extension is present and references fields that exist in the element list
9. At least one verily-structural-link extension is present (unless it's a standalone table)

Also verify infrastructure:
- BigQueryTableSchemaMetadata extension with table-name
- meta.security with a confidentiality code
- kind: "logical", abstract: false
- mapping[].identity = "vfig" declared

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
    8: {
        "name": "ValueSet Binding Completeness",
        "severity_default": "info",
        "prompt": """CHECK 8 — ValueSet Binding Completeness (Info)

For columns that appear to be coded/categorical (type = "code" or have controlled values):
- Verify a binding.valueSet is present
- Verify binding.strength is appropriate ("required" for closed enumeration, "extensible" for open)
- Verify the ValueSet URL follows the naming convention http://fhir.verily.com/ValueSet/{{name}}
- Check that the ValueSet name is descriptive

For columns that DON'T have a binding but look like they should (based on column name patterns
like _STATUS, _TYPE, _CODE, _FLAG, _CATEGORY):
- Flag them as missing bindings

FHIR StructureDefinition JSONs:
{fhir_jsons}
"""
    },
}


# ── Validation Execution ─────────────────────────────────────────────────────

def _run_single_check(
    check_number: int,
    fhir_jsons: list[dict],
    bq_tables: list[BQTableInfo],
    project_id: Optional[str] = None,
) -> CheckResult:
    """
    Run a single validation check via the Pro model.

    Args:
        check_number: Check number (1-8).
        fhir_jsons: List of generated StructureDefinition JSONs.
        bq_tables: List of BQ table info for comparison.
        project_id: GCP project for Vertex AI calls.

    Returns:
        CheckResult with issues found.
    """
    check_config = _CHECK_PROMPTS[check_number]
    check_name = check_config["name"]

    # Format BQ schemas
    bq_schemas = "\n\n".join(format_bq_schema_for_prompt(t) for t in bq_tables)

    # Format FHIR JSONs (truncate if too large)
    fhir_json_strs = []
    for fj in fhir_jsons:
        fj_str = json.dumps(fj, indent=2)
        # Truncate very large JSONs to avoid token limits
        if len(fj_str) > 30000:
            fj_str = fj_str[:30000] + "\n... (truncated)"
        fhir_json_strs.append(fj_str)
    fhir_jsons_text = "\n\n---\n\n".join(fhir_json_strs)

    # Build the check prompt
    check_prompt = check_config["prompt"].format(
        bq_schemas=bq_schemas,
        fhir_jsons=fhir_jsons_text,
    )

    try:
        response = call_gemini_pro(
            system_prompt=_VALIDATOR_SYSTEM_PROMPT,
            user_message=check_prompt,
            project_id=project_id,
        )

        # Parse the response
        result_json = extract_json_from_response(response)
        if result_json:
            issues = []
            for issue_data in result_json.get("issues", []):
                issues.append(ValidationIssue(
                    severity=issue_data.get("severity", check_config["severity_default"]),
                    check_name=check_name,
                    table_name=issue_data.get("table_name", ""),
                    column_name=issue_data.get("column_name"),
                    description=issue_data.get("description", ""),
                    current_value=issue_data.get("current_value", ""),
                    expected_value=issue_data.get("expected_value", ""),
                    fix_suggestion=issue_data.get("fix_suggestion", ""),
                ))

            return CheckResult(
                check_number=check_number,
                check_name=check_name,
                status=result_json.get("status", "pass"),
                issues=issues,
                summary=result_json.get("summary", ""),
                raw_response=response,
            )
        else:
            # Couldn't parse JSON — return the raw response as summary
            return CheckResult(
                check_number=check_number,
                check_name=check_name,
                status="warnings",
                summary=f"Could not parse structured response. Raw output: {response[:500]}",
                raw_response=response,
            )

    except Exception as e:
        return CheckResult(
            check_number=check_number,
            check_name=check_name,
            status="fail",
            summary=f"Check failed: {str(e)}",
        )


def run_validation_pipeline(
    fhir_jsons: list[dict],
    bq_tables: list[BQTableInfo],
    project_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> ValidationReport:
    """
    Run all 8 validation checks sequentially, calling progress_callback after each.

    Args:
        fhir_jsons: List of generated StructureDefinition JSONs.
        bq_tables: List of BQ table info for comparison.
        project_id: GCP project for Vertex AI calls.
        progress_callback: Called as progress_callback(check_number, check_name, status)
            where status is "running", "pass", "warnings", "fail", or "error".

    Returns:
        ValidationReport with all check results.
    """
    report = ValidationReport()

    for check_num in range(1, 9):
        check_name = _CHECK_PROMPTS[check_num]["name"]

        if progress_callback:
            progress_callback(check_num, check_name, "running")

        result = _run_single_check(
            check_number=check_num,
            fhir_jsons=fhir_jsons,
            bq_tables=bq_tables,
            project_id=project_id,
        )
        report.checks.append(result)

        if progress_callback:
            progress_callback(check_num, check_name, result.status)

    return report


# ── Fix Application ───────────────────────────────────────────────────────────

_FIX_SYSTEM_PROMPT = """You are a FHIR metadata specialist. Apply the requested fix to the
FHIR StructureDefinition JSON. Return the COMPLETE modified JSON in a ```json code block.
Only change what is needed for the fix — preserve everything else exactly."""

_FIX_USER_TEMPLATE = """Apply this fix to the following FHIR StructureDefinition JSON:

## Fix to Apply:
- Table: {table_name}
- Column: {column_name}
- Issue: {description}
- Current value: {current_value}
- Expected value: {expected_value}
- Suggestion: {fix_suggestion}

## Current JSON:
```json
{current_json}
```

Return the complete corrected JSON in a ```json code block.
"""


def apply_fix(
    issue: ValidationIssue,
    current_json: dict,
    project_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Apply a single validation fix to a StructureDefinition JSON.

    Uses the fast model (fixes are straightforward edits).

    Args:
        issue: The validation issue to fix.
        current_json: The current StructureDefinition JSON.
        project_id: GCP project for Vertex AI calls.

    Returns:
        Updated JSON dict, or None if fix failed.
    """
    from prompt_engine import call_gemini_fast

    user_message = _FIX_USER_TEMPLATE.format(
        table_name=issue.table_name,
        column_name=issue.column_name or "(table-level)",
        description=issue.description,
        current_value=issue.current_value,
        expected_value=issue.expected_value,
        fix_suggestion=issue.fix_suggestion,
        current_json=json.dumps(current_json, indent=2),
    )

    try:
        response = call_gemini_fast(
            system_prompt=_FIX_SYSTEM_PROMPT,
            user_message=user_message,
            project_id=project_id,
        )
        return extract_json_from_response(response)
    except Exception as e:
        print(f"⚠ Fix application failed: {e}")
        return None


# ── Report Formatting ─────────────────────────────────────────────────────────

def format_validation_report(report: ValidationReport) -> str:
    """Format the validation report for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("FHIR METADATA VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    for check in report.checks:
        icon = {"pass": "✅", "warnings": "⚠️", "fail": "❌"}.get(check.status, "❓")
        issue_counts = []
        if check.critical_count:
            issue_counts.append(f"{check.critical_count} critical")
        if check.warning_count:
            issue_counts.append(f"{check.warning_count} warnings")
        if check.info_count:
            issue_counts.append(f"{check.info_count} info")
        counts_str = f" ({', '.join(issue_counts)})" if issue_counts else ""

        lines.append(f"{icon} Check {check.check_number}: {check.check_name} — {check.status.upper()}{counts_str}")

        if check.summary:
            lines.append(f"   {check.summary}")

        for issue in check.issues:
            sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
            col_str = f".{issue.column_name}" if issue.column_name else ""
            lines.append(f"   {sev_icon} [{issue.severity}] {issue.table_name}{col_str}: {issue.description}")
            if issue.fix_suggestion:
                lines.append(f"      Fix: {issue.fix_suggestion}")

        lines.append("")

    lines.append("-" * 60)
    lines.append(f"Overall: {report.overall_status}")
    lines.append(f"  Critical: {report.critical_count} | Warnings: {report.warning_count} | Info: {report.info_count}")

    return "\n".join(lines)


def format_check_status_display(report: ValidationReport) -> str:
    """Format check statuses for Gradio display (markdown)."""
    lines = []
    for i in range(1, 9):
        check_name = _CHECK_PROMPTS[i]["name"]
        # Find if this check has been run
        check_result = next((c for c in report.checks if c.check_number == i), None)

        if check_result:
            icon = {"pass": "✅", "warnings": "⚠️", "fail": "❌"}.get(check_result.status, "❓")
            counts = []
            if check_result.critical_count:
                counts.append(f"{check_result.critical_count} critical")
            if check_result.warning_count:
                counts.append(f"{check_result.warning_count} warning")
            if check_result.info_count:
                counts.append(f"{check_result.info_count} info")
            counts_str = f" — {', '.join(counts)}" if counts else ""
            lines.append(f"{icon} **Check {i}: {check_name}** — {check_result.status.upper()}{counts_str}")
        else:
            lines.append(f"⏳ **Check {i}: {check_name}** — pending")

    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION — Validate the metadata table BEFORE JSON generation
# ══════════════════════════════════════════════════════════════════════════════

_INPUT_VALIDATOR_SYSTEM = """You are a senior FHIR metadata auditor. You validate column-level metadata
that a user has prepared BEFORE it is assembled into FHIR StructureDefinition JSON.

The input is a table of column metadata with these fields:
  Column, BQ Type, FHIR Type, Short Label, Description, Required, Null %, Distinct,
  Sensitivity, Measurement, FHIR Mapping, Coded

Plus table-level metadata: title, description, purpose, primary_key.

Respond ONLY with a JSON object:
```json
{
  "status": "pass" | "warnings" | "fail",
  "summary": "Brief overall summary",
  "issues": [
    {
      "severity": "critical" | "warning" | "info",
      "check_name": "Name of the check that found this",
      "column_name": "column name or null for table-level",
      "description": "What's wrong",
      "current_value": "What the metadata currently has",
      "suggested_value": "What it should be",
      "fix_action": "Brief instruction for the user"
    }
  ]
}
```
Be thorough but accurate. Do not report false positives.
"""

_INPUT_CHECK_PROMPTS = {
    1: {
        "name": "Description Quality",
        "prompt": """CHECK: Description Quality

For each column, evaluate the description (definition) field:
- Is it meaningful and clinical/business-rich? (not just restating the column name)
- Does it provide context about what the data means, how it's collected, or why it matters?
- Are descriptions overly generic or vague?
- For coded columns, does the description mention possible values or categories?

Flag:
- Empty or missing descriptions (critical)
- Descriptions that just restate the column name (warning)
- Descriptions that are too short (< 10 words) and uninformative (warning)
- Descriptions that are excellent (info — positive feedback, list sparingly)

Also check table-level description and purpose:
- Table title should be descriptive (not just the BQ table name)
- Table description should explain what data the table contains
- Purpose should describe granularity (e.g., "One record per ...")

{metadata_table}
""",
    },
    2: {
        "name": "Sensitivity & Security Labels",
        "prompt": """CHECK: Sensitivity & Security Label Accuracy

Evaluate the Sensitivity column for each row.

CRITICAL DISTINCTION — Individual-level vs. Aggregated data:
- PHI applies ONLY to **individual-level** health data that relates to a specific person:
  diagnoses of a patient, a patient's lab result, a patient's clinical measurement, individual treatment records.
- PHI does NOT apply to **aggregated / population-level statistics**. Columns like averages, counts,
  rates, percentages, medians, or any statistic computed across a group of people are NOT PHI.
  Examples of NON-PHI aggregated columns: "average birth weight", "average maternal age",
  "average gestational age", "infant mortality rate", "percent low birth weight",
  "average BMI", "average prenatal visits", "fertility rate".
  These are public health statistics, not individual health information.

Sensitivity label rules:
- PHI: individual-level health conditions, individual diagnoses, individual clinical measurements,
  individual lab results, individual treatment data, individual medical history, individual biometric data
- UID: subject/patient identifiers, record IDs, participant IDs
- PII: names, addresses, individual dates of birth, phone numbers, emails, SSNs
- P_BIRTHSEX: individual sex at birth
- P_RACE / P_ETHNICITY: individual race or ethnicity
- NONE / empty: administrative flags, study metadata, reporting codes, site IDs, visit numbers,
  AND any aggregated/statistical columns (averages, counts, rates, percentages, totals)

Flag:
- Individual-level clinical/health data missing a sensitivity label (critical)
- Subject identifiers missing UID label (critical)
- Aggregated/statistical columns incorrectly labeled PHI/UID/PII (critical — these are NOT sensitive)
- Non-sensitive columns (flags, codes, administrative fields) incorrectly labeled (warning)
- Sensitivity labels that are semantically wrong (e.g., a diagnosis labeled UID) (critical)

{metadata_table}
""",
    },
    3: {
        "name": "FHIR Type & Coded Column Accuracy",
        "prompt": """CHECK: FHIR Type Mapping & Coded Column Detection

Evaluate FHIR Type and Coded columns:

FHIR Type rules:
- BQ STRING → FHIR "string" (or "code" if categorical)
- BQ INTEGER/INT64 → FHIR "integer"
- BQ FLOAT64/FLOAT/NUMERIC → FHIR "decimal" (but binary 0/1 columns should be "integer" or "boolean")
- BQ BOOLEAN → FHIR "boolean"
- BQ DATE → FHIR "date"
- BQ TIMESTAMP/DATETIME → FHIR "dateTime"

Coded column rules:
- Columns with ≤25 distinct values are very likely coded → should have Coded = "Yes"
- Columns with Coded = "Yes" should have FHIR Type = "code"
- Columns with high distinct counts (>100) should generally NOT be coded
- Flag columns marked Coded but with type != "code"
- Flag columns with low distinct count but NOT marked coded

Required vs Null %:
- If Required = "Yes" but Null% > 0, flag it (data quality concern)
- If Required = "No" but Null% = 0, consider suggesting Required = "Yes" (info)

{metadata_table}
""",
    },
    4: {
        "name": "Measurement Method & FHIR Mapping",
        "prompt": """CHECK: Measurement Method & FHIR Mapping Plausibility

Evaluate Measurement and FHIR Mapping columns:

Measurement method rules:
- self-reported: questionnaire/survey responses, patient-provided data
- calculated: derived scores, computed fields, BMI, age calculations
- laboratory-measured: lab results, assay values, biomarker levels
- clinician-observed: vital signs, clinical exam findings, physician assessments
- device-collected: sensor data, ECG, wearables, imaging
- extracted-from-ehr: data pulled from EHR systems
- administrative: system metadata, flags, reporting codes, dates, IDs

FHIR Mapping rules:
- Mappings should be valid FHIR R4 resource.field paths
- Subject/patient IDs → Patient.identifier
- Diagnoses → Condition.code
- Lab values → Observation.valueQuantity
- Medications → MedicationStatement.medication
- Visit/encounter data → Encounter.*
- Demographics → Patient.* (gender, birthDate, extension:race)

Flag:
- Measurement methods that don't match the column semantics (warning)
- Invalid FHIR resource/field paths (warning)
- Missing measurement methods on columns that clearly need one (info)

{metadata_table}
""",
    },
    5: {
        "name": "Cross-Column Consistency",
        "prompt": """CHECK: Cross-Column & Cross-Table Consistency

Look for consistency issues across all columns and tables:
- If the same column name appears in multiple tables, do they have consistent
  descriptions, FHIR types, sensitivity labels, and measurement methods?
- Are related columns treated consistently? (e.g., all ID columns labeled UID,
  all score columns labeled calculated, all flag columns marked administrative)
- Do Short Labels make sense and follow a consistent style?
- Are there obviously related columns with contradictory metadata?

{metadata_table}
""",
    },
}


@dataclass
class InputValidationIssue:
    """A single issue found during input metadata validation."""
    severity: str           # "critical", "warning", "info"
    check_name: str
    column_name: Optional[str]   # None for table-level issues
    description: str
    current_value: str = ""
    suggested_value: str = ""
    fix_action: str = ""
    applied: bool = False
    skipped: bool = False


@dataclass
class InputValidationReport:
    """Complete input validation report."""
    checks: list[CheckResult] = field(default_factory=list)
    input_issues: list[InputValidationIssue] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "FAIL"
        if any(c.status == "warnings" for c in self.checks):
            return "PASS WITH WARNINGS"
        return "PASS"

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.input_issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.input_issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.input_issues if i.severity == "info")


def _format_metadata_for_validation(
    table_meta: dict,
    columns_df_records: list[dict],
    table_name: str = "",
) -> str:
    """Format metadata table and table-level fields for the LLM prompt."""
    lines = []
    lines.append(f"## Table: {table_name}")
    lines.append(f"- Title: {table_meta.get('title', '(empty)')}")
    lines.append(f"- Description: {table_meta.get('description', '(empty)')}")
    lines.append(f"- Purpose: {table_meta.get('purpose', '(empty)')}")
    lines.append(f"- Primary Key: {table_meta.get('primary_key', '(empty)')}")
    lines.append("")
    lines.append("## Column Metadata:")
    lines.append(
        "| Column | BQ Type | FHIR Type | Short Label | Description | Required | "
        "Null % | Distinct | Sensitivity | Measurement | FHIR Mapping | Coded |"
    )
    lines.append("| --- " * 12 + "|")
    for row in columns_df_records:
        cells = [
            str(row.get("Column", "")),
            str(row.get("BQ Type", "")),
            str(row.get("FHIR Type", "")),
            str(row.get("Short Label", "")),
            str(row.get("Description", ""))[:80],  # truncate for prompt
            str(row.get("Required", "")),
            str(row.get("Null %", "")),
            str(row.get("Distinct", "")),
            str(row.get("Sensitivity", "")),
            str(row.get("Measurement", "")),
            str(row.get("FHIR Mapping", "")),
            str(row.get("Coded", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_input_validation(
    table_meta_map: dict[str, dict],
    columns_df_records_map: dict[str, list[dict]],
    project_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> InputValidationReport:
    """
    Validate metadata table inputs using LLM-as-a-Judge (5 checks).

    Args:
        table_meta_map: {table_name: {title, description, purpose, primary_key}}
        columns_df_records_map: {table_name: [list of column row dicts]}
        project_id: GCP project for Vertex AI calls.
        progress_callback: Called as (check_num, check_name, status).

    Returns:
        InputValidationReport with all issues.
    """
    # Build the combined metadata string for all tables
    metadata_parts = []
    for tbl_name, col_records in columns_df_records_map.items():
        tbl_meta = table_meta_map.get(tbl_name, {})
        metadata_parts.append(
            _format_metadata_for_validation(tbl_meta, col_records, tbl_name)
        )
    metadata_table_str = "\n\n---\n\n".join(metadata_parts)

    report = InputValidationReport()

    for check_num in sorted(_INPUT_CHECK_PROMPTS.keys()):
        check_cfg = _INPUT_CHECK_PROMPTS[check_num]
        check_name = check_cfg["name"]

        if progress_callback:
            progress_callback(check_num, check_name, "running")

        prompt = check_cfg["prompt"].format(metadata_table=metadata_table_str)

        try:
            response = call_gemini_pro(
                system_prompt=_INPUT_VALIDATOR_SYSTEM,
                user_message=prompt,
                project_id=project_id,
            )
            result_json = extract_json_from_response(response)

            if result_json:
                issues = []
                for issue_data in result_json.get("issues", []):
                    issue = InputValidationIssue(
                        severity=issue_data.get("severity", "warning"),
                        check_name=check_name,
                        column_name=issue_data.get("column_name"),
                        description=issue_data.get("description", ""),
                        current_value=issue_data.get("current_value", ""),
                        suggested_value=issue_data.get("suggested_value", ""),
                        fix_action=issue_data.get("fix_action", ""),
                    )
                    issues.append(issue)
                    report.input_issues.append(issue)

                cr = CheckResult(
                    check_number=check_num,
                    check_name=check_name,
                    status=result_json.get("status", "pass"),
                    issues=[],  # issues tracked in report.input_issues
                    summary=result_json.get("summary", ""),
                    raw_response=response,
                )
            else:
                cr = CheckResult(
                    check_number=check_num,
                    check_name=check_name,
                    status="warnings",
                    summary=f"Could not parse response: {response[:300]}",
                    raw_response=response,
                )
        except Exception as e:
            cr = CheckResult(
                check_number=check_num,
                check_name=check_name,
                status="fail",
                summary=f"Check failed: {e}",
            )

        report.checks.append(cr)
        if progress_callback:
            progress_callback(check_num, check_name, cr.status)

    return report


def format_input_validation_report(report: InputValidationReport) -> str:
    """Format the input validation report for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("METADATA INPUT VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    for check in report.checks:
        icon = {"pass": "✅", "warnings": "⚠️", "fail": "❌"}.get(check.status, "❓")
        lines.append(f"{icon} **{check.check_name}** — {check.status.upper()}")
        if check.summary:
            lines.append(f"   {check.summary}")
        lines.append("")

    if report.input_issues:
        lines.append("-" * 60)
        lines.append("### Issues Found\n")
        for i, issue in enumerate(report.input_issues):
            sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
            col = f" → **{issue.column_name}**" if issue.column_name else ""
            lines.append(f"{i}. {sev_icon} [{issue.severity}] {issue.check_name}{col}")
            lines.append(f"   {issue.description}")
            if issue.current_value:
                lines.append(f"   Current: `{issue.current_value}`")
            if issue.suggested_value:
                lines.append(f"   Suggested: `{issue.suggested_value}`")
            if issue.fix_action:
                lines.append(f"   ✏️ {issue.fix_action}")
            lines.append("")
    else:
        lines.append("✅ **No issues found — metadata looks good!**")

    lines.append("-" * 60)
    lines.append(f"Overall: {report.overall_status}")
    lines.append(f"  Critical: {report.critical_count} | Warnings: {report.warning_count} | Info: {report.info_count}")

    return "\n".join(lines)


def format_input_check_status(report: InputValidationReport) -> str:
    """Format check statuses for Gradio markdown display."""
    lines = []
    for check in report.checks:
        icon = {"pass": "✅", "warnings": "⚠️", "fail": "❌"}.get(check.status, "❓")
        # Count issues for this check
        check_issues = [i for i in report.input_issues if i.check_name == check.check_name]
        crits = sum(1 for i in check_issues if i.severity == "critical")
        warns = sum(1 for i in check_issues if i.severity == "warning")
        infos = sum(1 for i in check_issues if i.severity == "info")
        counts = []
        if crits:
            counts.append(f"{crits} critical")
        if warns:
            counts.append(f"{warns} warning")
        if infos:
            counts.append(f"{infos} info")
        counts_str = f" — {', '.join(counts)}" if counts else ""
        lines.append(f"{icon} **Check {check.check_number}: {check.check_name}** — {check.status.upper()}{counts_str}")
    return "\n\n".join(lines)
