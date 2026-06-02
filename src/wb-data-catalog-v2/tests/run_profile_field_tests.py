#!/usr/bin/env python3
"""
Regression test: Semantic Profile Field Coverage

Validates that the semantic profiler produces ALL fields consumed by the app.
Run after any change to profiling code:
  - verily_profiler/semantic.py (LLM prompt, post-processing)
  - verily_profiler/models.py (dataclass, to_json_dict)
  - bulk_profiler.py (join inference, structural links)
  - join_inference.py (join detection)

Usage:
  python tests/run_profile_field_tests.py
  python tests/run_profile_field_tests.py --project wb-beamish-acorn-6393 --bucket metadata-json-wb-shrewd-papaya-8403

Tests:
  1. Dataclass field coverage — all expected fields exist on the model
  2. Serialization coverage — to_json_dict() emits all expected keys
  3. LLM prompt coverage — prompt text mentions all expected fields
  4. Post-processing coverage — profile_semantic() copies all fields
  5. Live profile validation — if --project given, check GCS profiles
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# ── Expected fields (source of truth) ────────────────────────────────────────

EXPECTED_TABLE_FIELDS = {
    "table",             # table name
    "profiled_at",       # timestamp
    "model_used",        # LLM model
    "business_name",     # friendly name
    "table_definition",  # description
    "primary_key",       # PK object
    "granularity",       # row-level description
    "semantic_domain",   # domain classification
    "entity_anchor",     # entity identifier column
    "entity_type",       # what the entity is
    "cohort_dimensions", # filterable columns
    "validation",        # judge results
    "columns",           # column profiles
}

EXPECTED_COLUMN_FIELDS = {
    "name",                  # column name
    "definition",            # business description
    "terminology_bindings",  # array of {system, code, display}
    "sensitivity",           # HIPAA sensitivity code
    "join_paths",            # cross-table join targets
    "confidence",            # high/medium/low
    "unit_of_measure",       # measurement unit
    "measurement_method",    # how data was captured
}

OPTIONAL_COLUMN_FIELDS = {
    "concept_binding",       # primary FHIR concept (object or null)
    "code_system_binding",   # code system for values (object or null)
    "value_set_binding",     # known value set (array or null)
}

EXPECTED_STRUCTURAL_LINK_FIELDS = {
    "source_column",
    "target_table",
    "target_column",
    "link_type",
    "cardinality",
    "confidence",
}

# ── Test functions ───────────────────────────────────────────────────────────

passed = 0
failed = 0
warnings = 0


def ok(msg):
    global passed
    passed += 1
    print(f"  ✓ {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  ✗ {msg}")


def warn(msg):
    global warnings
    warnings += 1
    print(f"  ⚠ {msg}")


def test_dataclass_fields():
    """Test 1: SemanticTableProfile and SemanticColumnProfile have all expected fields."""
    print("\n── Test 1: Dataclass field coverage ──")
    from verily_profiler.models import SemanticTableProfile, SemanticColumnProfile

    tp = SemanticTableProfile(table_name="test")
    tp_dict = tp.to_json_dict()

    for field in EXPECTED_TABLE_FIELDS:
        if field in tp_dict:
            ok(f"Table field '{field}' present in dataclass")
        else:
            fail(f"Table field '{field}' MISSING from SemanticTableProfile.to_json_dict()")

    cp = SemanticColumnProfile(column_name="test_col")
    cp_dict = cp.to_json_dict()

    for field in EXPECTED_COLUMN_FIELDS:
        if field in cp_dict:
            ok(f"Column field '{field}' present in dataclass")
        else:
            fail(f"Column field '{field}' MISSING from SemanticColumnProfile.to_json_dict()")

    # Optional fields: should exist on the class even if not in default serialization
    for field in OPTIONAL_COLUMN_FIELDS:
        if hasattr(cp, field.replace("-", "_")):
            ok(f"Column optional field '{field}' exists on dataclass")
        else:
            fail(f"Column optional field '{field}' MISSING from SemanticColumnProfile")


def test_serialization_roundtrip():
    """Test 2: to_json_dict() produces correct types for all fields."""
    print("\n── Test 2: Serialization types ──")
    from verily_profiler.models import SemanticTableProfile, SemanticColumnProfile, TerminologyBinding

    tp = SemanticTableProfile(
        table_name="project.dataset.table",
        business_name="Test Table",
        entity_anchor="patient_id",
        entity_type="patient",
        cohort_dimensions=["age", "gender"],
    )
    tp.columns.append(SemanticColumnProfile(
        column_name="age",
        definition="Patient age",
        concept_binding={"system": "LOINC", "code": "30525-0", "display": "Age", "confidence": "high"},
        code_system_binding=None,
        value_set_binding=["18-25", "26-35", "36-45"],
    ))
    d = tp.to_json_dict()

    checks = [
        ("table", str), ("business_name", str), ("entity_anchor", str),
        ("entity_type", str), ("cohort_dimensions", list), ("columns", list),
        ("primary_key", dict), ("semantic_domain", dict), ("validation", dict),
    ]
    for key, expected_type in checks:
        val = d.get(key)
        if val is not None and isinstance(val, expected_type):
            ok(f"Table '{key}' is {expected_type.__name__}")
        else:
            fail(f"Table '{key}' expected {expected_type.__name__}, got {type(val).__name__}")

    col = d["columns"][0]
    if col.get("concept_binding") and isinstance(col["concept_binding"], dict):
        ok("Column 'concept_binding' serialized as dict")
    else:
        fail("Column 'concept_binding' not serialized correctly")
    if col.get("value_set_binding") and isinstance(col["value_set_binding"], list):
        ok("Column 'value_set_binding' serialized as list")
    else:
        fail("Column 'value_set_binding' not serialized correctly")


def test_llm_prompt_coverage():
    """Test 3: LLM prompt mentions all required fields."""
    print("\n── Test 3: LLM prompt field coverage ──")
    from verily_profiler.semantic import _build_system_prompt

    prompt = _build_system_prompt()

    table_fields_in_prompt = [
        ("business_name", '"business_name"'),
        ("table_definition", '"table_definition"'),
        ("primary_key", '"primary_key"'),
        ("granularity", '"granularity"'),
        ("semantic_domain", '"semantic_domain"'),
        ("entity_anchor", '"entity_anchor"'),
        ("entity_type", '"entity_type"'),
        ("cohort_dimensions", '"cohort_dimensions"'),
        ("columns", '"columns"'),
    ]
    for field, needle in table_fields_in_prompt:
        if needle in prompt:
            ok(f"Prompt includes table field {field}")
        else:
            fail(f"Prompt MISSING table field {field}")

    col_fields_in_prompt = [
        ("column_name", '"column_name"'),
        ("definition", '"definition"'),
        ("terminology_bindings", '"terminology_bindings"'),
        ("sensitivity", '"sensitivity"'),
        ("join_paths", '"join_paths"'),
        ("confidence", '"confidence"'),
        ("unit_of_measure", '"unit_of_measure"'),
        ("measurement_method", '"measurement_method"'),
        ("concept_binding", '"concept_binding"'),
        ("code_system_binding", '"code_system_binding"'),
    ]
    for field, needle in col_fields_in_prompt:
        if needle in prompt:
            ok(f"Prompt includes column field {field}")
        else:
            fail(f"Prompt MISSING column field {field}")


def test_post_processing():
    """Test 4: profile_semantic() copies all table-level fields from LLM output."""
    print("\n── Test 4: Post-processing coverage ──")
    import inspect
    from verily_profiler.semantic import profile_semantic

    source = inspect.getsource(profile_semantic)

    table_fields = [
        "business_name", "table_definition", "granularity",
        "primary_key", "semantic_domain",
        "entity_anchor", "entity_type", "cohort_dimensions",
    ]
    for field in table_fields:
        if f'"{field}"' in source or f"'{field}'" in source:
            ok(f"profile_semantic() extracts '{field}' from LLM output")
        else:
            fail(f"profile_semantic() does NOT extract '{field}' from LLM output")


def test_build_semantic_column():
    """Test 5: _build_semantic_column() parses all column-level fields."""
    print("\n── Test 5: Column builder coverage ──")
    from verily_profiler.semantic import _build_semantic_column

    test_data = {
        "definition": "Test column",
        "terminology_bindings": [{"system": "LOINC", "code": "1234-5", "display": "Test"}],
        "sensitivity": "PHI-DIRECT",
        "join_paths": ["other_table.col"],
        "confidence": "high",
        "unit_of_measure": "mg/dL",
        "measurement_method": "lab-measured",
        "concept_binding": {"system": "LOINC", "code": "1234-5", "display": "Test", "confidence": "high"},
        "code_system_binding": {"system": "http://loinc.org", "display": "LOINC", "confidence": "high"},
    }
    col = _build_semantic_column("test_col", test_data)
    d = col.to_json_dict()

    for field in EXPECTED_COLUMN_FIELDS:
        if field in d:
            ok(f"Column builder produces '{field}'")
        else:
            fail(f"Column builder MISSING '{field}'")

    if col.concept_binding and col.concept_binding.get("system") == "LOINC":
        ok("Column builder parses 'concept_binding'")
    else:
        fail("Column builder does NOT parse 'concept_binding'")

    if col.code_system_binding and col.code_system_binding.get("system") == "http://loinc.org":
        ok("Column builder parses 'code_system_binding'")
    else:
        fail("Column builder does NOT parse 'code_system_binding'")


def test_frontend_type_alignment():
    """Test 6: Frontend TypeScript types match backend output."""
    print("\n── Test 6: Frontend type alignment ──")
    types_file = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "types", "profile.ts")
    if not os.path.isfile(types_file):
        warn("Frontend types file not found, skipping")
        return

    with open(types_file) as f:
        ts = f.read()

    for field in ["entity_anchor", "entity_type", "cohort_dimensions", "structural_links"]:
        if field in ts:
            ok(f"Frontend SemProfile type includes '{field}'")
        else:
            fail(f"Frontend SemProfile type MISSING '{field}'")

    for field in ["concept_binding", "code_system_binding", "value_set_binding"]:
        if field in ts:
            ok(f"Frontend SemColumn type includes '{field}'")
        else:
            fail(f"Frontend SemColumn type MISSING '{field}'")


def test_live_profiles(project: str, bucket: str):
    """Test 7: Live GCS profiles contain all expected fields."""
    print(f"\n── Test 7: Live profile validation ({project}) ──")
    try:
        from verily_profiler.storage import scan_profile_availability, read_sem_profile
    except ImportError:
        warn("Cannot import storage module, skipping live tests")
        return

    billing = os.environ.get("GCP_PROJECT_ID", bucket.replace("metadata-json-", ""))

    try:
        avail = scan_profile_availability(bucket, project, billing_project_id=billing)
    except Exception as e:
        warn(f"Cannot scan profiles: {e}")
        return

    sem_tables = [fq for fq, info in avail.items() if info.get("semantic")]
    if not sem_tables:
        warn("No semantic profiles found in GCS")
        return

    sample = sem_tables[:3]
    for fq in sample:
        try:
            sem = read_sem_profile(bucket, fq, project_id=billing)
        except Exception as e:
            fail(f"{fq}: cannot read profile: {e}")
            continue

        if not sem:
            fail(f"{fq}: profile is None")
            continue

        missing = EXPECTED_TABLE_FIELDS - set(sem.keys())
        if missing:
            fail(f"{fq}: missing table fields: {missing}")
        else:
            ok(f"{fq}: all table fields present")

        cols = sem.get("columns", [])
        if cols:
            col = cols[0]
            col_missing = EXPECTED_COLUMN_FIELDS - set(col.keys())
            if col_missing:
                fail(f"{fq}.{col.get('name','?')}: missing column fields: {col_missing}")
            else:
                ok(f"{fq}.{col.get('name','?')}: all column fields present")

        if "entity_anchor" in sem and sem["entity_anchor"]:
            ok(f"{fq}: entity_anchor = '{sem['entity_anchor']}'")
        else:
            warn(f"{fq}: entity_anchor is empty (may need re-profiling)")

        dims = sem.get("cohort_dimensions", [])
        if dims:
            ok(f"{fq}: cohort_dimensions has {len(dims)} entries")
        else:
            warn(f"{fq}: cohort_dimensions is empty (may need re-profiling)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic profile field coverage regression tests")
    parser.add_argument("--project", help="GCP data project for live profile validation")
    parser.add_argument("--bucket", help="GCS bucket for live profile validation")
    args = parser.parse_args()

    print("=" * 60)
    print("Semantic Profile Field Coverage — Regression Tests")
    print("=" * 60)

    test_dataclass_fields()
    test_serialization_roundtrip()
    test_llm_prompt_coverage()
    test_post_processing()
    test_build_semantic_column()
    test_frontend_type_alignment()

    if args.project and args.bucket:
        test_live_profiles(args.project, args.bucket)
    else:
        print("\n  (Skipping live profile tests — pass --project and --bucket to enable)")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
