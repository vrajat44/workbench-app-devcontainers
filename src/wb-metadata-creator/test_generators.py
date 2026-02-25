#!/usr/bin/env python3
"""
Backend-only tests for FHIR metadata generators.

No Gradio, no BigQuery, no LLM — pure function tests with mock data.
Validates output JSON structure against the gold-standard fhir_examples.

Run:
    cd WB_exp/WB_Metadata_Creator
    python test_generators.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0


def _assert(condition: bool, msg: str):
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  ✅ {msg}")
    else:
        _FAIL += 1
        print(f"  ❌ FAIL: {msg}")


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# MOCK DATA — Simulates the BHS Diagnoses table from the gold-standard example
# ══════════════════════════════════════════════════════════════════════════════

from fhir_generator import BQTableInfo, BQColumnInfo, StudyConfig, TableProfile, ColumnProfile, _NUMERIC_BQ_TYPES

MOCK_TABLE = BQTableInfo(
    project_id="verily-bhs",
    dataset_id="analysis",
    table_id="DIAGNOSES",
    row_count=3847,
    size_bytes=524288,
    columns=[
        BQColumnInfo(column_name="STUDYID", data_type="STRING", is_nullable="NO", ordinal_position=1),
        BQColumnInfo(column_name="SITEID", data_type="STRING", is_nullable="NO", ordinal_position=2),
        BQColumnInfo(column_name="SUBJID", data_type="STRING", is_nullable="NO", ordinal_position=3),
        BQColumnInfo(column_name="USUBJID", data_type="STRING", is_nullable="NO", ordinal_position=4),
        BQColumnInfo(column_name="VISIT", data_type="STRING", is_nullable="NO", ordinal_position=5),
        BQColumnInfo(column_name="VISITNUM", data_type="FLOAT64", is_nullable="NO", ordinal_position=6),
        BQColumnInfo(column_name="sex", data_type="STRING", is_nullable="NO", ordinal_position=7),
    ],
)

MOCK_STUDY_CONFIG = StudyConfig(
    study_name="BHS",
    compliance_zone="HIPAA-covered",
    retention_years=7,
    schema_stability="stable",
    domain_contact="domain:bhs-research",
    confidentiality="R",
)

MOCK_TABLE_META = {
    "title": "BHS Analysis — Diagnoses",
    "description": "Clinical diagnoses and demographic data for BHS participants at study start.",
    "purpose": "One record per participant per study visit",
    "primary_key": "USUBJID + VISIT",
}

# Column metadata — simulates what the LLM + user edits produce
# NOTE: Uses SPECIFIC sensitivity codes (P_BIRTHSEX, UID) not generic (PHI)
MOCK_COLUMNS_META = [
    {"Column": "STUDYID", "BQ Type": "STRING", "FHIR Type": "string", "Short Label": "Protocol/Study identifier",
     "Description": "Protocol/Study identifier. Fixed value 'BL001' for the Baseline Health Study.",
     "Required": "Yes", "Null %": "0.0", "Distinct": "1", "Sensitivity": "NONE",
     "Measurement": "administrative", "FHIR Mapping": "", "Coded": "No"},
    {"Column": "SITEID", "BQ Type": "STRING", "FHIR Type": "string", "Short Label": "Site ID",
     "Description": "Clinical site identifier — a 3-digit code identifying the site.",
     "Required": "Yes", "Null %": "0.0", "Distinct": "58", "Sensitivity": "NONE",
     "Measurement": "administrative", "FHIR Mapping": "Location.identifier", "Coded": "No"},
    {"Column": "SUBJID", "BQ Type": "STRING", "FHIR Type": "string", "Short Label": "Study Subject ID",
     "Description": "Study subject identifier — a 10-digit ID uniquely assigned to each participant.",
     "Required": "Yes", "Null %": "0.0", "Distinct": "3847", "Sensitivity": "UID",
     "Measurement": "administrative", "FHIR Mapping": "Patient.identifier", "Coded": "No"},
    {"Column": "USUBJID", "BQ Type": "STRING", "FHIR Type": "string", "Short Label": "Unique Subject ID",
     "Description": "Globally unique subject identifier — composed as STUDYID-SITEID-SUBJID.",
     "Required": "Yes", "Null %": "0.0", "Distinct": "3847", "Sensitivity": "UID",
     "Measurement": "administrative", "FHIR Mapping": "Patient.identifier", "Coded": "No"},
    {"Column": "VISIT", "BQ Type": "STRING", "FHIR Type": "code", "Short Label": "Study visit label",
     "Description": "Study visit label — identifies the study visit at which the diagnosis was recorded.",
     "Required": "Yes", "Null %": "0.0", "Distinct": "1", "Sensitivity": "NONE",
     "Measurement": "administrative", "FHIR Mapping": "Encounter.type", "Coded": "Yes"},
    {"Column": "VISITNUM", "BQ Type": "FLOAT64", "FHIR Type": "decimal", "Short Label": "Numeric visit number",
     "Description": "Ordinal visit number — numeric position of the visit in the study timeline.",
     "Required": "Yes", "Null %": "0.0", "Distinct": "1", "Sensitivity": "NONE",
     "Measurement": "administrative", "FHIR Mapping": "Encounter.period", "Coded": "No"},
    {"Column": "sex", "BQ Type": "STRING", "FHIR Type": "code", "Short Label": "Sex at birth",
     "Description": "Sex at birth of the participant. Coded as 'F' (Female) or 'M' (Male).",
     "Required": "Yes", "Null %": "0.0", "Distinct": "2", "Sensitivity": "P_BIRTHSEX",
     "Measurement": "self-reported", "FHIR Mapping": "Patient.gender", "Coded": "Yes"},
]

MOCK_PROFILE = TableProfile(
    table_name="verily-bhs.analysis.DIAGNOSES",
    total_rows=3847,
    columns={
        "STUDYID": ColumnProfile(column_name="STUDYID", null_count=0, null_percent=0.0, distinct_count=1,
                                  top_values=["BL001"], value_counts={"BL001": 3847},
                                  min_length=5, max_length=5, avg_length=5.0),
        "SITEID": ColumnProfile(column_name="SITEID", null_count=0, null_percent=0.0, distinct_count=58,
                                 top_values=["101", "205", "310"], value_counts={"101": 315, "205": 285, "310": 265},
                                 min_length=3, max_length=3, avg_length=3.0),
        "SUBJID": ColumnProfile(column_name="SUBJID", null_count=0, null_percent=0.0, distinct_count=3847,
                                 min_length=10, max_length=10, avg_length=10.0),
        "USUBJID": ColumnProfile(column_name="USUBJID", null_count=0, null_percent=0.0, distinct_count=3847,
                                  min_length=18, max_length=18, avg_length=18.0),
        "VISIT": ColumnProfile(column_name="VISIT", null_count=0, null_percent=0.0, distinct_count=1,
                                top_values=["Screening Visit"], value_counts={"Screening Visit": 3847},
                                min_length=10, max_length=22, avg_length=14.5),
        "VISITNUM": ColumnProfile(column_name="VISITNUM", null_count=0, null_percent=0.0, distinct_count=1,
                                   min_value=1.0, max_value=1.0, median=1.0, stddev=0.0),
        "sex": ColumnProfile(column_name="sex", null_count=0, null_percent=0.0, distinct_count=2,
                              top_values=["F", "M"], value_counts={"F": 2090, "M": 1757},
                              min_length=1, max_length=1, avg_length=1.0),
    },
)

# Load gold-standard for comparison
_EXAMPLES_DIR = Path(__file__).parent.parent.parent / "product_mgmnt" / "Metadata" / "fhir_examples"


def _load_example(filename: str) -> dict:
    with open(_EXAMPLES_DIR / filename) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: SENSITIVITY MODULE
# ══════════════════════════════════════════════════════════════════════════════

def test_sensitivity_module():
    _section("TEST 1: Sensitivity Module")

    from sensitivity import (
        SENSITIVITY_LABELS,
        is_sensitive,
        get_display,
        build_sec_label_extension,
        sensitivity_comment,
        sensitivity_vocabulary_for_prompt,
    )

    # 1a. CodeSystem loaded with all codes
    _assert(len(SENSITIVITY_LABELS) >= 29, f"CodeSystem has {len(SENSITIVITY_LABELS)} codes (expected ≥29)")

    # 1b. Specific codes exist
    for code in ["UID", "PHI", "PII", "P_BIRTHSEX", "P_RACEETHNICITY", "P_DOB",
                  "P_MRN", "P_PNAME", "P_SSN", "FREETEXT", "FINANCIAL", "PII_BUSINESS"]:
        _assert(code in SENSITIVITY_LABELS, f"Code '{code}' exists in CodeSystem")

    # 1c. is_sensitive works
    _assert(is_sensitive("UID"), "is_sensitive('UID') → True")
    _assert(is_sensitive("P_BIRTHSEX"), "is_sensitive('P_BIRTHSEX') → True")
    _assert(is_sensitive("PHI"), "is_sensitive('PHI') → True")
    _assert(not is_sensitive("NONE"), "is_sensitive('NONE') → False")
    _assert(not is_sensitive(""), "is_sensitive('') → False")

    # 1d. get_display returns correct names
    _assert(get_display("P_BIRTHSEX") == "Patient Birth Sex", f"get_display('P_BIRTHSEX') = '{get_display('P_BIRTHSEX')}'")
    _assert(get_display("UID") == "Unique Identifier", f"get_display('UID') = '{get_display('UID')}'")
    _assert(get_display("FREETEXT") == "Free Text", f"get_display('FREETEXT') = '{get_display('FREETEXT')}'")

    # 1e. build_sec_label_extension
    ext = build_sec_label_extension("P_BIRTHSEX")
    _assert(ext is not None, "build_sec_label_extension('P_BIRTHSEX') returns extension")
    _assert(ext["valueCoding"]["code"] == "P_BIRTHSEX", f"Extension code = '{ext['valueCoding']['code']}'")
    _assert(ext["valueCoding"]["display"] == "Patient Birth Sex", f"Extension display correct")
    _assert(ext["valueCoding"]["system"] == "http://fhir.verily.com/CodeSystem/SensitivityLabels", "Extension system correct")

    ext_none = build_sec_label_extension("NONE")
    _assert(ext_none is None, "build_sec_label_extension('NONE') → None")

    ext_empty = build_sec_label_extension("")
    _assert(ext_empty is None, "build_sec_label_extension('') → None")

    # 1f. sensitivity_comment
    _assert("P_BIRTHSEX" in sensitivity_comment("P_BIRTHSEX"), f"comment includes code: '{sensitivity_comment('P_BIRTHSEX')}'")
    _assert("Patient Birth Sex" in sensitivity_comment("P_BIRTHSEX"), "comment includes display")
    _assert(sensitivity_comment("NONE") == "NOT SENSITIVE", "NONE → 'NOT SENSITIVE'")
    _assert(sensitivity_comment("") == "NOT SENSITIVE", "empty → 'NOT SENSITIVE'")

    # 1g. Prompt vocabulary is generated
    vocab = sensitivity_vocabulary_for_prompt()
    _assert("P_BIRTHSEX" in vocab, "Prompt vocab includes P_BIRTHSEX")
    _assert("P_RACEETHNICITY" in vocab, "Prompt vocab includes P_RACEETHNICITY")
    _assert("FREETEXT" in vocab, "Prompt vocab includes FREETEXT")
    _assert("MOST SPECIFIC" in vocab, "Prompt vocab includes selection rules")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: STRUCTURE DEFINITION BUILDER — SENSITIVITY TAGS
# ══════════════════════════════════════════════════════════════════════════════

def test_structure_definition_sensitivity():
    _section("TEST 2: StructureDefinition Builder — Sensitivity Tags")

    from fhir_generator import build_structure_definition

    sd = build_structure_definition(MOCK_TABLE, MOCK_STUDY_CONFIG, MOCK_TABLE_META, MOCK_COLUMNS_META)

    # Load gold-standard
    gold = _load_example("bhs_diagnoses_structure_definition.json")

    # 2a. Basic structure
    _assert(sd["resourceType"] == "StructureDefinition", "resourceType = StructureDefinition")
    _assert(sd["kind"] == "logical", "kind = logical")

    # 2b. Find elements by column name
    elements = {e["path"].split(".")[-1]: e for e in sd["differential"]["element"] if "." in e.get("path", "")}
    gold_elements = {e["path"].split(".")[-1]: e for e in gold["differential"]["element"] if "." in e.get("path", "")}

    # 2c. sex → P_BIRTHSEX (not generic PHI)
    sex_elem = elements.get("sex")
    _assert(sex_elem is not None, "sex element exists")
    if sex_elem:
        sex_extensions = sex_elem.get("extension", [])
        sec_labels = [e for e in sex_extensions if "inline-sec-label" in e.get("url", "")]
        _assert(len(sec_labels) == 1, f"sex has 1 sec-label extension (got {len(sec_labels)})")
        if sec_labels:
            code = sec_labels[0]["valueCoding"]["code"]
            _assert(code == "P_BIRTHSEX", f"sex sensitivity = '{code}' (expected 'P_BIRTHSEX')")
            display = sec_labels[0]["valueCoding"]["display"]
            _assert(display == "Patient Birth Sex", f"sex display = '{display}'")

        # Compare with gold standard
        gold_sex = gold_elements.get("sex")
        if gold_sex:
            gold_sec_labels = [e for e in gold_sex.get("extension", []) if "inline-sec-label" in e.get("url", "")]
            if gold_sec_labels:
                _assert(
                    code == gold_sec_labels[0]["valueCoding"]["code"],
                    f"sex matches gold standard code: {gold_sec_labels[0]['valueCoding']['code']}"
                )

    # 2d. SUBJID → UID
    subjid_elem = elements.get("SUBJID")
    _assert(subjid_elem is not None, "SUBJID element exists")
    if subjid_elem:
        sec_labels = [e for e in subjid_elem.get("extension", []) if "inline-sec-label" in e.get("url", "")]
        _assert(len(sec_labels) == 1, f"SUBJID has 1 sec-label (got {len(sec_labels)})")
        if sec_labels:
            _assert(sec_labels[0]["valueCoding"]["code"] == "UID", f"SUBJID sensitivity = UID")

    # 2e. USUBJID → UID
    usubjid_elem = elements.get("USUBJID")
    if usubjid_elem:
        sec_labels = [e for e in usubjid_elem.get("extension", []) if "inline-sec-label" in e.get("url", "")]
        _assert(len(sec_labels) == 1, "USUBJID has 1 sec-label")
        if sec_labels:
            _assert(sec_labels[0]["valueCoding"]["code"] == "UID", "USUBJID sensitivity = UID")

    # 2f. Non-sensitive columns should NOT have sec-label
    for col_name in ["STUDYID", "SITEID", "VISIT", "VISITNUM"]:
        elem = elements.get(col_name)
        if elem:
            sec_labels = [e for e in elem.get("extension", []) if "inline-sec-label" in e.get("url", "")]
            _assert(len(sec_labels) == 0, f"{col_name} has NO sec-label (non-sensitive)")
            _assert("NOT SENSITIVE" in elem.get("comment", ""), f"{col_name} comment = 'NOT SENSITIVE'")

    # 2g. sex comment includes P_BIRTHSEX
    if sex_elem:
        _assert("P_BIRTHSEX" in sex_elem.get("comment", ""), f"sex comment includes 'P_BIRTHSEX'")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: DATA PROFILE (MeasureReport) — RICH PER-COLUMN METRICS
# ══════════════════════════════════════════════════════════════════════════════

def test_data_profile():
    _section("TEST 3: Data Profile MeasureReport — Per-Column Metrics")

    from fhir_generator import generate_data_profile

    dp = generate_data_profile(
        table_info=MOCK_TABLE,
        profile_id="bhs-analysis-diagnoses",
        table_profile=MOCK_PROFILE,
        columns_meta=MOCK_COLUMNS_META,
        sd_title="BHS Analysis — Diagnoses",
    )

    # Load gold standard
    gold = _load_example("bhs_diagnoses_data_profile.json")

    # 3a. Basic structure matches gold standard
    _assert(dp is not None, "Data profile generated")
    _assert(dp["resourceType"] == "MeasureReport", "resourceType = MeasureReport")
    _assert(dp["status"] == "complete", "status = complete")
    _assert(dp["type"] == "summary", "type = summary")

    # 3b. Correct measure reference
    _assert(dp["measure"] == "http://fhir.verily.com/Measure/data-profile-tabular",
            f"measure = data-profile-tabular (got {dp['measure']})")
    _assert(dp["measure"] == gold["measure"], "measure matches gold standard")

    # 3c. Meta.profile present
    _assert("meta" in dp, "meta present")
    _assert("profile" in dp.get("meta", {}), "meta.profile present")
    _assert(dp["meta"]["profile"][0] == gold["meta"]["profile"][0], "meta.profile matches gold standard")

    # 3d. Subject reference
    _assert(dp["subject"]["reference"] == "StructureDefinition/bhs-analysis-diagnoses", "subject reference correct")
    _assert(dp["subject"].get("display") == "BHS Analysis — Diagnoses", "subject display present")

    # 3e. Period present
    _assert("period" in dp, "period present")
    _assert("start" in dp["period"], "period.start present")
    _assert("end" in dp["period"], "period.end present")

    # 3f. Has 2 groups: table-metrics and element-metrics
    _assert(len(dp["group"]) == 2, f"Has 2 groups (got {len(dp['group'])})")

    # 3g. Table-level metrics
    table_group = dp["group"][0]
    _assert(table_group["id"] == "table-metrics", f"First group id = 'table-metrics'")
    table_codes = {g["code"]["coding"][0]["code"] for g in table_group.get("group", [])}
    _assert("row-count" in table_codes, "Table metrics include row-count")
    _assert("physical-size" in table_codes, "Table metrics include physical-size")

    # Verify row count value
    row_count_group = next(g for g in table_group["group"] if g["code"]["coding"][0]["code"] == "row-count")
    _assert(row_count_group["measureScore"]["value"] == 3847, f"Row count = 3847 (got {row_count_group['measureScore']['value']})")

    # 3h. Element-level metrics
    element_group = dp["group"][1]
    _assert(element_group["id"] == "element-metrics", f"Second group id = 'element-metrics'")

    # Check element groups exist for each column
    element_ids = {g["id"] for g in element_group.get("group", [])}
    for col_name in ["STUDYID", "SITEID", "SUBJID", "USUBJID", "VISIT", "VISITNUM", "sex"]:
        _assert(col_name in element_ids, f"Element group exists for '{col_name}'")

    # 3i. Per-column metrics structure
    element_by_id = {g["id"]: g for g in element_group.get("group", [])}

    # STUDYID — should have completeness + cardinality + freq-distribution (1 distinct, non-UID)
    studyid = element_by_id.get("STUDYID", {})
    studyid_metrics = {g["code"]["coding"][0]["code"] for g in studyid.get("group", [])}
    _assert("completeness" in studyid_metrics, "STUDYID has completeness")
    _assert("cardinality" in studyid_metrics, "STUDYID has cardinality")
    _assert("freq-distribution" in studyid_metrics, "STUDYID has freq-distribution (non-UID, has top_values)")

    # STUDYID completeness = 100%
    studyid_completeness = next(g for g in studyid["group"] if g["code"]["coding"][0]["code"] == "completeness")
    _assert(studyid_completeness["measureScore"]["value"] == 100.0, "STUDYID completeness = 100%")

    # STUDYID cardinality = ~0.03% (1/3847 * 100)
    studyid_cardinality = next(g for g in studyid["group"] if g["code"]["coding"][0]["code"] == "cardinality")
    _assert(studyid_cardinality["measureScore"]["value"] < 0.1, f"STUDYID cardinality ≈ 0.03% (got {studyid_cardinality['measureScore']['value']})")

    # 3j. SUBJID — UID tagged → NO freq-distribution (de-id safety)
    subjid = element_by_id.get("SUBJID", {})
    subjid_metrics = {g["code"]["coding"][0]["code"] for g in subjid.get("group", [])}
    _assert("completeness" in subjid_metrics, "SUBJID has completeness")
    _assert("cardinality" in subjid_metrics, "SUBJID has cardinality")
    _assert("freq-distribution" not in subjid_metrics, "SUBJID has NO freq-distribution (UID — de-id safe)")

    # USUBJID — also UID, no freq-distribution
    usubjid = element_by_id.get("USUBJID", {})
    usubjid_metrics = {g["code"]["coding"][0]["code"] for g in usubjid.get("group", [])}
    _assert("freq-distribution" not in usubjid_metrics, "USUBJID has NO freq-distribution (UID — de-id safe)")

    # 3k. sex — coded, has freq-distribution
    sex = element_by_id.get("sex", {})
    sex_metrics = {g["code"]["coding"][0]["code"] for g in sex.get("group", [])}
    _assert("completeness" in sex_metrics, "sex has completeness")
    _assert("cardinality" in sex_metrics, "sex has cardinality")
    _assert("freq-distribution" in sex_metrics, "sex has freq-distribution")

    # Check freq-distribution structure
    sex_freq = next((g for g in sex["group"] if g["code"]["coding"][0]["code"] == "freq-distribution"), None)
    if sex_freq:
        strata = sex_freq.get("stratifier", [{}])[0].get("stratum", [])
        _assert(len(strata) == 2, f"sex freq-distribution has 2 strata (F, M) — got {len(strata)}")
        if strata:
            _assert(strata[0]["value"]["text"] == "F", f"First stratum = 'F' (got {strata[0]['value']['text']})")
            _assert("measureScore" in strata[0], "Stratum has measureScore (percentage)")
            # Check ordinal codes
            ordinal_code = strata[0]["component"][0]["code"]["coding"][0]["display"]
            _assert(ordinal_code == "First", f"First stratum ordinal = 'First' (got {ordinal_code})")

    # 3l. SITEID — has top values, should have freq-distribution with up to 3 strata
    siteid = element_by_id.get("SITEID", {})
    siteid_freq = next((g for g in siteid.get("group", []) if g["code"]["coding"][0]["code"] == "freq-distribution"), None)
    if siteid_freq:
        strata = siteid_freq.get("stratifier", [{}])[0].get("stratum", [])
        _assert(len(strata) == 3, f"SITEID freq-distribution has 3 strata (top 3) — got {len(strata)}")

    # 3m. Compare structure with gold standard
    gold_table_group = gold["group"][0]
    gold_element_group = gold["group"][1]
    _assert(table_group["code"]["coding"][0]["system"] == gold_table_group["code"]["coding"][0]["system"],
            "Table metrics CodeSystem matches gold standard")
    _assert(element_group["code"]["coding"][0]["system"] == gold_element_group["code"]["coding"][0]["system"],
            "Element metrics CodeSystem matches gold standard")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: DATA PROFILE — BACKWARD COMPATIBLE (no TableProfile)
# ══════════════════════════════════════════════════════════════════════════════

def test_data_profile_backward_compat():
    _section("TEST 4: Data Profile — Backward Compatible (no TableProfile)")

    from fhir_generator import generate_data_profile

    # Without table_profile, should produce table-level only (backward compat)
    dp = generate_data_profile(MOCK_TABLE, "bhs-analysis-diagnoses")

    _assert(dp is not None, "Data profile generated without table_profile")
    _assert(dp["resourceType"] == "MeasureReport", "resourceType = MeasureReport")
    _assert(len(dp["group"]) == 1, f"Only 1 group (table-metrics) without profile — got {len(dp['group'])}")
    _assert(dp["measure"] == "http://fhir.verily.com/Measure/data-profile-tabular", "measure URL updated to tabular")

    # No table_info either → None
    from fhir_generator import BQTableInfo
    empty_table = BQTableInfo(project_id="p", dataset_id="d", table_id="t")
    dp_none = generate_data_profile(empty_table, "test-id")
    _assert(dp_none is None, "No stats → returns None")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: TERMINOLOGY BUNDLE — CodeSystem + ValueSet
# ══════════════════════════════════════════════════════════════════════════════

def test_terminology_bundle():
    _section("TEST 5: Terminology Bundle — CodeSystem + ValueSet")

    from fhir_generator import build_terminology_bundle, build_code_system

    # Load gold standard
    gold = _load_example("bhs_diagnoses_terminology.json")

    # 5a. Build individual CodeSystem
    cs = build_code_system("bhs-analysis-diagnoses", "sex", ["F", "M"], "BHS", "Sex at birth of the participant.")
    _assert(cs["resourceType"] == "CodeSystem", "CodeSystem resourceType correct")
    _assert(cs["content"] == "complete", "CodeSystem content = complete")
    _assert(cs["count"] == 2, f"CodeSystem count = 2 (got {cs['count']})")
    _assert(len(cs["concept"]) == 2, f"CodeSystem has 2 concepts (got {len(cs['concept'])})")
    _assert(cs["concept"][0]["code"] == "F", f"First concept = 'F'")
    _assert(cs["concept"][1]["code"] == "M", f"Second concept = 'M'")

    # 5b. Build terminology bundle
    coded_columns = [
        {"column_name": "VISIT", "coded_values": ["Screening Visit"], "description": "Study visit label"},
        {"column_name": "sex", "coded_values": ["F", "M"], "description": "Sex at birth of the participant."},
    ]
    bundle = build_terminology_bundle("bhs-analysis-diagnoses", "BHS", coded_columns)

    _assert(bundle is not None, "Terminology bundle generated")
    _assert(bundle["resourceType"] == "Bundle", "Bundle resourceType correct")
    _assert(bundle["type"] == "collection", "Bundle type = collection")
    _assert(bundle["resourceType"] == gold["resourceType"], "Bundle resourceType matches gold standard")
    _assert(bundle["type"] == gold["type"], "Bundle type matches gold standard")

    # 5c. Bundle has correct number of entries (2 coded cols × 2 resources each = 4)
    entries = bundle.get("entry", [])
    _assert(len(entries) == 4, f"Bundle has 4 entries (2 CS + 2 VS) — got {len(entries)}")

    # 5d. Entry types: alternating CodeSystem, ValueSet
    resource_types = [e["resource"]["resourceType"] for e in entries]
    _assert(resource_types == ["CodeSystem", "ValueSet", "CodeSystem", "ValueSet"],
            f"Entry order: CS, VS, CS, VS — got {resource_types}")

    # 5e. CodeSystem structure matches gold standard pattern
    cs_entries = [e["resource"] for e in entries if e["resource"]["resourceType"] == "CodeSystem"]
    sex_cs = next((cs for cs in cs_entries if "sex" in cs.get("id", "")), None)
    _assert(sex_cs is not None, "sex CodeSystem found in bundle")

    # Compare with gold standard CodeSystem
    gold_entries = gold.get("entry", [])
    gold_cs_entries = [e["resource"] for e in gold_entries if e["resource"]["resourceType"] == "CodeSystem"]
    if gold_cs_entries:
        gold_sex_cs = gold_cs_entries[0]  # First CS in gold standard
        _assert("concept" in sex_cs, "CodeSystem has concept array")
        _assert("content" in sex_cs, "CodeSystem has content field")
        _assert(sex_cs["content"] == gold_sex_cs["content"], f"content = '{sex_cs['content']}' matches gold standard")

    # 5f. ValueSet structure — has compose AND expansion
    vs_entries = [e["resource"] for e in entries if e["resource"]["resourceType"] == "ValueSet"]
    sex_vs = next((vs for vs in vs_entries if "sex" in vs.get("id", "")), None)
    _assert(sex_vs is not None, "sex ValueSet found in bundle")
    if sex_vs:
        _assert("compose" in sex_vs, "ValueSet has compose")
        _assert("expansion" in sex_vs, "ValueSet has expansion (gold standard feature)")
        expansion = sex_vs["expansion"]
        _assert("timestamp" in expansion, "expansion has timestamp")
        _assert("total" in expansion, "expansion has total")
        _assert("contains" in expansion, "expansion has contains")
        _assert(expansion["total"] == 2, f"expansion total = 2 (got {expansion['total']})")

        # Compare with gold standard ValueSet
        gold_vs_entries = [e["resource"] for e in gold_entries if e["resource"]["resourceType"] == "ValueSet"]
        if gold_vs_entries:
            gold_vs = gold_vs_entries[0]
            _assert("expansion" in gold_vs, "Gold standard has expansion too")

    # 5g. fullUrl conventions match
    for entry in entries:
        _assert("fullUrl" in entry, "Each entry has fullUrl")
        _assert(entry["fullUrl"].startswith("http://fhir.verily.com/"), f"fullUrl starts with verily namespace")

    # 5h. Bundle meta.lastUpdated
    _assert("meta" in bundle, "Bundle has meta")
    _assert("lastUpdated" in bundle.get("meta", {}), "Bundle meta has lastUpdated")

    # 5i. Empty input returns None
    empty_bundle = build_terminology_bundle("test", "TEST", [])
    _assert(empty_bundle is None, "Empty coded_columns → None")

    # 5j. Columns with no values get skipped
    partial = build_terminology_bundle("test", "TEST", [
        {"column_name": "col1", "coded_values": [], "description": "empty"},
        {"column_name": "col2", "coded_values": ["A", "B"], "description": "has values"},
    ])
    _assert(partial is not None, "Bundle with partial coded columns generated")
    _assert(len(partial["entry"]) == 2, f"Only 1 coded column → 2 entries (CS + VS) — got {len(partial['entry'])}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: END-TO-END — All artifacts for one table
# ══════════════════════════════════════════════════════════════════════════════

def test_end_to_end():
    _section("TEST 6: End-to-End — All artifacts for BHS DIAGNOSES table")

    from fhir_generator import (
        build_structure_definition,
        build_terminology_bundle,
        generate_data_profile,
    )

    # Build all 3 artifacts
    sd = build_structure_definition(MOCK_TABLE, MOCK_STUDY_CONFIG, MOCK_TABLE_META, MOCK_COLUMNS_META)
    profile_id = sd["id"]

    coded_columns = [
        {"column_name": "VISIT", "coded_values": ["Screening Visit"], "description": "Study visit label"},
        {"column_name": "sex", "coded_values": ["F", "M"], "description": "Sex at birth"},
    ]
    term_bundle = build_terminology_bundle(profile_id, "BHS", coded_columns)
    dp = generate_data_profile(MOCK_TABLE, profile_id, MOCK_PROFILE, MOCK_COLUMNS_META, sd.get("title", ""))

    # 6a. All artifacts generated
    _assert(sd is not None, "StructureDefinition generated")
    _assert(term_bundle is not None, "Terminology Bundle generated")
    _assert(dp is not None, "DataProfile MeasureReport generated")

    # 6b. Cross-references are consistent
    _assert(dp["subject"]["reference"] == f"StructureDefinition/{profile_id}", "DataProfile references StructureDefinition")
    _assert(dp["measure"] == "http://fhir.verily.com/Measure/data-profile-tabular", "DataProfile references correct Measure")

    # 6c. StructureDefinition coded columns reference ValueSets
    elements = sd["differential"]["element"]
    sex_elem = next((e for e in elements if e["path"].endswith(".sex")), None)
    _assert(sex_elem is not None and "binding" in sex_elem, "sex element has binding")
    if sex_elem and "binding" in sex_elem:
        vs_url = sex_elem["binding"]["valueSet"]
        _assert("ValueSet" in vs_url, f"sex binding references ValueSet: {vs_url}")

    # 6d. Terminology bundle contains resources for coded columns
    cs_ids = [e["resource"]["id"] for e in term_bundle["entry"] if e["resource"]["resourceType"] == "CodeSystem"]
    vs_ids = [e["resource"]["id"] for e in term_bundle["entry"] if e["resource"]["resourceType"] == "ValueSet"]
    _assert(any("sex" in cid for cid in cs_ids), f"Terminology bundle has sex CodeSystem")
    _assert(any("sex" in vid for vid in vs_ids), f"Terminology bundle has sex ValueSet")

    # 6e. Gold standard file count: SD + Terminology + DataProfile + Measure = 4 resource types
    _assert(sd["resourceType"] == "StructureDefinition", "SD type check")
    _assert(term_bundle["resourceType"] == "Bundle", "Term bundle type check")
    _assert(dp["resourceType"] == "MeasureReport", "DP type check")
    print(f"\n  📦 Generated 3 artifacts: {sd['id']}.json, {term_bundle['id']}.json, {dp['id']}.json")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: CONCEPT MAP — Pure builder (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def test_concept_map_builder():
    _section("TEST 7: ConceptMap Builder (no LLM)")

    from fhir_generator import build_concept_map

    # Load gold standard
    gold = _load_example("bhs_diagnoses_terminology.json")
    gold_cm_entries = [e["resource"] for e in gold["entry"] if e["resource"]["resourceType"] == "ConceptMap"]

    # Build a ConceptMap for sex → AdministrativeGender
    cm = build_concept_map(
        study_name="BHS",
        column_name="sex",
        source_codes=[{"code": "F", "display": "Female"}, {"code": "M", "display": "Male"}],
        target_system="http://hl7.org/fhir/administrative-gender",
        target_valueset="http://hl7.org/fhir/ValueSet/administrative-gender",
        target_codes=[
            {"source_code": "F", "target_code": "female", "target_display": "Female", "equivalence": "equivalent"},
            {"source_code": "M", "target_code": "male", "target_display": "Male", "equivalence": "equivalent"},
        ],
        description="Maps the physical sex-at-birth codes in BHS data (F, M) to the standard FHIR AdministrativeGender code system.",
    )

    # 7a. Basic structure
    _assert(cm["resourceType"] == "ConceptMap", "resourceType = ConceptMap")
    _assert(cm["status"] == "active", "status = active")

    # 7b. Source/target URIs
    _assert(cm["sourceUri"] == "http://fhir.verily.com/ValueSet/bhs-sex", f"sourceUri correct: {cm['sourceUri']}")
    _assert(cm["targetUri"] == "http://hl7.org/fhir/ValueSet/administrative-gender", "targetUri correct")

    # Compare with gold standard
    if gold_cm_entries:
        gold_cm = gold_cm_entries[0]
        _assert(cm["sourceUri"] == gold_cm["sourceUri"], "sourceUri matches gold standard")
        _assert(cm["targetUri"] == gold_cm["targetUri"], "targetUri matches gold standard")

    # 7c. Group structure
    _assert(len(cm["group"]) == 1, f"Has 1 group (got {len(cm['group'])})")
    group = cm["group"][0]
    _assert(group["source"] == "http://fhir.verily.com/CodeSystem/bhs-sex", "group source correct")
    _assert(group["target"] == "http://hl7.org/fhir/administrative-gender", "group target correct")

    # Compare with gold standard
    if gold_cm_entries:
        gold_group = gold_cm_entries[0]["group"][0]
        _assert(group["source"] == gold_group["source"], "group source matches gold standard")
        _assert(group["target"] == gold_group["target"], "group target matches gold standard")

    # 7d. Element mappings
    elements = group["element"]
    _assert(len(elements) == 2, f"Has 2 elements (F, M) — got {len(elements)}")

    f_elem = next((e for e in elements if e["code"] == "F"), None)
    _assert(f_elem is not None, "F element exists")
    if f_elem:
        _assert(f_elem["display"] == "Female", f"F display = 'Female'")
        _assert(f_elem["target"][0]["code"] == "female", "F → female")
        _assert(f_elem["target"][0]["equivalence"] == "equivalent", "F equivalence = equivalent")

    m_elem = next((e for e in elements if e["code"] == "M"), None)
    _assert(m_elem is not None, "M element exists")
    if m_elem:
        _assert(m_elem["target"][0]["code"] == "male", "M → male")
        _assert(m_elem["target"][0]["equivalence"] == "equivalent", "M equivalence = equivalent")

    # Compare element structure with gold standard
    if gold_cm_entries:
        gold_elements = gold_cm_entries[0]["group"][0]["element"]
        gold_f = next((e for e in gold_elements if e["code"] == "F"), None)
        if gold_f:
            _assert(f_elem["target"][0]["code"] == gold_f["target"][0]["code"], "F mapping matches gold standard")
            _assert(f_elem["target"][0]["equivalence"] == gold_f["target"][0]["equivalence"], "F equivalence matches gold standard")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: CONCEPT MAP — LLM Response Parser
# ══════════════════════════════════════════════════════════════════════════════

def test_concept_map_parser():
    _section("TEST 8: ConceptMap LLM Response Parser")

    from fhir_generator import _parse_concept_map_response

    # 8a. Parse a well-formed response (array of {column_name, confidence, concept_map})
    mock_llm_response = '''```json
[
  {
    "column_name": "sex",
    "confidence": "high",
    "target_system_name": "HL7 Administrative Gender",
    "concept_map": {
      "resourceType": "ConceptMap",
      "id": "bhs-sex-to-administrative-gender",
      "url": "http://fhir.verily.com/ConceptMap/bhs-sex-to-administrative-gender",
      "name": "BHSSexToAdministrativeGender",
      "title": "BHS Sex → FHIR AdministrativeGender",
      "status": "active",
      "description": "Maps F, M to female, male",
      "sourceUri": "http://fhir.verily.com/ValueSet/bhs-sex",
      "targetUri": "http://hl7.org/fhir/ValueSet/administrative-gender",
      "group": [
        {
          "source": "http://fhir.verily.com/CodeSystem/bhs-sex",
          "target": "http://hl7.org/fhir/administrative-gender",
          "element": [
            {
              "code": "F",
              "display": "Female",
              "target": [{"code": "female", "display": "Female", "equivalence": "equivalent"}]
            },
            {
              "code": "M",
              "display": "Male",
              "target": [{"code": "male", "display": "Male", "equivalence": "equivalent"}]
            }
          ]
        }
      ]
    }
  }
]
```'''

    result = _parse_concept_map_response(mock_llm_response)
    _assert(len(result) == 1, f"Parsed 1 ConceptMap (got {len(result)})")
    if result:
        cm = result[0]
        _assert(cm["resourceType"] == "ConceptMap", "Parsed resourceType = ConceptMap")
        _assert(cm["id"] == "bhs-sex-to-administrative-gender", f"Parsed id correct")
        _assert(len(cm["group"][0]["element"]) == 2, "Parsed 2 elements")
        # New enriched fields
        _assert(cm.get("_column_name") == "sex", f"_column_name = 'sex'")
        _assert(cm.get("_confidence") == "high", f"_confidence = 'high'")
        _assert(cm.get("_target_system_name") == "HL7 Administrative Gender", f"_target_system_name correct")

    # 8b. Empty array (no mappings)
    empty_result = _parse_concept_map_response("```json\n[]\n```")
    _assert(len(empty_result) == 0, "Empty array → 0 ConceptMaps")

    # 8c. Garbled response
    garbled_result = _parse_concept_map_response("I'm sorry, I can't help with that.")
    _assert(len(garbled_result) == 0, "Garbled response → 0 ConceptMaps")

    # 8d. Multiple ConceptMaps
    multi_response = '''```json
[
  {
    "column_name": "sex",
    "confidence": "high",
    "target_system_name": "AdminGender",
    "concept_map": {
      "resourceType": "ConceptMap", "id": "cm1", "status": "active",
      "sourceUri": "x", "targetUri": "y", "group": []
    }
  },
  {
    "column_name": "visit",
    "confidence": "low",
    "target_system_name": "EncounterType",
    "concept_map": {
      "resourceType": "ConceptMap", "id": "cm2", "status": "active",
      "sourceUri": "a", "targetUri": "b", "group": []
    }
  }
]
```'''
    multi_result = _parse_concept_map_response(multi_response)
    _assert(len(multi_result) == 2, f"Parsed 2 ConceptMaps (got {len(multi_result)})")
    if len(multi_result) == 2:
        _assert(multi_result[0].get("_column_name") == "sex", "First CM _column_name = sex")
        _assert(multi_result[1].get("_confidence") == "low", "Second CM _confidence = low")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: TERMINOLOGY BUNDLE WITH CONCEPT MAPS
# ══════════════════════════════════════════════════════════════════════════════

def test_terminology_bundle_with_concept_maps():
    _section("TEST 9: Terminology Bundle with ConceptMaps")

    from fhir_generator import build_terminology_bundle_with_concept_maps, build_concept_map

    # Load gold standard
    gold = _load_example("bhs_diagnoses_terminology.json")

    coded_columns = [
        {"column_name": "sex", "coded_values": ["F", "M"], "description": "Sex at birth"},
    ]

    # Build a ConceptMap (simulating LLM output)
    sex_cm = build_concept_map(
        study_name="BHS",
        column_name="sex",
        source_codes=[{"code": "F", "display": "Female"}, {"code": "M", "display": "Male"}],
        target_system="http://hl7.org/fhir/administrative-gender",
        target_valueset="http://hl7.org/fhir/ValueSet/administrative-gender",
        target_codes=[
            {"source_code": "F", "target_code": "female", "target_display": "Female", "equivalence": "equivalent"},
            {"source_code": "M", "target_code": "male", "target_display": "Male", "equivalence": "equivalent"},
        ],
    )

    bundle = build_terminology_bundle_with_concept_maps(
        "bhs-analysis-diagnoses", "BHS", coded_columns, [sex_cm]
    )

    _assert(bundle is not None, "Bundle generated")
    _assert(bundle["resourceType"] == "Bundle", "Bundle type correct")

    # 9a. Bundle has 3 entries: CodeSystem + ValueSet + ConceptMap
    entries = bundle.get("entry", [])
    _assert(len(entries) == 3, f"Bundle has 3 entries (CS + VS + CM) — got {len(entries)}")

    resource_types = [e["resource"]["resourceType"] for e in entries]
    _assert("CodeSystem" in resource_types, "Bundle contains CodeSystem")
    _assert("ValueSet" in resource_types, "Bundle contains ValueSet")
    _assert("ConceptMap" in resource_types, "Bundle contains ConceptMap")

    # 9b. Compare with gold standard — it has CS + VS + CM for sex + CS + VS for visit = 5 entries
    gold_resource_types = [e["resource"]["resourceType"] for e in gold["entry"]]
    _assert("ConceptMap" in gold_resource_types, "Gold standard also has ConceptMap")

    # 9c. ConceptMap entry has correct fullUrl
    cm_entry = next(e for e in entries if e["resource"]["resourceType"] == "ConceptMap")
    _assert(cm_entry["fullUrl"].startswith("http://fhir.verily.com/ConceptMap/"), "ConceptMap fullUrl correct")

    # 9d. Without concept_maps, should still work (just CS + VS)
    bundle_no_cm = build_terminology_bundle_with_concept_maps(
        "bhs-analysis-diagnoses", "BHS", coded_columns, None
    )
    _assert(bundle_no_cm is not None, "Bundle without CMs generated")
    _assert(len(bundle_no_cm["entry"]) == 2, f"Bundle without CMs has 2 entries (CS + VS) — got {len(bundle_no_cm['entry'])}")

    # 9e. LLM-parsed ConceptMaps with _prefixed metadata are stripped before bundling
    enriched_cm = dict(sex_cm)
    enriched_cm["_column_name"] = "sex"
    enriched_cm["_confidence"] = "high"
    enriched_cm["_target_system_name"] = "HL7 Administrative Gender"
    bundle_enriched = build_terminology_bundle_with_concept_maps(
        "bhs-analysis-diagnoses", "BHS", coded_columns, [enriched_cm]
    )
    cm_in_bundle = next(e["resource"] for e in bundle_enriched["entry"] if e["resource"]["resourceType"] == "ConceptMap")
    _assert("_column_name" not in cm_in_bundle, "Internal _column_name stripped from bundle")
    _assert("_confidence" not in cm_in_bundle, "Internal _confidence stripped from bundle")
    _assert("_target_system_name" not in cm_in_bundle, "Internal _target_system_name stripped from bundle")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: MEASURE DEFINITION (data-profile-tabular)
# ══════════════════════════════════════════════════════════════════════════════

def test_measure_definition():
    _section("TEST 10: Measure Definition (data-profile-tabular)")

    from fhir_generator import build_measure_definition

    measure = build_measure_definition()

    # Load gold standard
    gold = _load_example("data_profile_measure.json")

    # 10a. Basic structure matches
    _assert(measure["resourceType"] == "Measure", "resourceType = Measure")
    _assert(measure["resourceType"] == gold["resourceType"], "resourceType matches gold standard")
    _assert(measure["id"] == "data-profile-tabular", f"id = data-profile-tabular")
    _assert(measure["id"] == gold["id"], "id matches gold standard")
    _assert(measure["url"] == gold["url"], "url matches gold standard")
    _assert(measure["name"] == gold["name"], "name matches gold standard")
    _assert(measure["status"] == gold["status"], "status matches gold standard")

    # 10b. Has 2 groups: table-metrics and element-metrics
    _assert(len(measure["group"]) == 2, f"Has 2 groups (got {len(measure['group'])})")
    _assert(len(measure["group"]) == len(gold["group"]), "Same number of groups as gold standard")

    # 10c. Table metrics group matches gold standard
    table_group = measure["group"][0]
    gold_table_group = gold["group"][0]
    _assert(table_group["id"] == gold_table_group["id"], f"Table group id matches: {table_group['id']}")

    table_metric_ids = {g["id"] for g in table_group["group"]}
    gold_table_ids = {g["id"] for g in gold_table_group["group"]}
    _assert(table_metric_ids == gold_table_ids, f"Table metric ids match: {table_metric_ids}")

    # 10d. Element metrics group matches gold standard
    elem_group = measure["group"][1]
    gold_elem_group = gold["group"][1]
    _assert(elem_group["id"] == gold_elem_group["id"], f"Element group id matches: {elem_group['id']}")

    elem_metric_ids = {g["id"] for g in elem_group["group"]}
    gold_elem_ids = {g["id"] for g in gold_elem_group["group"]}
    _assert(elem_metric_ids == gold_elem_ids, f"Element metric ids match gold standard: {elem_metric_ids}")

    # 10e. freq-distribution has stratifier with SNOMED ordinal codes
    freq_def = next(g for g in elem_group["group"] if g["id"] == "freq-distribution")
    gold_freq_def = next(g for g in gold_elem_group["group"] if g["id"] == "freq-distribution")
    _assert("stratifier" in freq_def, "freq-distribution has stratifier")
    _assert(len(freq_def["stratifier"][0]["component"]) == 3, "stratifier has 3 ordinal components")
    _assert(
        freq_def["stratifier"][0]["component"][0]["code"]["coding"][0]["code"]
        == gold_freq_def["stratifier"][0]["component"][0]["code"]["coding"][0]["code"],
        "First ordinal SNOMED code matches gold standard"
    )

    # 10f. All metric descriptions match
    for g, gold_g in zip(elem_group["group"], gold_elem_group["group"]):
        if "description" in gold_g:
            _assert(g.get("description") == gold_g["description"],
                    f"Description for '{g['id']}' matches gold standard")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: DEEPER PROFILING — String length + Numeric stats in MeasureReport
# ══════════════════════════════════════════════════════════════════════════════

def test_deeper_profiling_in_measure_report():
    _section("TEST 11: Deeper Profiling — String Length + Numeric Stats")

    from fhir_generator import generate_data_profile

    dp = generate_data_profile(
        table_info=MOCK_TABLE,
        profile_id="bhs-analysis-diagnoses",
        table_profile=MOCK_PROFILE,
        columns_meta=MOCK_COLUMNS_META,
        sd_title="BHS Analysis — Diagnoses",
    )

    # Load gold standard for comparison
    gold = _load_example("bhs_diagnoses_data_profile.json")

    element_group = dp["group"][1]
    elements_by_id = {g["id"]: g for g in element_group["group"]}

    gold_element_group = gold["group"][1]
    gold_by_id = {g["id"]: g for g in gold_element_group["group"]}

    # 11a. String columns: STUDYID should have min-length, max-length, avg-length
    studyid = elements_by_id.get("STUDYID", {})
    studyid_metrics = {g["code"]["coding"][0]["code"]: g for g in studyid.get("group", [])}
    _assert("min-length" in studyid_metrics, "STUDYID has min-length")
    _assert("max-length" in studyid_metrics, "STUDYID has max-length")
    _assert("avg-length" in studyid_metrics, "STUDYID has avg-length")

    # Check values
    _assert(studyid_metrics["min-length"]["measureScore"]["value"] == 5, "STUDYID min-length = 5")
    _assert(studyid_metrics["max-length"]["measureScore"]["value"] == 5, "STUDYID max-length = 5")
    _assert(studyid_metrics["avg-length"]["measureScore"]["value"] == 5.0, "STUDYID avg-length = 5.0")

    # Compare with gold standard
    gold_studyid = gold_by_id.get("STUDYID", {})
    gold_studyid_metrics = {g["code"]["coding"][0]["code"]: g for g in gold_studyid.get("group", [])}
    if "min-length" in gold_studyid_metrics:
        _assert(
            studyid_metrics["min-length"]["measureScore"]["value"]
            == gold_studyid_metrics["min-length"]["measureScore"]["value"],
            "STUDYID min-length matches gold standard"
        )

    # 11b. Characters unit (no UCUM)
    _assert(studyid_metrics["min-length"]["measureScore"]["unit"] == "characters",
            "min-length unit = 'characters'")
    _assert("system" not in studyid_metrics["min-length"]["measureScore"],
            "min-length has no UCUM system (matches gold standard)")

    # 11c. sex has string length stats (coded STRING column)
    sex = elements_by_id.get("sex", {})
    sex_metrics = {g["code"]["coding"][0]["code"]: g for g in sex.get("group", [])}
    _assert("min-length" in sex_metrics, "sex has min-length")
    _assert(sex_metrics["min-length"]["measureScore"]["value"] == 1, "sex min-length = 1 (single char)")

    # 11d. SITEID has string length stats
    siteid = elements_by_id.get("SITEID", {})
    siteid_metrics = {g["code"]["coding"][0]["code"]: g for g in siteid.get("group", [])}
    _assert("min-length" in siteid_metrics, "SITEID has min-length")
    _assert(siteid_metrics["min-length"]["measureScore"]["value"] == 3, "SITEID min-length = 3")

    # 11e. Numeric column: VISITNUM should have min-value, max-value, median, stddev
    visitnum = elements_by_id.get("VISITNUM", {})
    visitnum_metrics = {g["code"]["coding"][0]["code"]: g for g in visitnum.get("group", [])}
    _assert("min-value" in visitnum_metrics, "VISITNUM has min-value")
    _assert("max-value" in visitnum_metrics, "VISITNUM has max-value")
    _assert("median" in visitnum_metrics, "VISITNUM has median")
    _assert("stddev" in visitnum_metrics, "VISITNUM has stddev")

    # Check values
    _assert(visitnum_metrics["min-value"]["measureScore"]["value"] == 1.0, "VISITNUM min-value = 1.0")
    _assert(visitnum_metrics["max-value"]["measureScore"]["value"] == 1.0, "VISITNUM max-value = 1.0")
    _assert(visitnum_metrics["median"]["measureScore"]["value"] == 1.0, "VISITNUM median = 1.0")
    _assert(visitnum_metrics["stddev"]["measureScore"]["value"] == 0.0, "VISITNUM stddev = 0.0")

    # Compare with gold standard
    gold_visitnum = gold_by_id.get("VISITNUM", {})
    gold_visitnum_metrics = {g["code"]["coding"][0]["code"]: g for g in gold_visitnum.get("group", [])}
    if "min-value" in gold_visitnum_metrics:
        _assert(
            visitnum_metrics["min-value"]["measureScore"]["value"]
            == gold_visitnum_metrics["min-value"]["measureScore"]["value"],
            "VISITNUM min-value matches gold standard"
        )
    if "stddev" in gold_visitnum_metrics:
        _assert(
            visitnum_metrics["stddev"]["measureScore"]["value"]
            == gold_visitnum_metrics["stddev"]["measureScore"]["value"],
            "VISITNUM stddev matches gold standard"
        )

    # 11f. Numeric columns should NOT have string length stats
    _assert("min-length" not in visitnum_metrics, "VISITNUM has NO min-length (numeric)")
    _assert("max-length" not in visitnum_metrics, "VISITNUM has NO max-length (numeric)")
    _assert("avg-length" not in visitnum_metrics, "VISITNUM has NO avg-length (numeric)")

    # 11g. String columns should NOT have numeric stats
    _assert("min-value" not in studyid_metrics, "STUDYID has NO min-value (string)")
    _assert("max-value" not in studyid_metrics, "STUDYID has NO max-value (string)")
    _assert("median" not in studyid_metrics, "STUDYID has NO median (string)")

    # 11h. Freq-distribution now uses precise percentages from value_counts
    sex_freq = next((g for g in sex.get("group", []) if g["code"]["coding"][0]["code"] == "freq-distribution"), None)
    if sex_freq:
        strata = sex_freq["stratifier"][0]["stratum"]
        f_stratum = next(s for s in strata if s["value"]["text"] == "F")
        # F count: 2090 / 3847 = 54.3%
        f_pct = f_stratum["measureScore"]["value"]
        _assert(abs(f_pct - 54.3) < 0.2, f"F percentage ≈ 54.3% (got {f_pct}) — uses value_counts")

        m_stratum = next(s for s in strata if s["value"]["text"] == "M")
        m_pct = m_stratum["measureScore"]["value"]
        _assert(abs(m_pct - 45.7) < 0.2, f"M percentage ≈ 45.7% (got {m_pct}) — uses value_counts")

    # 11i. Full metric set per column type (gold standard parity check)
    print("\n  📊 Metric summary per column:")
    for col_id, elem in elements_by_id.items():
        metric_codes = [g["code"]["coding"][0]["code"] for g in elem.get("group", [])]
        print(f"    {col_id}: {', '.join(metric_codes)}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: FULL END-TO-END WITH ALL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def test_full_end_to_end():
    _section("TEST 12: Full End-to-End — All 4 artifact types")

    from fhir_generator import (
        build_structure_definition,
        build_terminology_bundle_with_concept_maps,
        build_concept_map,
        generate_data_profile,
        build_measure_definition,
    )

    # Build all 4 artifact types
    sd = build_structure_definition(MOCK_TABLE, MOCK_STUDY_CONFIG, MOCK_TABLE_META, MOCK_COLUMNS_META)
    profile_id = sd["id"]

    coded_columns = [
        {"column_name": "VISIT", "coded_values": ["Screening Visit"], "description": "Study visit label"},
        {"column_name": "sex", "coded_values": ["F", "M"], "description": "Sex at birth"},
    ]

    sex_cm = build_concept_map(
        study_name="BHS", column_name="sex",
        source_codes=[{"code": "F", "display": "Female"}, {"code": "M", "display": "Male"}],
        target_system="http://hl7.org/fhir/administrative-gender",
        target_valueset="http://hl7.org/fhir/ValueSet/administrative-gender",
        target_codes=[
            {"source_code": "F", "target_code": "female", "target_display": "Female", "equivalence": "equivalent"},
            {"source_code": "M", "target_code": "male", "target_display": "Male", "equivalence": "equivalent"},
        ],
    )

    term_bundle = build_terminology_bundle_with_concept_maps(profile_id, "BHS", coded_columns, [sex_cm])
    dp = generate_data_profile(MOCK_TABLE, profile_id, MOCK_PROFILE, MOCK_COLUMNS_META, sd.get("title", ""))
    measure = build_measure_definition()

    # 12a. All artifacts generated
    _assert(sd is not None, "StructureDefinition generated")
    _assert(term_bundle is not None, "Terminology Bundle (with ConceptMap) generated")
    _assert(dp is not None, "DataProfile MeasureReport generated")
    _assert(measure is not None, "Measure definition generated")

    # 12b. MeasureReport.measure points to Measure.url
    _assert(dp["measure"] == measure["url"], "MeasureReport.measure == Measure.url")

    # 12c. Full artifact inventory matches gold standard set
    artifact_types = {
        "StructureDefinition": sd["resourceType"],
        "Bundle (terminology)": term_bundle["resourceType"],
        "MeasureReport": dp["resourceType"],
        "Measure": measure["resourceType"],
    }
    _assert(len(artifact_types) == 4, f"4 distinct artifact types generated")
    print(f"\n  📦 Artifact inventory:")
    for label, rt in artifact_types.items():
        print(f"    ✓ {label} → {rt}")

    # 12d. Terminology bundle has all 3 resource types
    term_rtypes = {e["resource"]["resourceType"] for e in term_bundle["entry"]}
    _assert(term_rtypes == {"CodeSystem", "ValueSet", "ConceptMap"},
            f"Terminology bundle has CS + VS + CM: {term_rtypes}")

    # 12e. DataProfile has per-column metrics with type-appropriate stats
    elem_group = dp["group"][1]
    visitnum_elem = next((g for g in elem_group["group"] if g["id"] == "VISITNUM"), None)
    sex_elem = next((g for g in elem_group["group"] if g["id"] == "sex"), None)
    if visitnum_elem:
        vm_codes = {g["code"]["coding"][0]["code"] for g in visitnum_elem["group"]}
        _assert("min-value" in vm_codes, "VISITNUM has numeric stats in final output")
    if sex_elem:
        sm_codes = {g["code"]["coding"][0]["code"] for g in sex_elem["group"]}
        _assert("min-length" in sm_codes, "sex has string stats in final output")
        _assert("freq-distribution" in sm_codes, "sex has freq-distribution in final output")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🧪 FHIR Metadata Generator — Backend Tests")
    print("=" * 60)

    test_sensitivity_module()
    test_structure_definition_sensitivity()
    test_data_profile()
    test_data_profile_backward_compat()
    test_terminology_bundle()
    test_end_to_end()
    test_concept_map_builder()
    test_concept_map_parser()
    test_terminology_bundle_with_concept_maps()
    test_measure_definition()
    test_deeper_profiling_in_measure_report()
    test_full_end_to_end()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {_PASS} passed, {_FAIL} failed")
    print(f"{'='*60}\n")

    sys.exit(1 if _FAIL > 0 else 0)
