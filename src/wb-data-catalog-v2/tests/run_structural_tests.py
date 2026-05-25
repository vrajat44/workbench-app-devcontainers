#!/usr/bin/env python3
"""
Structural Compliance Tests — validates that semantic profiles are well-formed.

Checks schema structure, allowed values, field presence, and cross-profile
consistency. Does NOT evaluate semantic accuracy (that's Layer 2).

Usage:
    cd backend
    python ../tests/run_structural_tests.py \
        --project wb-shrewd-papaya-8403 \
        --bucket metadata-json-wb-shrewd-papaya-8403
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, "../backend")

from verily_profiler.storage import (
    scan_profile_availability, read_tech_profile, read_sem_profile,
)


# ── Allowed value sets ──────────────────────────────────────────────────────

ALLOWED_PK_TYPES = {"single", "composite", "none"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_SENSITIVITY = {
    "", "P_PNAME", "P_STREETADDR", "P_GEOREGION", "P_POSTALCODE",
    "P_DATE", "P_DOB", "P_DOD", "P_AGE", "P_PHONE", "P_FAX", "P_EMAIL",
    "P_SSN", "P_MRN", "P_HPBN", "P_ACCOUNTNUM", "P_CERT", "P_VEHICLEID",
    "P_DEVICEID", "P_BIOMETRIC", "P_IPADDR", "P_URL", "P_FULLFACEPHOTO",
    "P_RACEETHNICITY", "UID", "FREETEXT",
    "PHI", "PII",  # legacy codes still accepted
}
ALLOWED_MEASUREMENT_METHODS = {
    "self-reported", "lab-measured", "device-collected",
    "calculated", "administrative", "",
}
ALLOWED_LINK_TYPES = {"entity_key", "foreign_key", "shared_dimension", "temporal"}

DOMAIN_TAXONOMY = [
    "Clinical / EHR", "Genomics / Omics", "Claims / Billing", "Demographics",
    "Social Determinants of Health", "Research / Clinical Trials",
    "Administrative / Operations", "Imaging / Radiology",
    "Public Health / Epidemiology", "Geospatial", "Financial",
    "IoT / Wearables / Device", "Pharmacy / Medication", "Laboratory",
    "Survey / Patient-Reported", "General / Other",
]

STANDARD_SYSTEM_URIS = {
    "http://loinc.org", "http://snomed.info/sct",
    "http://hl7.org/fhir/sid/icd-10", "http://hl7.org/fhir/sid/icd-10-cm",
    "http://hl7.org/fhir/sid/ndc", "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://www.ama-assn.org/go/cpt",
    "https://www.cms.gov/Medicare/Coding/HCPCSReleaseCodeSets",
    "urn:verily:custom",
}

ALLOWED_ENTITY_TYPES = {
    "", "subject", "participant", "patient", "encounter", "observation",
    "claim", "specimen", "sample", "provider", "organization", "device",
    "medication_order", "adverse_event", "reference", "metadata",
    "site", "event", "assay",
}


# ── Test infrastructure ──────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    table: str
    description: str
    status: str = "pass"
    details: str = ""


@dataclass
class TestReport:
    project: str
    results: list[TestResult] = field(default_factory=list)

    def add(self, r: TestResult):
        self.results.append(r)

    @property
    def passed(self) -> int: return sum(1 for r in self.results if r.status == "pass")
    @property
    def failed(self) -> int: return sum(1 for r in self.results if r.status == "fail")
    @property
    def warned(self) -> int: return sum(1 for r in self.results if r.status == "warn")

    def print_summary(self):
        total = len(self.results)
        print(f"\n{'='*70}")
        print(f"Structural Compliance: {self.project}")
        print(f"{'='*70}")
        print(f"Total: {total} | Pass: {self.passed} | Fail: {self.failed} | Warn: {self.warned}")
        print(f"{'='*70}")

        for r in self.results:
            icon = {"pass": "OK", "fail": "FAIL", "warn": "WARN"}[r.status]
            line = f"  [{icon:4s}] {r.test_id}: {r.description}"
            if r.status in ("fail", "warn") and r.details:
                line += f"\n         {r.details}"
            print(line)
        print()


# ── Test functions ───────────────────────────────────────────────────────────

def check_required_table_fields(sem: dict, fq: str, short: str, report: TestReport):
    """Check all required table-level fields are present."""
    required = ["table", "profiled_at", "business_name", "table_definition",
                 "primary_key", "granularity", "semantic_domain", "columns", "validation"]
    missing = [f for f in required if f not in sem]
    if missing:
        report.add(TestResult("TBL-FIELDS", fq, f"{short}: required table fields present",
                              "fail", f"Missing: {missing}"))
    else:
        report.add(TestResult("TBL-FIELDS", fq, f"{short}: all required table fields present", "pass"))


def check_domain_taxonomy(sem: dict, fq: str, short: str, report: TestReport):
    """Check semantic_domain.primary is from the fixed taxonomy."""
    domain = sem.get("semantic_domain", {})
    primary = domain.get("primary", "")
    if primary in DOMAIN_TAXONOMY:
        report.add(TestResult("DOMAIN-VALID", fq, f"{short}: domain '{primary}' is valid", "pass"))
    elif not primary:
        report.add(TestResult("DOMAIN-VALID", fq, f"{short}: domain is empty", "warn"))
    else:
        report.add(TestResult("DOMAIN-VALID", fq, f"{short}: domain '{primary}' not in taxonomy",
                              "fail", f"Allowed: {DOMAIN_TAXONOMY}"))


def check_pk_type(sem: dict, fq: str, short: str, report: TestReport):
    """Check primary_key.pk_type is valid."""
    pk = sem.get("primary_key", {})
    pk_type = pk.get("pk_type", pk.get("type", ""))
    if pk_type in ALLOWED_PK_TYPES:
        report.add(TestResult("PK-TYPE", fq, f"{short}: pk_type '{pk_type}' is valid", "pass"))
    else:
        report.add(TestResult("PK-TYPE", fq, f"{short}: pk_type '{pk_type}' invalid",
                              "fail", f"Allowed: {ALLOWED_PK_TYPES}"))


def check_entity_type(sem: dict, fq: str, short: str, report: TestReport):
    """Check entity_type is from allowed set."""
    et = sem.get("entity_type", "")
    if et in ALLOWED_ENTITY_TYPES:
        report.add(TestResult("ENTITY-TYPE", fq, f"{short}: entity_type '{et}' valid", "pass"))
    else:
        report.add(TestResult("ENTITY-TYPE", fq, f"{short}: entity_type '{et}' not in allowed set",
                              "fail"))


def check_columns_match_tech(sem: dict, tech: Optional[dict], fq: str, short: str, report: TestReport):
    """Check all sem column names exist in tech profile."""
    if not tech:
        report.add(TestResult("COL-MATCH", fq, f"{short}: column cross-check", "warn",
                              "No tech profile to compare"))
        return
    tech_names = {c.get("name", c.get("column_name", "")) for c in tech.get("columns", [])}
    sem_names = [c.get("name", "") for c in sem.get("columns", [])]
    unknown = [n for n in sem_names if n and n not in tech_names]
    if unknown:
        report.add(TestResult("COL-MATCH", fq, f"{short}: sem columns match tech schema",
                              "fail", f"Unknown columns: {unknown}"))
    else:
        report.add(TestResult("COL-MATCH", fq,
                              f"{short}: all {len(sem_names)} sem columns match tech schema", "pass"))


def check_column_fields(sem: dict, fq: str, short: str, report: TestReport):
    """Check column-level field values are from allowed sets."""
    issues: list[str] = []
    for col in sem.get("columns", []):
        name = col.get("name", "?")

        sensitivity = col.get("sensitivity", "")
        if sensitivity not in ALLOWED_SENSITIVITY:
            issues.append(f"{name}: sensitivity='{sensitivity}'")

        confidence = col.get("confidence", "")
        if confidence and confidence not in ALLOWED_CONFIDENCE:
            issues.append(f"{name}: confidence='{confidence}'")

        mm = col.get("measurement_method", "")
        if mm and mm not in ALLOWED_MEASUREMENT_METHODS:
            issues.append(f"{name}: measurement_method='{mm}'")

        # concept_binding and code_system_binding mutual exclusion
        has_concept = bool(col.get("concept_binding"))
        has_code_sys = bool(col.get("code_system_binding"))
        if has_concept and has_code_sys:
            issues.append(f"{name}: has both concept_binding AND code_system_binding")

        # terminology_bindings URIs
        for tb in col.get("terminology_bindings", []):
            uri = tb.get("system", "")
            if uri and uri not in STANDARD_SYSTEM_URIS:
                issues.append(f"{name}: unknown terminology URI '{uri}'")

    if issues:
        report.add(TestResult("COL-VALUES", fq, f"{short}: column field values valid",
                              "fail", "; ".join(issues[:5]) + (f" (+{len(issues)-5} more)" if len(issues) > 5 else "")))
    else:
        report.add(TestResult("COL-VALUES", fq, f"{short}: all column field values valid", "pass"))


def check_structural_links(sem: dict, fq: str, short: str, report: TestReport):
    """Check structural_links have valid link_type values."""
    links = sem.get("structural_links", [])
    if not links:
        return
    bad = [l for l in links if l.get("link_type", "") not in ALLOWED_LINK_TYPES]
    if bad:
        types = [l.get("link_type", "") for l in bad]
        report.add(TestResult("LINKS-TYPE", fq, f"{short}: structural_links have valid link_type",
                              "fail", f"Invalid: {types}"))
    else:
        report.add(TestResult("LINKS-TYPE", fq,
                              f"{short}: {len(links)} structural_links all have valid link_type", "pass"))


def check_entity_anchor_is_column(sem: dict, tech: Optional[dict], fq: str, short: str, report: TestReport):
    """Check entity_anchor names a real column."""
    anchor = sem.get("entity_anchor", "")
    if not anchor:
        report.add(TestResult("ANCHOR-COL", fq, f"{short}: entity_anchor empty", "warn"))
        return
    if not tech:
        return
    tech_cols = {c.get("name", c.get("column_name", "")) for c in tech.get("columns", [])}
    if anchor in tech_cols:
        report.add(TestResult("ANCHOR-COL", fq, f"{short}: entity_anchor '{anchor}' is a real column", "pass"))
    else:
        report.add(TestResult("ANCHOR-COL", fq, f"{short}: entity_anchor '{anchor}' not found in schema",
                              "fail", f"Columns: {sorted(tech_cols)[:10]}"))


def check_validation_not_fail(sem: dict, fq: str, short: str, report: TestReport):
    """Check validation.status is not 'fail'."""
    v = sem.get("validation", {})
    st = v.get("status", "pass")
    if st == "fail":
        issues = v.get("issues", [])
        report.add(TestResult("VAL-STATUS", fq, f"{short}: validation status",
                              "fail", f"Issues: {issues[:3]}"))
    else:
        report.add(TestResult("VAL-STATUS", fq, f"{short}: validation status='{st}'", "pass"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structural compliance tests for semantic profiles")
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--billing-project", default=None)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    billing = args.billing_project or args.project

    print(f"Loading profiles from gs://{args.bucket} for {args.project}...")
    avail = scan_profile_availability(args.bucket, args.project, billing_project_id=billing)

    report = TestReport(project=args.project)
    tested = 0

    for fq in sorted(avail):
        info = avail[fq]
        parts = fq.split(".")
        ds = parts[1] if len(parts) >= 2 else ""
        if args.dataset and ds != args.dataset:
            continue
        if not info.get("semantic"):
            continue

        short = parts[-1] if parts else fq
        sem = read_sem_profile(args.bucket, fq, project_id=billing)
        tech = read_tech_profile(args.bucket, fq, project_id=billing) if info.get("technical") else None

        if not sem:
            report.add(TestResult("SEM-LOAD", fq, f"{short}: semantic profile loads", "fail",
                                  "read_sem_profile returned None"))
            continue

        tested += 1
        check_required_table_fields(sem, fq, short, report)
        check_domain_taxonomy(sem, fq, short, report)
        check_pk_type(sem, fq, short, report)
        check_entity_type(sem, fq, short, report)
        check_columns_match_tech(sem, tech, fq, short, report)
        check_column_fields(sem, fq, short, report)
        check_structural_links(sem, fq, short, report)
        check_entity_anchor_is_column(sem, tech, fq, short, report)
        check_validation_not_fail(sem, fq, short, report)

    print(f"Tested {tested} tables with semantic profiles")
    report.print_summary()
    sys.exit(1 if report.failed > 0 else 0)


if __name__ == "__main__":
    main()
