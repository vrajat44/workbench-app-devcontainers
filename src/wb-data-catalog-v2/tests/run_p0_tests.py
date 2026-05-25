#!/usr/bin/env python3
"""
P0 Test Runner — validates entity classification and cross-table context
for any profiled project/dataset.

Usage:
    python run_p0_tests.py --project PROJECT --bucket BUCKET [--billing-project BP] [--dataset DS]

Reads semantic + technical profiles from GCS and validates:
  - entity_anchor is a real column
  - entity_type is from the allowed set
  - cohort_dimensions excludes IDs and constants
  - join_paths reference real tables
  - new fields survive write/read round-trip
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


ALLOWED_ENTITY_TYPES = {
    "", "subject", "participant", "patient", "encounter", "observation",
    "claim", "specimen", "sample", "provider", "organization", "device",
    "medication_order", "adverse_event", "reference", "metadata",
    "site", "event", "assay",
}


@dataclass
class TestResult:
    test_id: str
    table: str
    description: str
    status: str = "pass"   # pass, fail, skip, warn
    details: str = ""


@dataclass
class TestReport:
    project: str
    bucket: str
    results: list[TestResult] = field(default_factory=list)

    def add(self, r: TestResult):
        self.results.append(r)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skip")

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.status == "warn")

    def print_summary(self):
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"P0 Test Results: {self.project}")
        print(f"{'='*60}")
        print(f"Total: {total} | Pass: {self.passed} | Fail: {self.failed} | "
              f"Warn: {self.warned} | Skip: {self.skipped}")
        print(f"{'='*60}")

        for r in self.results:
            icon = {"pass": "OK", "fail": "FAIL", "skip": "SKIP", "warn": "WARN"}[r.status]
            line = f"  [{icon:4s}] {r.test_id}: {r.description}"
            if r.status in ("fail", "warn") and r.details:
                line += f"\n         {r.details}"
            print(line)

        print()


def load_profiles(bucket: str, project: str, billing_project: str, dataset_filter: Optional[str]):
    """Load all tech + semantic profiles from GCS."""
    sys.path.insert(0, "../backend")
    from verily_profiler.storage import (
        scan_profile_availability, read_tech_profile, read_sem_profile,
    )

    avail = scan_profile_availability(bucket, project, billing_project_id=billing_project)

    profiles: dict[str, dict[str, Any]] = {}
    for fq, info in sorted(avail.items()):
        parts = fq.split(".")
        ds = parts[1] if len(parts) >= 2 else ""
        if dataset_filter and ds != dataset_filter:
            continue
        entry: dict[str, Any] = {"tech": None, "sem": None}
        if info.get("technical"):
            entry["tech"] = read_tech_profile(bucket, fq, project_id=billing_project)
        if info.get("semantic"):
            entry["sem"] = read_sem_profile(bucket, fq, project_id=billing_project)
        if entry["tech"] or entry["sem"]:
            profiles[fq] = entry

    return profiles


def get_tech_columns(tech: Optional[dict]) -> set[str]:
    if not tech:
        return set()
    return {c.get("name", c.get("column_name", "")) for c in tech.get("columns", [])}


def get_tech_column(tech: Optional[dict], col_name: str) -> Optional[dict]:
    if not tech:
        return None
    for c in tech.get("columns", []):
        name = c.get("name", c.get("column_name", ""))
        if name == col_name:
            return c
    return None


def run_tests(profiles: dict[str, dict], all_table_names: set[str], report: TestReport):
    """Run all P0 validation tests across all profiled tables."""

    for fq, data in sorted(profiles.items()):
        tech = data.get("tech")
        sem = data.get("sem")
        short = fq.split(".")[-1] if "." in fq else fq

        if not sem:
            report.add(TestResult(
                test_id="SEM-EXISTS", table=fq,
                description=f"{short}: semantic profile exists",
                status="skip", details="No semantic profile found",
            ))
            continue

        report.add(TestResult(
            test_id="SEM-EXISTS", table=fq,
            description=f"{short}: semantic profile exists",
            status="pass",
        ))

        tech_cols = get_tech_columns(tech)

        # --- entity_anchor is a real column ---
        entity_anchor = sem.get("entity_anchor", "")
        if entity_anchor:
            if tech_cols and entity_anchor not in tech_cols:
                report.add(TestResult(
                    test_id="ENTITY-ANCHOR-VALID", table=fq,
                    description=f"{short}: entity_anchor '{entity_anchor}' is a real column",
                    status="fail",
                    details=f"Column '{entity_anchor}' not found in tech profile. Columns: {sorted(tech_cols)[:10]}",
                ))
            else:
                report.add(TestResult(
                    test_id="ENTITY-ANCHOR-VALID", table=fq,
                    description=f"{short}: entity_anchor '{entity_anchor}' is a real column",
                    status="pass",
                ))
        else:
            report.add(TestResult(
                test_id="ENTITY-ANCHOR-VALID", table=fq,
                description=f"{short}: entity_anchor is empty",
                status="warn", details="No entity_anchor set — may be expected for reference/metadata tables",
            ))

        # --- entity_type is from the allowed set ---
        entity_type = sem.get("entity_type", "")
        if entity_type in ALLOWED_ENTITY_TYPES:
            report.add(TestResult(
                test_id="ENTITY-TYPE-VALID", table=fq,
                description=f"{short}: entity_type '{entity_type}' is valid",
                status="pass",
            ))
        else:
            report.add(TestResult(
                test_id="ENTITY-TYPE-VALID", table=fq,
                description=f"{short}: entity_type '{entity_type}' is not in allowed set",
                status="fail",
                details=f"Got '{entity_type}', expected one of: {sorted(ALLOWED_ENTITY_TYPES - {''})}",
            ))

        # --- cohort_dimensions excludes IDs and constants ---
        cohort_dims = sem.get("cohort_dimensions", [])
        row_count = (tech or {}).get("row_count", 0)
        bad_dims: list[str] = []
        for dim in cohort_dims:
            tc = get_tech_column(tech, dim)
            if not tc:
                continue
            dc = tc.get("distinct_count", 0)
            if dc == 1:
                bad_dims.append(f"{dim} (constant: distinct_count=1)")
            elif row_count > 0 and dc == row_count:
                bad_dims.append(f"{dim} (unique ID: distinct_count={dc}=row_count)")

        if bad_dims:
            report.add(TestResult(
                test_id="COHORT-DIMS-CLEAN", table=fq,
                description=f"{short}: cohort_dimensions excludes IDs and constants",
                status="fail",
                details=f"Bad dimensions: {'; '.join(bad_dims)}",
            ))
        elif cohort_dims:
            report.add(TestResult(
                test_id="COHORT-DIMS-CLEAN", table=fq,
                description=f"{short}: cohort_dimensions ({len(cohort_dims)} cols) excludes IDs and constants",
                status="pass",
            ))
        else:
            report.add(TestResult(
                test_id="COHORT-DIMS-CLEAN", table=fq,
                description=f"{short}: cohort_dimensions is empty",
                status="warn", details="May be expected for reference/metadata tables",
            ))

        # --- cohort_dimensions are real columns ---
        if cohort_dims and tech_cols:
            missing = [d for d in cohort_dims if d not in tech_cols]
            if missing:
                report.add(TestResult(
                    test_id="COHORT-DIMS-EXIST", table=fq,
                    description=f"{short}: cohort_dimensions are real columns",
                    status="fail",
                    details=f"Not found in tech profile: {missing}",
                ))
            else:
                report.add(TestResult(
                    test_id="COHORT-DIMS-EXIST", table=fq,
                    description=f"{short}: all {len(cohort_dims)} cohort_dimensions are real columns",
                    status="pass",
                ))

        # --- join_paths reference real tables ---
        sem_columns = sem.get("columns", [])
        all_join_paths: list[str] = []
        for sc in sem_columns:
            all_join_paths.extend(sc.get("join_paths", []))

        if all_join_paths:
            bad_joins: list[str] = []
            for jp in all_join_paths:
                parts = jp.split(".")
                ref_table = parts[0] if len(parts) >= 2 else jp
                matched = any(ref_table in tn for tn in all_table_names)
                if not matched:
                    bad_joins.append(jp)

            if bad_joins:
                report.add(TestResult(
                    test_id="JOIN-PATHS-VALID", table=fq,
                    description=f"{short}: join_paths reference real tables",
                    status="warn",
                    details=f"Unresolved: {bad_joins[:5]}",
                ))
            else:
                report.add(TestResult(
                    test_id="JOIN-PATHS-VALID", table=fq,
                    description=f"{short}: all {len(all_join_paths)} join_paths reference real tables",
                    status="pass",
                ))
        else:
            report.add(TestResult(
                test_id="JOIN-PATHS-VALID", table=fq,
                description=f"{short}: no join_paths (empty)",
                status="warn", details="No cross-table joins discovered",
            ))

        # --- new fields present in JSON (round-trip) ---
        has_entity = "entity_anchor" in sem and "entity_type" in sem
        has_cohort = "cohort_dimensions" in sem
        if has_entity and has_cohort:
            report.add(TestResult(
                test_id="FIELDS-ROUNDTRIP", table=fq,
                description=f"{short}: entity_anchor, entity_type, cohort_dimensions present in profile JSON",
                status="pass",
            ))
        else:
            missing_fields = []
            if "entity_anchor" not in sem:
                missing_fields.append("entity_anchor")
            if "entity_type" not in sem:
                missing_fields.append("entity_type")
            if "cohort_dimensions" not in sem:
                missing_fields.append("cohort_dimensions")
            report.add(TestResult(
                test_id="FIELDS-ROUNDTRIP", table=fq,
                description=f"{short}: new fields present in profile JSON",
                status="fail",
                details=f"Missing: {missing_fields}",
            ))

        # --- validation status is not "fail" ---
        validation = sem.get("validation", {})
        val_status = validation.get("status", "pass")
        if val_status == "fail":
            issues = validation.get("issues", [])
            report.add(TestResult(
                test_id="VALIDATION-OK", table=fq,
                description=f"{short}: validation status is not 'fail'",
                status="fail",
                details=f"Issues: {issues[:3]}",
            ))
        else:
            report.add(TestResult(
                test_id="VALIDATION-OK", table=fq,
                description=f"{short}: validation status = '{val_status}'",
                status="pass",
            ))


def main():
    parser = argparse.ArgumentParser(description="P0 Test Runner — validate semantic profiles")
    parser.add_argument("--project", required=True, help="Data project ID (e.g., wb-beamish-acorn-6393)")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs:// prefix)")
    parser.add_argument("--billing-project", default=None, help="Billing project (defaults to --project)")
    parser.add_argument("--dataset", default=None, help="Filter to a specific dataset")
    args = parser.parse_args()

    billing = args.billing_project or args.project

    print(f"Loading profiles from gs://{args.bucket} for project {args.project}...")
    if args.dataset:
        print(f"  Filtering to dataset: {args.dataset}")

    profiles = load_profiles(args.bucket, args.project, billing, args.dataset)

    if not profiles:
        print("No profiles found. Nothing to test.")
        sys.exit(0)

    print(f"Loaded {len(profiles)} profiled tables")

    all_table_names = set()
    for fq in profiles:
        all_table_names.add(fq)
        parts = fq.split(".")
        if len(parts) >= 3:
            all_table_names.add(parts[2])  # short name
            all_table_names.add(f"{parts[1]}.{parts[2]}")  # dataset.table

    report = TestReport(project=args.project, bucket=args.bucket)
    run_tests(profiles, all_table_names, report)
    report.print_summary()

    sys.exit(1 if report.failed > 0 else 0)


if __name__ == "__main__":
    main()
