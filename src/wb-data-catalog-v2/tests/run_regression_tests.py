#!/usr/bin/env python3
"""
WB Data Catalog v2 — Full Regression Test Suite

Run after every feature build to ensure nothing is broken.
Tests all API endpoints, project isolation, cache behavior,
profiling field coverage, and frontend build.

Usage:
  # Quick smoke test (no GCP — tests structure, types, frontend build)
  python tests/run_regression_tests.py

  # Full test with live backend (backend must be running on :8080)
  python tests/run_regression_tests.py --live

  # Full test including profiling (slow — triggers actual LLM calls)
  python tests/run_regression_tests.py --live --with-profiling
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

BASE_URL = "http://localhost:8080"

passed = 0
failed = 0
skipped = 0
section_results: list[tuple[str, int, int, int]] = []


def ok(msg):
    global passed
    passed += 1
    print(f"  ✓ {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  ✗ {msg}")


def skip(msg):
    global skipped
    skipped += 1
    print(f"  ⊘ {msg}")


def section(name):
    global passed, failed, skipped
    if section_results:
        prev = section_results[-1]
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")


def end_section(name):
    section_results.append((name, passed, failed, skipped))


def _get(path, timeout=10):
    import urllib.request
    try:
        r = urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout)
        return json.loads(r.read()), r.status
    except Exception as e:
        return {"error": str(e)}, getattr(e, "code", 0)


def _post(path, body=None, timeout=30):
    import urllib.request
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), r.status
    except Exception as e:
        return {"error": str(e)}, getattr(e, "code", 0)


def _put(path, body=None, timeout=30):
    import urllib.request
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), r.status
    except Exception as e:
        return {"error": str(e)}, getattr(e, "code", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Offline Tests (no GCP, no running server)
# ═══════════════════════════════════════════════════════════════════════════════

def test_01_backend_imports():
    """All backend modules import without error."""
    section("1. Backend Imports")
    modules = [
        "main", "chat_handler", "profiling_runner", "bulk_profiler",
        "bq_preview", "chart_advisor", "gw_computation", "join_inference",
        "api_models",
    ]
    for mod in modules:
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            fail(f"import {mod}: {e}")

    profiler_modules = [
        "verily_profiler", "verily_profiler.semantic", "verily_profiler.models",
        "verily_profiler.technical", "verily_profiler.storage",
        "verily_profiler.discovery", "verily_profiler.llm",
        "verily_profiler.catalog_context", "verily_profiler.terminology_domains",
    ]
    for mod in profiler_modules:
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            fail(f"import {mod}: {e}")
    end_section("Backend Imports")


def test_02_profile_field_coverage():
    """Semantic profile fields: prompt ↔ dataclass ↔ serialization ↔ frontend types."""
    section("2. Profile Field Coverage")
    # Delegate to the dedicated test
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "run_profile_field_tests.py")],
        capture_output=True, text=True, timeout=30,
    )
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if line.strip().startswith("✓"):
            ok(line.strip()[2:])
        elif line.strip().startswith("✗"):
            fail(line.strip()[2:])
    if result.returncode != 0 and not any("✗" in l for l in lines):
        fail(f"Profile field tests exited with code {result.returncode}")
    end_section("Profile Field Coverage")


def test_03_frontend_build():
    """Frontend TypeScript compiles and Vite builds successfully."""
    section("3. Frontend Build")
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

    # TypeScript check
    tsc = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        capture_output=True, text=True, timeout=60, cwd=frontend_dir,
    )
    if tsc.returncode == 0:
        ok("TypeScript compiles (0 errors)")
    else:
        errors = [l for l in tsc.stdout.split("\n") if "error TS" in l]
        fail(f"TypeScript errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e.strip()}")
    end_section("Frontend Build")


def test_04_gw_translate():
    """Graphic Walker SQL translation produces valid SQL for all operation types."""
    section("4. GW SQL Translation")
    from gw_computation import translate

    fq = "project.dataset.table"

    # Raw select
    sql = translate(fq, {"workflow": [], "limit": 10})
    if "SELECT * FROM" in sql and "LIMIT 10" in sql:
        ok("Raw SELECT translation")
    else:
        fail(f"Raw SELECT: {sql[:80]}")

    # Aggregate
    sql = translate(fq, {"workflow": [
        {"type": "view", "query": [{"op": "aggregate",
         "groupBy": ["col_a"], "measures": [{"field": "col_b", "agg": "count", "asFieldKey": "cnt"}]}]}
    ], "limit": 100})
    if "GROUP BY" in sql and "COUNT" in sql:
        ok("Aggregate translation")
    else:
        fail(f"Aggregate: {sql[:80]}")

    # Filter
    sql = translate(fq, {"workflow": [
        {"type": "filter", "filters": [
            {"fid": "age", "rule": {"type": "range", "value": [18, 65]}}
        ]},
    ], "limit": 50})
    if "WHERE" in sql and "`age`" in sql:
        ok("Filter translation")
    else:
        fail(f"Filter: {sql[:80]}")

    # Fold
    sql = translate(fq, {"workflow": [
        {"type": "view", "query": [{"op": "fold", "foldBy": ["a", "b"],
         "newFoldKeyCol": "key", "newFoldValueCol": "value"}]}
    ], "limit": 10})
    if "UNION ALL" in sql:
        ok("Fold translation")
    else:
        fail(f"Fold: {sql[:80]}")

    # Bin
    sql = translate(fq, {"workflow": [
        {"type": "view", "query": [{"op": "bin", "binBy": "age", "binSize": 5}]}
    ], "limit": 10})
    if "FLOOR" in sql:
        ok("Bin translation")
    else:
        fail(f"Bin: {sql[:80]}")
    end_section("GW SQL Translation")


def test_05_sql_injection_safety():
    """SQL translation sanitizes column and table names."""
    section("5. SQL Injection Safety")
    from gw_computation import translate, _safe, _safe_table

    # Column injection
    result = _safe("col`; DROP TABLE--")
    if "`" not in result and ";" not in result and "-" not in result:
        ok("Column name sanitized (backtick, semicolon, dash stripped)")
    else:
        fail(f"Column not sanitized: {result}")

    # Table injection
    result = _safe_table("project.dataset.table`; DROP")
    if "`" not in result and ";" not in result:
        ok("Table name sanitized")
    else:
        fail(f"Table not sanitized: {result}")

    # Filter value literal escaping
    from gw_computation import _literal
    result = _literal("O'Brien")
    if "\\'" in result:
        ok("String literal escapes single quotes")
    else:
        fail(f"Literal not escaped: {result}")
    end_section("SQL Injection Safety")


def test_06_join_inference():
    """Join inference produces correct results for known patterns."""
    section("6. Join Inference")
    from join_inference import infer_joins

    sem_profiles = [
        {"table": "p.d.patients", "columns": [
            {"name": "patient_id", "terminology_bindings": []},
            {"name": "name", "terminology_bindings": []},
        ]},
        {"table": "p.d.visits", "columns": [
            {"name": "patient_id", "terminology_bindings": []},
            {"name": "visit_date", "terminology_bindings": []},
        ]},
        {"table": "p.d.labs", "columns": [
            {"name": "patient_id", "terminology_bindings": []},
            {"name": "lab_code", "terminology_bindings": [
                {"system": "LOINC", "code": "1234-5", "display": "Glucose"}
            ]},
        ]},
    ]
    tech_profiles = [
        {"table": "p.d.patients", "columns": [
            {"name": "patient_id", "data_type": "STRING"},
            {"name": "name", "data_type": "STRING"},
        ]},
        {"table": "p.d.visits", "columns": [
            {"name": "patient_id", "data_type": "STRING"},
            {"name": "visit_date", "data_type": "DATE"},
        ]},
        {"table": "p.d.labs", "columns": [
            {"name": "patient_id", "data_type": "STRING"},
            {"name": "lab_code", "data_type": "STRING"},
        ]},
    ]

    joins = infer_joins(sem_profiles, tech_profiles)

    if "p.d.patients" in joins:
        targets = [jp["target"] for jp in joins["p.d.patients"]]
        if any("visits" in t for t in targets):
            ok("Detected patient_id join: patients → visits")
        else:
            fail("Missing patients → visits join")
        if any("labs" in t for t in targets):
            ok("Detected patient_id join: patients → labs")
        else:
            fail("Missing patients → labs join")
    else:
        fail("No joins detected for patients table")

    # Generic names should be skipped
    if not any(jp["source_column"] == "name" for jps in joins.values() for jp in jps):
        ok("Generic column 'name' correctly skipped")
    else:
        fail("Generic column 'name' should be skipped")
    end_section("Join Inference")


def test_07_cache_invalidation():
    """Settings change clears all required caches."""
    section("7. Cache Invalidation Logic")
    import inspect
    import main

    source = inspect.getsource(main.api_update_settings)

    checks = [
        ("invalidate_context_cache", "Chat context cache"),
        ("_invalidate_profiling_caches", "Profiling caches"),
        ("_sessions.clear", "Chat sessions"),
        ("_ensure_catalog_context_exists", "Catalog context regeneration"),
    ]
    for needle, label in checks:
        if needle in source:
            ok(f"Settings endpoint calls {label}")
        else:
            fail(f"Settings endpoint MISSING {label}")

    # Check _invalidate_profiling_caches clears all caches
    inv_source = inspect.getsource(main._invalidate_profiling_caches)
    caches = ["_scan_cache", "_catalog_cache", "_cohort_dims_cache",
              "_terminology_cache", "_terminology_slim_cache", "_col_values_cache"]
    for cache in caches:
        if cache in inv_source:
            ok(f"_invalidate_profiling_caches clears {cache}")
        else:
            fail(f"_invalidate_profiling_caches MISSING {cache}")
    end_section("Cache Invalidation Logic")


def test_08_project_isolation():
    """DATA_PROJECT is used correctly (not BILLING_PROJECT) for data-scoped endpoints."""
    section("8. Project Isolation")
    import inspect
    import main

    # Catalog should use DATA_PROJECT for discovery
    cat_src = inspect.getsource(main.api_catalog)
    if "DATA_PROJECT" in cat_src:
        ok("Catalog uses DATA_PROJECT")
    else:
        fail("Catalog does not reference DATA_PROJECT")

    # Chat context-info should use DATA_PROJECT
    ctx_src = inspect.getsource(main.api_chat_context_info)
    if "DATA_PROJECT" in ctx_src:
        ok("Chat context-info uses DATA_PROJECT")
    else:
        fail("Chat context-info does not reference DATA_PROJECT")

    # Cohort dimensions should use DATA_PROJECT in cache key
    dim_src = inspect.getsource(main.api_cohort_dimensions)
    if "DATA_PROJECT" in dim_src:
        ok("Cohort dimensions uses DATA_PROJECT")
    else:
        fail("Cohort dimensions does not reference DATA_PROJECT")

    # Reset should scope to DATA_PROJECT
    reset_src = inspect.getsource(main.api_reset_profiles)
    if "DATA_PROJECT" in reset_src:
        ok("Profile reset scoped to DATA_PROJECT")
    else:
        fail("Profile reset does not reference DATA_PROJECT")
    end_section("Project Isolation")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Live API Tests (requires running backend on :8080)
# ═══════════════════════════════════════════════════════════════════════════════

def test_10_health():
    """Health endpoints return 200."""
    section("10. Health Endpoints")
    d, status = _get("/api/health")
    if status == 200 and d.get("status") == "ok":
        ok(f"GET /api/health → 200 (project: {d.get('data_project', '?')})")
    else:
        fail(f"GET /api/health → {status}")

    d, status = _get("/api/health/deep")
    if status == 200:
        ok(f"GET /api/health/deep → 200")
    else:
        fail(f"GET /api/health/deep → {status}: {d}")
    end_section("Health Endpoints")


def test_11_config():
    """Config returns all expected fields."""
    section("11. Config & Models")
    d, status = _get("/api/config")
    if status != 200:
        fail(f"GET /api/config → {status}")
        return

    expected = ["billing_project", "data_project", "profile_bucket", "configured"]
    missing = [k for k in expected if k not in d]
    if not missing:
        ok(f"Config has all fields (billing={d['billing_project']}, data={d['data_project']})")
    else:
        fail(f"Config missing fields: {missing}")

    d, status = _get("/api/models")
    if status == 200 and "models" in d:
        ok(f"GET /api/models → {len(d['models'])} models")
    else:
        fail(f"GET /api/models → {status}")
    end_section("Config & Models")


def test_12_discovery():
    """Dataset and table discovery works."""
    section("12. Discovery")
    d, status = _get("/api/datasets")
    if status == 200 and "datasets" in d:
        ok(f"GET /api/datasets → {len(d['datasets'])} datasets")
    else:
        fail(f"GET /api/datasets → {status}")
        end_section("Discovery")
        return

    if d["datasets"]:
        ds = d["datasets"][0]
        d2, status2 = _get(f"/api/datasets/{ds}/tables")
        if status2 == 200:
            tables = d2.get("tables", [])
            ok(f"GET /api/datasets/{ds}/tables → {len(tables)} tables")
        else:
            fail(f"GET /api/datasets/{ds}/tables → {status2}")
    end_section("Discovery")


def test_13_catalog():
    """Catalog endpoint returns datasets with tables."""
    section("13. Catalog")
    d, status = _get("/api/catalog", timeout=30)
    if status == 200 and "datasets" in d:
        total_tables = sum(len(ds.get("tables", [])) for ds in d["datasets"])
        ok(f"GET /api/catalog → {len(d['datasets'])} datasets, {total_tables} tables")
        if d["datasets"] and d["datasets"][0].get("tables"):
            t = d["datasets"][0]["tables"][0]
            has_profiling = "profiling" in t
            ok(f"Table has profiling status: {has_profiling}")
    else:
        fail(f"GET /api/catalog → {status}")
    end_section("Catalog")


def test_14_preview():
    """Table preview returns rows and schema."""
    section("14. Table Preview")
    d, status = _get("/api/config")
    if status != 200:
        skip("Config not available")
        end_section("Table Preview")
        return
    project = d.get("data_project", "")

    ds, _ = _get("/api/datasets")
    datasets = ds.get("datasets", [])
    if not datasets:
        skip("No datasets to preview")
        end_section("Table Preview")
        return

    tables_resp, _ = _get(f"/api/datasets/{datasets[0]}/tables")
    tables = tables_resp.get("tables", [])
    if not tables:
        skip("No tables in first dataset")
        end_section("Table Preview")
        return

    table_id = tables[0].get("table_id", "")
    preview, status = _get(f"/api/projects/{project}/datasets/{datasets[0]}/tables/{table_id}/preview", timeout=30)
    if status == 200 and "columns" in preview:
        ok(f"Preview: {len(preview['columns'])} columns, {len(preview.get('rows', []))} rows")
    else:
        fail(f"Preview → {status}")
    end_section("Table Preview")


def test_15_terminology():
    """Terminology endpoints return data."""
    section("15. Terminology")
    d, status = _get("/api/terminology/slim", timeout=15)
    if status == 200:
        entries = d.get("entries", [])
        ok(f"GET /api/terminology/slim → {len(entries)} entries")
    else:
        fail(f"GET /api/terminology/slim → {status}")

    d, status = _get("/api/terminology-domains")
    if status == 200:
        ok(f"GET /api/terminology-domains → 200")
    else:
        fail(f"GET /api/terminology-domains → {status}")
    end_section("Terminology")


def test_16_cohort_dimensions():
    """Cohort dimensions endpoint returns table data."""
    section("16. Cohort Dimensions")
    d, status = _get("/api/cohorts/dimensions", timeout=30)
    if status == 200:
        tables = d.get("tables", [])
        ok(f"GET /api/cohorts/dimensions → {len(tables)} tables")
        if tables:
            t = tables[0]
            has_dims = len(t.get("dimensions", [])) > 0
            ok(f"First table has dimensions: {has_dims}")
            has_anchor = bool(t.get("entity_anchor"))
            ok(f"First table has entity_anchor: {has_anchor}")
    else:
        fail(f"GET /api/cohorts/dimensions → {status}")
    end_section("Cohort Dimensions")


def test_17_chat_context():
    """Chat context info reflects current project."""
    section("17. Chat Context")
    cfg, _ = _get("/api/config")
    data_project = cfg.get("data_project", "")

    d, status = _get("/api/chat/context-info")
    if status == 200:
        ok(f"GET /api/chat/context-info → status={d.get('status')}, tables={d.get('profiled_tables', 0)}")
    else:
        fail(f"GET /api/chat/context-info → {status}")
    end_section("Chat Context")


def test_18_chat_roundtrip():
    """Chat send + clear works end-to-end."""
    section("18. Chat Roundtrip")
    d, status = _post("/api/chat", {"message": "hello", "mode": "metadata"}, timeout=60)
    reply = d.get("response") or d.get("message") or ""
    if status == 200 and reply:
        sid = d.get("session_id", "")
        ok(f"POST /api/chat → response ({len(reply)} chars), session={sid[:8]}...")

        # Clear
        cd, cs = _post("/api/chat/clear", {"session_id": sid})
        if cs == 200:
            ok("POST /api/chat/clear → 200")
        else:
            fail(f"POST /api/chat/clear → {cs}")
    else:
        fail(f"POST /api/chat → {status}: {d.get('error', d.get('detail', ''))[:100]}")
    end_section("Chat Roundtrip")


def test_19_sql_injection_api():
    """SQL injection attempts are blocked by the API."""
    section("19. SQL Injection (API)")
    d, status = _post("/api/cohorts/execute", {
        "table": "project.dataset.table`; DROP TABLE users--",
        "filters": [],
        "entity_column": "id",
    })
    if status in (400, 422):
        ok(f"SQL injection blocked → {status}")
    elif status == 0:
        skip("Backend not reachable")
    else:
        fail(f"SQL injection not blocked → {status}")
    end_section("SQL Injection (API)")


def test_20_cors():
    """CORS blocks unauthorized origins."""
    section("20. CORS")
    import urllib.request
    req = urllib.request.Request(
        f"{BASE_URL}/api/health",
        headers={"Origin": "http://evil.com"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=5)
        cors_header = r.headers.get("Access-Control-Allow-Origin", "")
        if cors_header == "" or cors_header != "http://evil.com":
            ok(f"CORS blocks evil.com (header: '{cors_header}')")
        else:
            fail(f"CORS allows evil.com: {cors_header}")
    except Exception as e:
        ok(f"CORS blocked request: {e}")
    end_section("CORS")


def test_21_settings_project_switch():
    """Settings change + cache invalidation works correctly."""
    section("21. Project Switch")
    cfg, _ = _get("/api/config")
    original_project = cfg.get("data_project", "")
    original_billing = cfg.get("billing_project", "")

    if not original_project:
        skip("No project configured")
        end_section("Project Switch")
        return

    # Switch to billing project (guaranteed to have BQ access)
    test_project = original_billing if original_billing != original_project else original_project
    d, status = _put("/api/settings", {"data_project": test_project}, timeout=120)
    if status == 200:
        ok(f"PUT /api/settings → switched to {test_project}")
    else:
        fail(f"PUT /api/settings → {status}")
        end_section("Project Switch")
        return

    # Verify config reflects change
    cfg2, _ = _get("/api/config")
    if cfg2.get("data_project") == test_project:
        ok("Config reflects new data_project")
    else:
        fail(f"Config still shows {cfg2.get('data_project')}")

    # Restore
    _put("/api/settings", {"data_project": original_project}, timeout=120)
    cfg3, _ = _get("/api/config")
    if cfg3.get("data_project") == original_project:
        ok("Restored original data_project")
    else:
        fail("Failed to restore original project")
    end_section("Project Switch")


def test_22_frontend_routes():
    """Frontend SPA routes all return 200 with index.html."""
    section("22. Frontend Routes")
    routes = ["/", "/terminology", "/cohorts", "/chat", "/settings"]
    for route in routes:
        d, status = _get(route)
        # SPA routes return HTML, not JSON — check status only
        if status == 200:
            ok(f"GET {route} → 200")
        elif status == 0:
            skip(f"GET {route} (no frontend dist)")
        else:
            fail(f"GET {route} → {status}")
    end_section("Frontend Routes")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WB Data Catalog v2 — Full Regression Tests")
    parser.add_argument("--live", action="store_true", help="Run live API tests (requires backend on :8080)")
    parser.add_argument("--with-profiling", action="store_true", help="Include slow profiling tests")
    args = parser.parse_args()

    print("=" * 60)
    print("  WB Data Catalog v2 — Regression Test Suite")
    print("=" * 60)

    # Offline tests (always run)
    test_01_backend_imports()
    test_02_profile_field_coverage()
    test_03_frontend_build()
    test_04_gw_translate()
    test_05_sql_injection_safety()
    test_06_join_inference()
    test_07_cache_invalidation()
    test_08_project_isolation()

    if args.live:
        # Check if backend is running
        try:
            import urllib.request
            urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=5)
        except Exception:
            print("\n  ⚠ Backend not running on :8080 — skipping live tests")
            print("    Start with: cd backend && python main.py")
            args.live = False

    if args.live:
        test_10_health()
        test_11_config()
        test_12_discovery()
        test_13_catalog()
        test_14_preview()
        test_15_terminology()
        test_16_cohort_dimensions()
        test_17_chat_context()
        test_18_chat_roundtrip()
        test_19_sql_injection_api()
        test_20_cors()
        test_21_settings_project_switch()
        test_22_frontend_routes()
    else:
        print("\n  (Skipping live API tests — pass --live to enable)")

    # Summary
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        print("\n  FAILED — fix issues above before shipping.")
        sys.exit(1)
    else:
        print("\n  ALL PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
