# WB Data Catalog v2 — Profiling Metadata Inventory

**FHIR mapping (tabular L3/L4 + MeasureReport):** see [Metadata field mapping at a glance](../../Metadata/design/metadata_field_mapping.md) in `product_mgmnt/Metadata/design/`.

Canonical profile shapes are defined in the **`verily-profiler`** package under this app. Profiles are written to GCS as `tech_profile.json` and `semantic_profile.json` under `profiling/{project}/{dataset}/{table}/`. The backend API and web UI consume the same JSON field names.

---

## Technical Profiling — pure BigQuery, no LLM

Technical profiling uses one combined `SELECT` over the table for nulls, approximate distincts, string lengths, numeric min/max/stddev, and approximate medians; low-cardinality top-value queries and high-cardinality pattern samples run in parallel afterward.

### Table-level metadata

| Field | Description |
|---|---|
| `table` | Fully-qualified table name (`project.dataset.table`) |
| `row_count` | Total row count |
| `size_bytes` | Table storage size in bytes (from table metadata API if not in discovery) |
| `profiled_at` | ISO 8601 UTC timestamp of profiling run |
| `validation.status` | `pass` or `fail` |
| `validation.anomalies` | Critical issues (e.g. combined query failure, 100% NULL columns) |
| `validation.warnings` | Non-critical flags (near-all-null, single-value columns) |

### Column-level metadata (per column)

| Field | Applies to | Description |
|---|---|---|
| `name` | All | Column name |
| `data_type` | All | BigQuery data type |
| `nullable` | All | Whether the column is nullable (`INFORMATION_SCHEMA`) |
| `null_count` | All | Number of NULL rows |
| `null_percent` | All | Percentage of rows that are NULL |
| `distinct_count` | All | Approximate distinct value count (`APPROX_COUNT_DISTINCT`) |
| `top_values` | Low-cardinality (2–50 distinct values) | Frequent values (BQ query up to 25 rows; JSON keeps up to **15**) |
| `value_counts` | Same as `top_values` | Value-to-count mapping (top **15** in JSON) |
| `string_stats.min_length` | STRING / BYTES | Minimum string length |
| `string_stats.max_length` | STRING / BYTES | Maximum string length |
| `string_stats.avg_length` | STRING / BYTES | Average string length |
| `numeric_stats.min` | INT64, FLOAT64, NUMERIC, BIGNUMERIC, … | Minimum value |
| `numeric_stats.max` | Numeric types | Maximum value |
| `numeric_stats.median` | Numeric types | Approximate median (`APPROX_QUANTILES`, 2 buckets → offset 1) |
| `numeric_stats.stddev` | Numeric types | Standard deviation |
| `pattern` | STRING / BYTES | Detected pattern when ≥80% of sampled values match: `UUID`, `EMAIL`, `URL`, `IP_V4`, `DATE_ISO`, `DATETIME_ISO`, `PHONE_US` |
| `anomalies` | All | Flags: `all_null`, `near_all_null`, `single_value`, `unique_key_candidate`, or `profiling_failed: …` |

---

## Semantic Profiling (v2) — LLM-driven (Vertex / Gemini)

Semantic profiling consumes the technical profile for the same table, optional project terminology registry context, optional **neighbor catalog** markdown (`_catalog_context.md`) for join suggestions, and optional free-text context. An optional **LLM-as-Judge** pass plus a cross-check catch hallucinated column names. Table-level **`structural_links`** are **derived** from column `join_paths` after the main LLM response (not returned as a separate top-level block from that call).

### Table-level metadata

| Field | Description |
|---|---|
| `table` | Fully-qualified table name |
| `profiled_at` | ISO 8601 UTC timestamp |
| `model_used` | Gemini model used (e.g. `gemini-2.5-flash`) |
| `business_name` | Short, human-friendly name for the table |
| `table_definition` | 2–3 sentence plain-language description |
| **`primary_key.columns`** | Column name(s) forming the primary key |
| **`primary_key.pk_type`** | `single`, `composite`, or `none` (model output may use `type`; persisted JSON uses **`pk_type`**) |
| **`primary_key.confidence`** | `high`, `medium`, or `low` |
| **`granularity`** | Plain-English description of what a single record represents |
| **`semantic_domain.primary`** | Primary domain from fixed taxonomy (see below); invalid values coerced to **`General / Other`** |
| **`semantic_domain.sub_domain`** | Free-text sub-domain for specificity |
| **`entity_anchor`** | Column that best identifies the primary entity (e.g. patient / encounter id); empty if unclear |
| **`entity_type`** | Normalized lowercase hint: e.g. `subject`, `patient`, `encounter`, `claim`, `reference`, … |
| **`cohort_dimensions`** | Column names suggested as useful **categorical** cohort filters (excludes IDs, constants, high-cardinality numerics per profiling rules) |
| **`structural_links`** | Typed edges parsed from `join_paths`: `source_column`, `target_table`, `target_column`, **`link_type`** (`entity_key`, `foreign_key`, `shared_dimension`; **`temporal`** reserved for future use), **`cardinality`**, **`confidence`** |
| `validation.status` | `pass`, `warning`, or `fail` |
| `validation.issues` | Real errors (e.g. unknown column names from the LLM, LLM failure) |
| **`validation.warnings`** | Applicability + judge informational strings |

### Column-level metadata (per column)

| Field | Description |
|---|---|
| `name` | Column name (must match technical profile) |
| `definition` | Plain-language description |
| `terminology_bindings` | Array of `{system, code, display}` — standard URIs or `urn:verily:custom` with stable slug `code` |
| **`concept_binding`** | Optional `{system, code, display, confidence}` — column **is** one fixed concept (e.g. a specific LOINC observation) |
| **`code_system_binding`** | Optional `{system, display, confidence}` — column **contains** codes from a system (e.g. CPT column) |
| `sensitivity` | `PHI`, `PII`, `UID`, or empty |
| `join_paths` | Suggested relationships as `dataset.table.column` or `project.dataset.table.column` strings (should use real names from catalog context when provided) |
| `confidence` | `high` / `medium` / `low` |
| **`unit_of_measure`** | Measurement unit when applicable; empty otherwise |
| **`measurement_method`** | When applicable: `self-reported`, `lab-measured`, `device-collected`, `calculated`, `administrative` |
| **`value_set_binding`** | Optional list of allowed categorical values (aligned with cohort / low-cardinality semantics in profiling instructions) |

**Mutual exclusion:** A column should have **`concept_binding`** or **`code_system_binding`**, not both.

### Semantic Domain Taxonomy

The model must pick exactly one `primary` value from this fixed list:

| Domain | Example sub-domains |
|---|---|
| Clinical / EHR | Oncology, Cardiology, Emergency Medicine |
| Genomics / Omics | Whole Genome Sequencing, Proteomics |
| Claims / Billing | Medicare Claims, Prior Authorization |
| Demographics | Patient Demographics, Census Data |
| Social Determinants of Health | Housing, Food Security, Transportation |
| Research / Clinical Trials | Phase III Trial Data, Cohort Studies |
| Administrative / Operations | Scheduling, Staff Management |
| Imaging / Radiology | DICOM Metadata, Pathology Slides |
| Public Health / Epidemiology | Disease Surveillance, Mortality Statistics |
| Geospatial | County Health Rankings, Facility Locations |
| Financial | Revenue Cycle, Cost Accounting |
| IoT / Wearables / Device | Fitbit Data, Continuous Glucose Monitoring |
| Pharmacy / Medication | Prescription Data, Drug Interactions |
| Laboratory | Blood Chemistry Panels, Microbiology Cultures |
| Survey / Patient-Reported | PHQ-9, Patient Satisfaction Scores |
| General / Other | Fallback when no specific domain applies |

### Standard terminology systems

Supported binding URIs include LOINC, SNOMED CT, ICD-10, ICD-10-CM, NDC, **RxNorm**, **CPT**, **HCPCS**, and **`urn:verily:custom`**. Profiling instructions encourage standard URIs from this set for bindings.

### Validation layer (v2)

1. **Cross-check** — any LLM-referenced column not in the technical schema → **`validation.issues`** → status **`fail`**.
2. **Applicability warnings** (always merged when there are no cross-check failures; also merged with judge warnings when the judge runs), including cases such as:
   - No primary key / `pk_type` is `none`
   - Domain is **`General / Other`**
   - Empty **`granularity`**
   - Numeric technical columns missing **`unit_of_measure`**
3. **LLM-as-Judge** (optional) — second Gemini call reviews definitions, bindings, sensitivity, join paths, confidence, UoM, PK, granularity, domain, **`entity_anchor`**, **`entity_type`**, **`cohort_dimensions`**. Judge failure is non-blocking (surfaces as a warning). Final status: **`fail`** if any issues; else **`warning`** if any warnings; else **`pass`**.

Warnings are intended as **amber** UX signals vs red errors. Semantic profiling requires an existing technical profile; the HTTP API returns **409 Conflict** if semantic is requested first.

---

## Project-level artifacts (not per-table profile JSON)

| Artifact | Role |
|---|---|
| **`_terminology_registry.json`** | Project-wide registry: aggregated terminology rows (`system`, `code`, `display`, `concept_key`, `source_columns`, timestamps). Injected into semantic prompts for reuse. |
| **`_catalog_context.md`** | Markdown summary of profiled tables for **join_paths** / neighbor context; regenerated when profiles change. |
| **Reconciliation** | Post-batch LLM pass can propose canonical codes for duplicate concepts; an apply step may rewrite semantic profiles and the registry (used from bulk and on-demand flows when configured). |

---

## Combined export

Combined export JSON nests **`technical`** and **`semantic`** objects under a top-level **`table`** key when both layers are merged for a single table.
