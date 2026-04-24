# WB Data Catalog v2 — Profiling Metadata Inventory

## Technical Profiling — pure BigQuery, no LLM

### Table-level metadata

| Field | Description |
|---|---|
| `table` | Fully-qualified table name (`project.dataset.table`) |
| `row_count` | Total row count |
| `size_bytes` | Table storage size in bytes |
| `profiled_at` | ISO 8601 UTC timestamp of profiling run |
| `validation.status` | `pass` or `fail` |
| `validation.anomalies` | Critical issues (e.g. 100% NULL columns) |
| `validation.warnings` | Non-critical flags (near-all-null, single-value columns) |

### Column-level metadata (per column)

| Field | Applies to | Description |
|---|---|---|
| `name` | All | Column name |
| `data_type` | All | BigQuery data type |
| `nullable` | All | Whether the column is nullable |
| `null_count` | All | Number of NULL rows |
| `null_percent` | All | Percentage of rows that are NULL |
| `distinct_count` | All | Approximate distinct value count |
| `top_values` | Low-cardinality (≤50 distinct) | Up to 15 most frequent values |
| `value_counts` | Low-cardinality (≤50 distinct) | Value-to-count mapping (top 15) |
| `string_stats.min_length` | STRING / BYTES | Minimum string length |
| `string_stats.max_length` | STRING / BYTES | Maximum string length |
| `string_stats.avg_length` | STRING / BYTES | Average string length |
| `numeric_stats.min` | Numeric types | Minimum value |
| `numeric_stats.max` | Numeric types | Maximum value |
| `numeric_stats.median` | Numeric types | Approximate median |
| `numeric_stats.stddev` | Numeric types | Standard deviation |
| `pattern` | STRING / BYTES | Detected pattern: `UUID`, `EMAIL`, `URL`, `IP_V4`, `DATE_ISO`, `DATETIME_ISO`, `PHONE_US` |
| `anomalies` | All | Flags: `all_null`, `near_all_null`, `single_value`, `unique_key_candidate`, `profiling_failed` |

---

## Semantic Profiling (v2) — LLM-driven (Gemini)

### Table-level metadata

| Field | Description |
|---|---|
| `table` | Fully-qualified table name |
| `profiled_at` | ISO 8601 UTC timestamp |
| `model_used` | Gemini model used (e.g. `gemini-2.5-flash`) |
| `business_name` | Short, human-friendly name for the table |
| `table_definition` | 2–3 sentence plain-language description |
| **`primary_key.columns`** | Column name(s) forming the primary key |
| **`primary_key.pk_type`** | `single`, `composite`, or `none` |
| **`primary_key.confidence`** | `high`, `medium`, or `low` |
| **`granularity`** | Plain-English description of what a single record represents (e.g. "One observation per patient per day") |
| **`semantic_domain.primary`** | Primary domain from fixed taxonomy (see below) |
| **`semantic_domain.sub_domain`** | Free-text sub-domain for specificity |
| `validation.status` | `pass`, `warning`, or `fail` |
| `validation.issues` | Real errors found in the metadata |
| **`validation.warnings`** | Non-applicable or generic fields (informational, not errors) |

### Column-level metadata (per column)

| Field | Description |
|---|---|
| `name` | Column name (must match technical profile) |
| `definition` | Plain-language description (2–3 sentences) |
| `terminology_bindings` | Array of `{system, code, display}` — LOINC, ICD-10, SNOMED CT, NDC |
| `sensitivity` | `PHI`, `PII`, `UID`, or empty |
| `join_paths` | Suggested foreign key relationships in `table.column` format |
| `confidence` | `high` / `medium` / `low` |
| **`unit_of_measure`** | Measurement unit for applicable columns (e.g. `mg/dL`, `kg`, `years`, `USD`). Empty if not applicable. |

### Semantic Domain Taxonomy

The LLM assigns exactly one primary domain from this fixed set:

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

### Validation layer (v2)

The semantic profiler runs validation with three tiers:

1. **Cross-check against technical profile** — flags any LLM-referenced column names that don't exist.
2. **LLM-as-Judge** — a second Gemini call reviews all metadata including the new v2 fields (PK, granularity, domain, UoM).
3. **Applicability warnings** — automatically generated when:
   - No primary key could be identified
   - Generic domain ("General / Other") was assigned
   - Granularity could not be determined
   - Unit of measure missing for numeric columns

Warnings are surfaced as **amber indicators** in the UI, distinct from red error indicators. A table with only warnings (no issues) gets `status: "warning"` rather than `"fail"`.
