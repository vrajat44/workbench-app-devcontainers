# P0 Test Cases: Entity Classification + Cross-Table Context

**Project**: `wb-beamish-acorn-6393`
**Dataset**: `analysis` (16 fully profiled tables)
**Key observation**: All tables share `SUBJID` and/or `USUBJID` as subject identifiers. Many share `VISIT`/`VISITNUM`. Currently, zero `join_paths` are generated and no entity metadata exists.

---

## Feature A: Entity Anchor + Entity Type + Cohort Dimensions

These test cases validate that re-running semantic profiling produces the three new fields correctly.

### TC-01: Enrollment table identifies subject entity

**Table**: `analysis.ENRDT` (2 columns: SUBJID, enrollment_date)
**Expected**:
- `entity_anchor` = `"SUBJID"`
- `entity_type` = `"subject"` or `"participant"`
- `cohort_dimensions` = `["enrollment_date"]`

**Rationale**: ENRDT is the simplest table — one ID column, one date. The profiler must recognize SUBJID as the entity anchor and enrollment_date as a temporal cohort filter.

---

### TC-02: Diagnoses table identifies subject entity with clinical cohort dimensions

**Table**: `analysis.DIAGNOSES` (128 columns, 2,502 rows)
**Expected**:
- `entity_anchor` = `"SUBJID"` or `"USUBJID"`
- `entity_type` = `"subject"` or `"participant"`
- `cohort_dimensions` includes at least 5 of: `sex`, `smoking_status`, `der_hx_cvd`, `der_hx_diabetes`, `der_hx_cad`, `der_hx_hypertension`, or similar diagnosis flag columns
- `cohort_dimensions` does NOT include: `STUDYID` (single value), `SUBJID` (unique ID), `USUBJID` (unique ID)

**Rationale**: DIAGNOSES has many binary/categorical clinical columns ideal for cohort filtering. The profiler must distinguish filterable dimensions from IDs and constants.

---

### TC-03: Score table identifies composite entity with score-based dimensions

**Table**: `analysis.PHQ9A_SCORES` (9 columns, 14,291 rows, PK: USUBJID+VISIT)
**Expected**:
- `entity_anchor` = `"USUBJID"`
- `entity_type` = `"subject"` or `"observation"`
- `cohort_dimensions` includes at least: `VISIT`, `score`, `score_flag` (or similar score/category columns)
- `cohort_dimensions` does NOT include: `STUDYID`, `USUBJID`

**Rationale**: PHQ-9 is a depression screening score. Each row is one subject at one visit. The score and visit are natural cohort filters ("subjects with PHQ-9 > 10 at baseline").

---

### TC-04: Genomics/lab table identifies specimen-level entity

**Table**: `analysis.MTQC` (11 columns, 2,034 rows, PK: idat_name)
**Expected**:
- `entity_anchor` = `"idat_name"` or `"SUBJID"`
- `entity_type` = `"specimen"` or `"sample"` or `"assay"`
- `cohort_dimensions` includes at least: `array_type`, `visit`
- `cohort_dimensions` does NOT include: `idat_name` (unique ID), `mean_p_value` (continuous metric)

**Rationale**: MTQC is a DNA methylation QC table. Each row is one IDAT file (a specimen/array), not a patient. The profiler should recognize this is specimen-level, not patient-level. However, it still links to subjects via SUBJID.

---

### TC-05: Cohort dimensions exclude inappropriate columns

**Table**: `analysis.ASCVD` (10 columns, 10,389 rows)
**Expected**:
- `cohort_dimensions` MUST NOT include:
  - `STUDYID` — single constant value across all rows (distinct_count=1)
  - `USUBJID` — unique per-subject identifier (high cardinality)
  - `study_day` — continuous numeric, not a useful filter category
- `cohort_dimensions` SHOULD include:
  - `VISIT` — categorical study visit identifier
  - `score_flag` — if it exists as a binary/categorical flag
  - `heart_attack_or_stroke` — binary clinical outcome

**Rationale**: Validates that the profiler uses cardinality and type information to exclude IDs, constants, and continuous numerics from cohort dimensions while including categorical and binary clinical columns.

---

## Feature B: Cross-Table Context Injection

These test cases validate that injecting already-profiled table summaries into the profiling prompt improves join_path accuracy.

### TC-06: Score table discovers join to enrollment via SUBJID

**Table**: `analysis.PHQ9A_SCORES` (re-profiled WITH cross-table context)
**Context injected**: Summary of `ENRDT` (columns: SUBJID, enrollment_date; PK: SUBJID+enrollment_date)
**Expected**:
- Column `SUBJID` or `USUBJID` has `join_paths` containing `"ENRDT.SUBJID"`
- Column `VISIT` may suggest joins to other score tables

**Baseline (current, WITHOUT context)**: All `join_paths` are `[]` (empty).

**Rationale**: The profiler currently sees PHQ9A_SCORES in isolation and produces no join paths. With ENRDT's schema visible, the LLM should recognize that SUBJID is shared.

---

### TC-07: Diagnoses table discovers joins to multiple score tables

**Table**: `analysis.DIAGNOSES` (re-profiled WITH cross-table context)
**Context injected**: Summaries of all 15 other analysis tables (their names, PKs, and column lists)
**Expected**:
- `SUBJID` or `USUBJID` has `join_paths` referencing at least 3 other tables (e.g., `"ENRDT.SUBJID"`, `"PHQ9A_SCORES.USUBJID"`, `"ASCVD.USUBJID"`)
- `VISIT` or `VISITNUM` has `join_paths` referencing score tables that share the VISIT column

**Baseline (current)**: All `join_paths` are `[]`.

**Rationale**: DIAGNOSES is the hub table connecting to all score tables via SUBJID/USUBJID. With cross-table context, the LLM should identify these relationships.

---

### TC-08: Lab table discovers join to subject tables but not unrelated tables

**Table**: `analysis.MTQC` (re-profiled WITH cross-table context)
**Context injected**: Summaries of ENRDT, DIAGNOSES, PHQ9A_SCORES, ASSAYS
**Expected**:
- `SUBJID` has `join_paths` containing `"ENRDT.SUBJID"` and/or `"DIAGNOSES.SUBJID"`
- `visit` has `join_paths` referencing tables with VISIT columns
- `idat_name` does NOT have join_paths to unrelated tables (no false positives)

**Rationale**: MTQC should join to subject-level tables via SUBJID but should not hallucinate joins for its unique specimen ID column.

---

### TC-09: Cross-table context uses fully-qualified table references

**Table**: Any `analysis.*` table (re-profiled WITH context)
**Expected**:
- All `join_paths` entries use the format `"TABLE_NAME.COLUMN_NAME"` (within same dataset) or `"dataset.TABLE_NAME.COLUMN_NAME"` (cross-dataset)
- No vague references like `"patient_table.id"` or `"other_table.subject_id"`
- Referenced table names match actual table names in the project (e.g., `"ENRDT"`, not `"enrollment"`)

**Rationale**: Join paths must reference real tables by their actual names, not hallucinated or genericized names. The cross-table context provides exact table names, so the LLM should use them verbatim.

---

### TC-10: Entity fields persist through write/read cycle

**Table**: Any table re-profiled with new fields
**Expected**:
- After `profile_semantic()` returns a `SemanticTableProfile` with `entity_anchor`, `entity_type`, and `cohort_dimensions` populated
- After `write_sem_profile()` to GCS and `read_sem_profile()` back:
  - `entity_anchor` value matches what was written
  - `entity_type` value matches what was written
  - `cohort_dimensions` list matches what was written
- The catalog context markdown (`_catalog_context.md`) includes entity metadata in the table summary

**Rationale**: Validates the full round-trip: model fields, JSON serialization, GCS storage, deserialization, and catalog context generation all handle the new fields correctly.

---

## Generalization Tests (non-analysis datasets)

These test cases validate that the features work beyond the `analysis` dataset's uniform SUBJID/VISIT pattern.

### TC-11: Reference/admin table with no entity anchor

**Table**: `admin.SITEHIST` (6 columns) — site history, a reference/lookup table
**Expected**:
- `entity_anchor` = `""` (empty)
- `entity_type` = `""` or `"reference"` or `"site"`
- `cohort_dimensions` = `[]` or a minimal list — reference tables are not cohortable
- The profiler does NOT force-fit a subject/patient entity onto a table that has none

**Rationale**: Not every table is patient-centric. Admin/reference tables (site codes, study metadata, version tables) have no entity to cohort on. The profiler must gracefully handle this rather than hallucinating an entity anchor.

---

### TC-12: Cross-dataset join discovery (analysis ↔ crf)

**Table**: `crf.VS` (24 columns, vital signs CRF) — profiled WITH cross-table context
**Context injected**: Summaries of `analysis.DIAGNOSES`, `analysis.ENRDT`, `analysis.ASCVD`
**Expected**:
- If `VS` contains a `SUBJID` or `USUBJID` column, `join_paths` should reference `analysis.DIAGNOSES.SUBJID` or `analysis.ENRDT.SUBJID` (cross-dataset)
- Join path format should include the dataset name to disambiguate: `"DIAGNOSES.SUBJID"` or `"analysis.DIAGNOSES.SUBJID"`
- If `VS` has a `VISIT`/`VISITNUM` column, it may reference analysis tables' VISIT columns

**Rationale**: Real-world queries often span datasets (CRF source data joined to analysis views). The profiler must suggest cross-dataset joins when the context shows matching columns in other datasets.

---

### TC-13: Multi-entity table with two valid anchors

**Table**: `crf.AE` (47 columns, adverse events) — expected to have both subject-level and event-level identifiers
**Expected**:
- `entity_anchor` = the **primary** entity identifier (likely `USUBJID` or `SUBJID` — the subject)
- `entity_type` = `"adverse_event"` or `"event"` or `"subject"`
- If there is also an event-level ID (e.g., `AESEQ`, `AESPID`, or a sequence number), it should NOT replace the subject anchor but may appear in the granularity description
- `cohort_dimensions` should include adverse event classification columns (severity, category, outcome) but NOT the event sequence IDs

**Rationale**: Some tables have multiple levels of identity — a subject ID and an event/row ID. The profiler must pick the most meaningful entity anchor (usually the subject) rather than the row-level surrogate key, while reflecting the event-level granularity in the `granularity` field.

---

### TC-14: Inconsistent column naming across tables

**Table**: `screener.DM` (30 columns, demographics) — profiled WITH cross-table context
**Context injected**: Summaries of `analysis.ENRDT` (has `SUBJID`), `analysis.DIAGNOSES` (has `SUBJID`, `USUBJID`)
**Expected**:
- If `DM` uses a different ID column name (e.g., `SUBJECTID`, `PTID`, `subject_id`, or just `SUBJID`), the profiler still:
  - Sets `entity_anchor` to whatever the subject ID column is named in DM
  - Suggests `join_paths` to `ENRDT.SUBJID` or `DIAGNOSES.SUBJID` despite the name difference
- `entity_type` = `"subject"` or `"participant"` (demographics is always subject-level)
- `cohort_dimensions` should include demographic columns (age, sex, race, ethnicity, etc.)

**Rationale**: Column naming is inconsistent across real-world datasets. `patient_id`, `SUBJID`, `person_id`, `PTID` all mean the same thing. The LLM must use semantic understanding (not just exact name matching) to identify cross-table joins when names differ.

---

### TC-15: Single-column or minimal table (edge case)

**Table**: `bhs_underlay_index_20250718.VERSION` (1 column) — a metadata/version table
**Expected**:
- `entity_anchor` = `""` (empty)
- `entity_type` = `""` or `"metadata"`
- `cohort_dimensions` = `[]`
- `join_paths` = `[]` for all columns
- The profiler does NOT error, crash, or produce nonsensical output
- Validation status should be `"pass"` or `"warning"` (not `"fail"`)

**Rationale**: Edge case — a table with a single column has no entity, no joins, no cohort dimensions. The profiler must handle degenerate inputs gracefully without forcing fields that don't apply.

---

## Validation Criteria

| Test | Category | Pass Condition |
|------|----------|---------------|
| TC-01 | Entity | `entity_anchor` is `SUBJID`, `entity_type` is subject/participant, `cohort_dimensions` has `enrollment_date` |
| TC-02 | Entity | `entity_anchor` is SUBJID/USUBJID, `cohort_dimensions` has >=5 diagnosis flags, excludes IDs |
| TC-03 | Entity | `entity_anchor` is `USUBJID`, `cohort_dimensions` includes score/visit, excludes IDs |
| TC-04 | Entity | `entity_type` is specimen/sample/assay (NOT subject), `entity_anchor` is `idat_name` or `SUBJID` |
| TC-05 | Entity | `cohort_dimensions` excludes STUDYID/USUBJID/study_day, includes VISIT + clinical flags |
| TC-06 | Context | `join_paths` for SUBJID/USUBJID contains `ENRDT.SUBJID` (was empty before) |
| TC-07 | Context | `join_paths` for SUBJID references >=3 other analysis tables |
| TC-08 | Context | SUBJID joins to subject tables, idat_name does NOT join to unrelated tables |
| TC-09 | Context | All join_path values reference actual table names, not hallucinated names |
| TC-10 | Roundtrip | New fields survive write → GCS → read round-trip and appear in catalog context |
| TC-11 | Generalize | No entity anchor forced onto reference/admin tables; `cohort_dimensions` is empty or minimal |
| TC-12 | Generalize | Cross-dataset joins discovered when context includes tables from other datasets |
| TC-13 | Generalize | Multi-entity table picks subject as primary anchor; event-level ID reflected in granularity only |
| TC-14 | Generalize | Joins discovered despite inconsistent column naming across datasets (SUBJID vs PTID etc.) |
| TC-15 | Generalize | Single-column table produces empty entity/cohort fields without errors or nonsensical output |
