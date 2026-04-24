# Cohort Building & Cross-Table Joins — Research & Feature Plan

**Objective**: Improve semantic profiling so it enables building cohorts across datasets/tables and enables cross-dataset and cross-table joins — using FHIR concepts and the consumption-driven metadata framework as the backbone.

**Date**: 2026-04-24

**Key Reference**: `/product_mgmnt/Metadata/design/consumption_driven_metadata_fields_summary.md` — the 22-field consumption-driven framework that defines what metadata is needed and why.

**Cohort Builder Spec**: `research/cohort_builder_feature_spec.md` — detailed feature spec for the AI-assisted cohort builder (CB-1 through CB-10).

**Test Cases**: `research/test_cases.md` — 5-10 test cases per feature (200+ total), dataset-agnostic, validating behavior across any project.

---

## Executive Summary

**37 features** across 7 priority tiers, delivering a progression from broken bulk profiling → FHIR-aligned metadata → intelligent agent tools → rich frontend → AI-assisted cohort builder → FHIR export.

### What We're Building

The semantic profiler today produces table/column metadata in isolation. This plan transforms it into a **connected metadata graph** where:
- Every column knows what concept it represents (LOINC, ICD-10, SNOMED) and whether it IS the concept or CONTAINS codes from a code system
- Every table knows which other tables it can join to, via what columns, with what cardinality
- Every filterable column carries its allowed values as a value set
- Every measurement column declares how its values were captured (self-reported, lab-measured, calculated)
- An AI agent can use all of this to discover data across datasets, build multi-table cohort queries, and warn about comparability issues
- Users can build cohorts through natural language, AI suggestions, or visual filter builders — then save, query, and share them

### Feature Count by Priority

| Priority | Theme | Features | Status |
|----------|-------|----------|--------|
| **P0** | Fix bulk profiling + validate | 3 (P0-1 to P0-3) | Unblocks everything |
| **P1** | FHIR-aligned concept bindings | 4 (P1-1 to P1-4) | The multiplier — enables all downstream features |
| **P2** | Chat & agent integration | 4 (P2-1 to P2-4) | Makes metadata actionable via NL |
| **P3** | Frontend display | 7 (P3-1 to P3-7) | Users see concept bindings, value sets, join graph |
| **P4** | Profiling quality & scale | 5 (P4-1 to P4-5) | Structured join index, entity reconciliation, concept catalog |
| **P5** | AI-assisted cohort builder | 10 (CB-1 to CB-10) | NL cohorts, AI suggestions, filter UI, persistence |
| **P6** | FHIR export & lineage | 2 (P6-1 to P6-2) | Publish to FHIR IG, data lineage |
| | **Total** | **35** | |

### Key Features at a Glance

**P0 — Ship first** (Week 1)
- P0-1: Pass neighbor context in bulk profiling (~10 lines, fixes empty join_paths)
- P0-2: Two-pass profiling so early tables learn about later ones
- P0-3: Run and validate the 15 P0 test cases

**P1 — The multiplier** (Weeks 2-3)
- P1-1: Split `terminology_bindings` into **Fixed Concept Binding** (column IS LOINC 44261-6) vs **Code System Binding** (column CONTAINS CPT codes) — enables cross-dataset concept matching
- P1-2: **Value Set Binding** on cohort dimensions — `sex → {Male, Female, Other}` — enables filter dropdowns and valid WHERE clauses
- P1-3: **Measurement Method** — self-reported vs lab-measured — enables comparability warnings
- P1-4: **Structural Links** — typed join_paths with cardinality and confidence — replaces free-text strings

**P2 — Agent power-up** (Weeks 3-4)
- P2-1: Expose all new metadata in agent system prompt
- P2-2: `find_joinable_tables` tool
- P2-3: `find_concept_across_tables` tool — "find all PHQ-9 data across all datasets"
- P2-4: Cohort query templates in agent prompt

**P3 — Make it visible** (Weeks 5-6)
- P3-1/2: TypeScript types + entity display
- P3-3: Concept binding badges per column (Fixed vs Code System)
- P3-4: Value set popover on cohort dimensions (with value counts)
- P3-5: "Concepts in this Table" summary panel
- P3-6/7: Clickable join links + join graph visualization

**P4 — Quality at scale** (Week 7)
- P4-1: Structured `_join_index.json` for the whole project
- P4-2: Entity reconciliation (SUBJID = USUBJID = PTID = person_id)
- P4-5: Concept-level catalog context (concept-first view for LLM)

**P5 — Cohort builder** (Weeks 8-11, see `cohort_builder_feature_spec.md`)
- CB-1/2/3: Data model + SQL generator + API endpoints
- CB-4: Natural language → structured cohort definition
- CB-5: AI-suggested cohorts with clinical rationale
- CB-6: Visual filter builder with value set dropdowns + join picker
- CB-7: Results with AI-generated insights (cohort vs base population)
- CB-8/9: Save cohorts to GCS + query-on-cohort via agent CTE
- CB-10: Cohort library page and routing

**P6 — Future** (Week 12+)
- P6-1: Generate FHIR StructureDefinition + ElementDefinition from profiler output
- P6-2: Data lineage tracking

### Critical Path

```
P0 (bulk fix) → P1 (FHIR bindings) → P2 (agent tools) → P5 (cohort builder)
                      ↓
                 P3 (frontend display)
                      ↓
                 P4 (quality at scale) → P6 (FHIR export)
```

P1 is the bottleneck — concept bindings, value set bindings, structural links, and measurement method feed into every downstream feature. Nothing in P2-P5 reaches full potential without P1.

---

## Design Principle: Consumption-Driven Metadata

Every metadata field must be traceable to one of five consumption themes:

| # | Theme | What It Enables |
|---|-------|-----------------|
| 1 | **Asset Discovery** | Find the right data by topic, not by table name |
| 2 | **Cross-Dataset Concept Matching** | Recognize PHQ-9 scores and ICD-10 F32.x both = "depression" |
| 3 | **Data Quality & Fitness** | Assess completeness, temporal frequency, comparability |
| 4 | **Cross-Source Joining & Cohort Assembly** | Join records across tables/datasets via shared keys |
| 5 | **Metric Computation & Segmentation** | Correct aggregations with proper units, group-bys, filters |

The profiler's job is to **auto-generate the AI-populatable fields** (14 of 22) from technical stats + LLM inference, then surface them for steward confirmation.

---

## Gap Analysis: Current Profiler vs. Consumption-Driven Framework

### Level 3 (Table) — 6 fields

| Framework Field | Profiler Field | Status | Gap |
|----------------|---------------|--------|-----|
| **Entity Definition** | `business_name` + `table_definition` | Implemented | Adequate — maps to `SD.title` + `SD.description` |
| **Granularity Definition** | `granularity` | Implemented | Adequate — maps to `SD.purpose` |
| **Primary Identity** | `primary_key` | Implemented | Adequate — maps to `ext[verily-primary-identity]` |
| **Structural Links** | `join_paths` (free-text per column) | Partial | **Free-text strings, not structured FHIR canonical references. No join type, cardinality, or confidence. Not produced in bulk mode.** |
| **Domain Context** | — | Missing | Not in profiler scope (steward-only field) |
| **Lifecycle State** | — | Missing | Not in profiler scope (steward-only field) |

### Level 4 (Column) — 9 fields

| Framework Field | Profiler Field | Status | Gap |
|----------------|---------------|--------|-----|
| **Steward Definition** | `definition` | Implemented | Adequate — maps to `ED.definition` |
| **Fixed Concept Binding** | `terminology_bindings` | Partial | **Profiler conflates Fixed and Code System binding into one list. No distinction between "this column IS LOINC 44261-6" vs "this column CONTAINS CPT codes."** |
| **Code System Binding** | `terminology_bindings` | Partial | **Same issue — both binding types lumped together** |
| **Unit of Measure** | `unit_of_measure` | Implemented | Adequate — could add UCUM validation |
| **Completeness** | `null_percent` (tech profile) | Partial | **Tech profile has stats; no business rule (Required vs Optional). No `ED.min`/`max` equivalent.** |
| **Value Set Binding** | `top_values` / `value_counts` (tech) | Partial | **Tech profile extracts distinct values, but profiler doesn't produce a formal value set binding. Cohort dimensions don't carry allowed values.** |
| **Physical Path** | — (derivable from fq_table + column_name) | Implicit | Could be explicit |
| **Security Label** | `sensitivity` (PHI/PII/UID) | Implemented | Adequate — maps to DS4P `ext-inline-sec-label` |
| **Measurement Method** | — | Missing | **Not implemented. Critical for cross-dataset comparability (self-reported vs lab-measured vs calculated).** |

### Summary: 22 fields → 8 implemented, 6 partial, 4 missing (2 steward-only, 2 need implementation)

---

## How FHIR Concepts Solve the Core Problems

### Problem 1: Cross-Dataset Concept Matching

**Today**: The profiler produces `terminology_bindings` like `[{system: "http://loinc.org", code: "44261-6", display: "PHQ-9 Total Score"}]`. But it doesn't distinguish whether `phq9_total_score` IS the concept (fixed binding) vs. whether `diagnosis_code` CONTAINS ICD-10 codes (code system binding).

**FHIR solution**: Two distinct binding patterns from the design framework:

| Pattern | Meaning | Example | Agent Query Pattern |
|---------|---------|---------|---------------------|
| **Fixed Concept Binding** | Column IS the concept | `phq9_total_score` → LOINC 44261-6 | `WHERE phq9_total_score > 10` |
| **Code System Binding** | Column CONTAINS codes from a terminology | `procedure_code` → CPT system | `WHERE procedure_code = '99213'` |

**Impact**: An agent that knows `phq9_total_score` in BHS and `phq9a_total` in PRESCO both have fixed binding → LOINC 44261-6 can match them as the same concept without the user knowing the column names differ.

### Problem 2: Cohort Assembly Without Valid Values

**Today**: `cohort_dimensions` lists column names suitable for filtering (e.g., `["sex", "smoking_status", "der_hx_diabetes"]`) but doesn't say what values those columns accept.

**FHIR solution**: **Value Set Binding** — each cohort dimension carries its allowed values:
```
sex → ValueSet: {Male, Female, Other, Unknown}
smoking_status → ValueSet: {Current, Former, Never, Unknown}
der_hx_diabetes → ValueSet: {0, 1}
```

**Impact**: Agent can construct valid WHERE clauses without first querying `SELECT DISTINCT`. Users see "filter by sex: Male/Female/Other" instead of a free-text input.

### Problem 3: Structural Links (Joins) as Structured References

**Today**: `join_paths` = `["ENRDT.SUBJID", "DIAGNOSES.SUBJID"]` — untyped strings.

**FHIR solution**: **Structural Links** as typed references with cardinality:
```json
{
  "source_column": "USUBJID",
  "target": "analysis.ENRDT.SUBJID",
  "link_type": "entity_key",
  "cardinality": "many_to_one",
  "confidence": "high"
}
```

Maps to FHIR `ext[verily-structural-link]` with `valueCanonical` pointing to the target StructureDefinition.

### Problem 4: Cross-Dataset Comparability

**Today**: No way to know if `phq9_total_score` in BHS (lab-measured? self-reported?) is comparable to `phq9_score` in PRESCO.

**FHIR solution**: **Measurement Method** — `{self-reported, lab-measured, device-collected, calculated, administrative}`. Agent warns: "BHS PHQ-9 is self-reported; PRESCO PHQ-9 is clinician-administered — scores may not be directly comparable."

---

## Updated Feature Plan

### P0 — Fix Critical Blockers (ship first)

These unblock the entire feature set. Without them, entity/cohort/join data is not reliably produced.

#### P0-1: Bulk profiler passes neighbor context

**Problem**: `bulk_profiler.py` calls `profile_semantic()` without `neighbor_context`, so bulk-profiled tables produce empty `join_paths`.

**Fix**:
- Load `_catalog_context.md` at batch start (from prior profiling runs)
- Pass it to every `profile_semantic()` call in `_run_semantic_batch()` and `_run_pipeline_batch()`

**Files**: `bulk_profiler.py`
**Effort**: Small (~10 lines)
**Risk**: Low

#### P0-2: Two-pass semantic profiling

**Problem**: First batch has no neighbor context. Tables profiled early don't know about tables profiled later.

**Fix**: After bulk batch completes and catalog context is regenerated, run a second pass on tables with empty `join_paths` on entity anchor columns.

**Files**: `bulk_profiler.py`, `profiling_runner.py`
**Effort**: Medium
**Risk**: Medium — add `--two-pass` flag

#### P0-3: Execute and validate P0 test cases (TC-01 through TC-15)

**Files**: `tests/run_p0_tests.py`
**Effort**: Medium
**Risk**: Low

---

### P1 — FHIR-Aligned Concept Bindings (the multiplier)

This is where the consumption-driven framework transforms the profiler. These features are what make cross-dataset concept matching and intelligent cohort assembly possible.

#### P1-1: Split terminology_bindings into Fixed Concept Binding + Code System Binding

**Problem**: The profiler lumps both binding types into one `terminology_bindings` list. An agent can't distinguish "this column IS LOINC 44261-6" from "this column CONTAINS LOINC codes."

**Fix — Model changes** (`models.py`):
```python
@dataclass
class SemanticColumnProfile:
    # ... existing fields ...
    concept_binding: Optional[ConceptBinding] = None     # fixed: column IS this concept
    code_system_binding: Optional[CodeSystemBinding] = None  # column CONTAINS codes from this system
    terminology_bindings: list[TerminologyBinding] = ...  # keep for backward compat
```

```python
@dataclass
class ConceptBinding:
    system: str      # e.g. "http://loinc.org"
    code: str        # e.g. "44261-6"
    display: str     # e.g. "PHQ-9 Total Score"
    confidence: str  # high/medium/low

@dataclass
class CodeSystemBinding:
    system: str      # e.g. "http://www.ama-assn.org/go/cpt"
    display: str     # e.g. "CPT Procedure Codes"
    confidence: str
```

**Fix — Prompt changes** (`semantic.py`):
Update system prompt to instruct the LLM to distinguish:
- `concept_binding`: "If this column always represents ONE specific concept (e.g., `phq9_total_score` always = PHQ-9 score), bind it to the concept's code."
- `code_system_binding`: "If this column contains codes FROM a code system (e.g., `procedure_code` contains CPT codes), bind it to the code system."

**Consumption themes served**: Asset Discovery, Cross-Dataset Concept Matching
**Files**: `models.py`, `semantic.py`, `catalog_context.py`
**Effort**: Medium
**Risk**: Low — backward compatible (keep `terminology_bindings` populated)

#### P1-2: Value Set Binding for cohort dimensions

**Problem**: `cohort_dimensions` lists column names but not their allowed values. Agent must query `SELECT DISTINCT` to know what to filter on.

**Fix**: Extend `SemanticColumnProfile` with:
```python
value_set_binding: list[str] = field(default_factory=list)  # allowed/common values
```

Population strategy:
- For columns in `cohort_dimensions`, the profiler already has `top_values` and `value_counts` from tech profile
- LLM confirms which values are semantically meaningful (exclude nulls, codes like "99999")
- Store as `value_set_binding` on the column

**Prompt addition**: "For each column in `cohort_dimensions`, also populate `value_set_binding` — the list of allowed or meaningful categorical values from the technical stats. Exclude null placeholders and sentinel values."

**Consumption themes served**: Cross-Dataset Concept Matching, Metric Computation
**Files**: `models.py`, `semantic.py`
**Effort**: Small
**Risk**: Low

#### P1-3: Measurement Method

**Problem**: No way to assess cross-dataset comparability. Is `phq9_total_score` self-reported or clinician-administered?

**Fix**: Add to `SemanticColumnProfile`:
```python
measurement_method: str = ""  # self-reported, lab-measured, device-collected, calculated, administrative
```

Prompt the LLM: "For measurement/observation columns, classify how the value was captured: `self-reported` (questionnaires, surveys), `lab-measured` (blood assays, lab panels), `device-collected` (sensors, wearables), `calculated` (derived scores, risk models), or `administrative` (billing codes, enrollment flags). Set to empty if not applicable."

Maps to FHIR: `ext[VerilyAttributeSemanticMetadata]` bound to `VerilyMeasurementMethod_VS`

**Consumption themes served**: Data Quality & Fitness, Cross-Dataset Concept Matching
**Files**: `models.py`, `semantic.py`
**Effort**: Small
**Risk**: Low

#### P1-4: Structural Links (typed, structured join_paths)

**Problem**: `join_paths` = `["ENRDT.SUBJID"]` — no type, cardinality, or confidence.

**Fix**: Add a structured representation alongside the existing `join_paths`:
```python
@dataclass
class StructuralLink:
    source_column: str
    target_table: str          # fully qualified
    target_column: str
    link_type: str             # entity_key, foreign_key, shared_dimension, temporal
    cardinality: str           # one_to_one, one_to_many, many_to_one, many_to_many
    confidence: str            # high, medium, low

@dataclass
class SemanticTableProfile:
    # ... existing fields ...
    structural_links: list[StructuralLink] = field(default_factory=list)
```

The LLM prompt already asks for `join_paths` per column. Post-processing aggregates these into table-level `structural_links` with type and cardinality inferred from:
- Entity anchor columns → `entity_key` link type
- Matching column names → `foreign_key`
- Shared categorical columns (VISIT, site) → `shared_dimension`

Maps to FHIR: `ext[verily-structural-link]` with `valueCanonical`

**Consumption themes served**: Cross-Source Joining & Cohort Assembly
**Files**: `models.py`, `semantic.py`, `catalog_context.py`
**Effort**: Medium
**Risk**: Low — `join_paths` kept for backward compat

---

### P2 — Chat & Agent Integration

These make the FHIR-aligned metadata actionable through the agent.

#### P2-1: Expose entity, cohort, concept bindings in chat prompt

**Fix**: Update `context.py:format_table_for_prompt()` to include:
- Entity anchor + entity type
- Cohort dimensions with their value set bindings
- Fixed concept bindings (so agent can match columns across tables by concept)
- Structural links (so agent knows which joins are valid)
- Measurement method (so agent can warn about comparability)

Add agent prompt rules:
- "Use `concept_binding` to match columns across tables that represent the same concept"
- "Use `cohort_dimensions` with `value_set_binding` for WHERE clauses"
- "Use `structural_links` for JOIN conditions — prefer `entity_key` links"
- "When comparing values across datasets, check `measurement_method` for comparability"

**Files**: `packages/verily-chat/src/verily_chat/context.py`
**Effort**: Small
**Risk**: Low

#### P2-2: Add `find_joinable_tables` agent tool

Returns structural links and join paths for a given table, grouped by link type.

**Files**: `packages/verily-chat/src/verily_chat/agent.py`
**Effort**: Medium
**Risk**: Low

#### P2-3: Add `find_concept_across_tables` agent tool

**New tool** — the concept-matching power tool:
```python
def find_concept_across_tables(concept: str) -> str:
    """Find all columns across all tables that represent a given concept.
    Searches by concept code (e.g. LOINC:44261-6), display name (e.g. 'PHQ-9'),
    or column name pattern (e.g. 'phq9')."""
```

This enables the cross-dataset concept matching theme: "Find all PHQ-9 data across BHS and PRESCO" → agent searches concept bindings across all profiled tables.

**Files**: `packages/verily-chat/src/verily_chat/agent.py`
**Effort**: Medium
**Risk**: Low

#### P2-4: Cohort query builder in agent prompt

Add cohort query templates that use the new metadata:
```sql
-- Cross-dataset cohort using concept bindings
-- phq9_total_score (BHS) and phq9a_total (PRESCO) both → LOINC 44261-6
SELECT b.SUBJID, b.phq9_total_score, p.phq9a_total
FROM `bhs.analysis.SCORES` b
JOIN `presco.analysis.PHQ9` p
  ON b.SUBJID = p.participant_id  -- structural_link: entity_key
WHERE b.phq9_total_score > 10     -- cohort_dimension + value_set_binding
```

**Files**: `packages/verily-chat/src/verily_chat/context.py`
**Effort**: Small
**Risk**: Low

---

### P3 — Frontend & UI

#### P3-1: Add new fields to TypeScript types

Add `entity_anchor`, `entity_type`, `cohort_dimensions`, `concept_binding`, `code_system_binding`, `measurement_method`, `value_set_binding`, `structural_links` to TypeScript interfaces.

**Files**: `frontend/src/types/profile.ts`
**Effort**: Small
**Risk**: Low

#### P3-2: Entity & concept metadata display

Show in the semantic profile view:
- Entity Anchor badge with entity type
- Measurement method indicator per column
- Structural links section with link type badges

**Files**: `frontend/src/components/SemProfile.tsx`
**Effort**: Medium
**Risk**: Low

#### P3-3: Concept Binding display (per column)

Each column shows its binding type and detail inline in the profile table:

- **Fixed Concept Binding**: Rendered as a badge with the terminology system icon + code + display name. E.g., `LOINC 44261-6 — PHQ-9 Total Score` with a "Fixed" indicator. Clicking the code opens the terminology reference (e.g., loinc.org lookup).
- **Code System Binding**: Rendered as a badge showing the terminology system the column draws from. E.g., `CPT (procedure codes)` with a "Code System" indicator. Distinct visual treatment from Fixed — different color/icon to make the pattern difference obvious.
- **No binding**: Columns with no concept or code system binding show nothing (no clutter).

Users should see at a glance: "This column IS a PHQ-9 score" vs "This column CONTAINS CPT codes."

**Files**: `frontend/src/components/SemProfile.tsx`, `frontend/src/types/profile.ts`
**Effort**: Small
**Risk**: Low

#### P3-4: Value Set Binding display (per cohort dimension)

For columns marked as `cohort_dimensions`, show their `value_set_binding` — the allowed/meaningful values:

- In the column table, cohort dimension columns get a "Cohort Filter" tag
- Clicking the tag (or hovering) shows a popover/dropdown with the value set: e.g., `sex → {Male, Female, Other, Unknown}`
- Value sets with many values (>15) show the top values with a "... and N more" indicator
- If `value_counts` are available from the tech profile, show counts alongside values: `Male (1,247) | Female (1,255) | Unknown (12)`

This lets users see what they can filter on without running a query.

**Files**: `frontend/src/components/SemProfile.tsx`, possibly new `ValueSetPopover.tsx`
**Effort**: Medium
**Risk**: Low

#### P3-5: Terminology & concept bindings summary panel

A new collapsible section in the table detail view (above or below the column table) that aggregates all concept and code system bindings for the table:

**"Concepts in this Table" panel**:
```
Fixed Concept Bindings:
  LOINC 44261-6 — PHQ-9 Total Score       → phq9_total_score
  LOINC 44261-7 — PHQ-9 Item Score        → phq9_item_1, phq9_item_2, ...

Code System Bindings:
  ICD-10 (Diagnosis codes)                → diagnosis_code, icd10code1
  CPT (Procedure codes)                   → procedure_code

Cohort Dimensions (6):
  sex: Male, Female, Other, Unknown
  smoking_status: Current, Former, Never
  der_hx_diabetes: 0, 1
  VISIT: Baseline, Month 3, Month 6, Month 12
  ...
```

This gives a fast overview of the semantic richness of the table — what standard concepts it maps to, what code systems it uses, and what dimensions it can be filtered on. Useful for the "Asset Discovery" consumption theme.

**Files**: New `frontend/src/components/ConceptSummaryPanel.tsx`, integrate into `TablePage.tsx`
**Effort**: Medium
**Risk**: Low

#### P3-6: Join paths as clickable links

Render structural links as navigable links to target table detail pages.

**Files**: `frontend/src/components/SemProfile.tsx`
**Effort**: Small
**Risk**: Low

#### P3-7: Join graph visualization

Tables as nodes, structural_links as typed edges. Entity type as node color, link type as edge style.

**Files**: New `frontend/src/components/JoinGraph.tsx`
**Effort**: Large
**Risk**: Medium

---

### P4 — Profiling Quality & Scale

#### P4-1: Structured join index (`_join_index.json`)

Project-level file aggregating all structural_links across all tables. Enables:
- Agent `find_joinable_tables` without scanning all profiles
- Frontend join graph without reading all semantic profiles
- Validation: bidirectional links, orphan table detection

**Files**: `models.py`, `storage.py`, new `join_index.py`
**Effort**: Medium
**Risk**: Low

#### P4-2: Entity reconciliation across tables

Group tables by entity_type. Within each group, identify columns representing the same entity (SUBJID, USUBJID, PTID, person_id). Produce an entity registry:
```json
{
  "subject": {
    "canonical_name": "subject_id",
    "aliases": ["SUBJID", "USUBJID", "PTID", "person_id", "participant_id"],
    "tables": ["analysis.ENRDT", "analysis.DIAGNOSES", "crf.DM", ...]
  }
}
```

Inject into future profiling prompts so the LLM knows these are equivalent.

**Files**: New `entity_reconcile.py`
**Effort**: Large
**Risk**: Medium

#### P4-3: Smart neighbor context (prioritized, not truncated)

Rank neighbor tables by relevance to the table being profiled:
1. Same dataset, shared column names → full detail
2. Same entity_type, different dataset → summary
3. Unrelated → name only

Budget: 12KB detailed + 4KB summary

**Files**: `semantic.py`, `catalog_context.py`
**Effort**: Medium
**Risk**: Low

#### P4-4: Cross-dataset join format standardization

Enforce `DATASET.TABLE.COLUMN` format for cross-dataset joins. Post-processing validation in `semantic.py`.

**Files**: `semantic.py`
**Effort**: Small
**Risk**: Low

#### P4-5: Concept-level catalog context

In addition to table-by-table catalog context, generate a **concept index**:
```markdown
## Concepts in this project

### Depression Screening (LOINC 44261-6 — PHQ-9 Total Score)
- analysis.SCORES.phq9_total_score (Fixed, self-reported)
- analysis.PHQ9A_SCORES.score (Fixed, self-reported)

### Subject Identifier
- analysis.ENRDT.SUBJID (entity_key)
- analysis.DIAGNOSES.SUBJID (entity_key)
- analysis.DIAGNOSES.USUBJID (entity_key)
- crf.DM.SUBJID (entity_key)
```

This gives the LLM a concept-first view for cross-table matching.

**Files**: `catalog_context.py`
**Effort**: Medium
**Risk**: Low

---

### P5 — AI-Assisted Cohort Builder

Full feature spec: **`research/cohort_builder_feature_spec.md`**

10 features (CB-1 through CB-10) delivering three interaction modes:

1. **Natural Language**: "Find diabetic females with PHQ-9 above 10" → AI translates to structured cohort definition using concept bindings and structural links
2. **AI-Suggested Cohorts**: System proactively suggests interesting cohorts with clinical rationale, estimated size, and analysis opportunities
3. **Filter-Based Builder**: Visual UI with cohort dimension dropdowns (populated from value set bindings), join picker (from structural links), and live count estimation

After creation, cohorts are first-class objects: queryable (via agent-scoped CTE), saveable (GCS), and browsable (cohort library page).

| Phase | Features | Delivers |
|-------|----------|----------|
| Phase 1 | CB-1, CB-2, CB-3, CB-10 | Data model, SQL generator, API, routing |
| Phase 2 | CB-4, CB-5 | NL → cohort, AI suggestions |
| Phase 3 | CB-6, CB-7 | Filter builder UI, results + AI insights |
| Phase 4 | CB-8, CB-9 | Persistence, query-on-cohort |

**Depends on**: P1-1 (concept bindings), P1-2 (value set bindings), P1-3 (measurement method), P1-4 (structural links)

**Relationship to P2**: P2-4 (cohort query templates in agent prompt) is a lightweight precursor to CB-4 (full NL → cohort). P2-4 teaches the agent SQL patterns; CB-4 produces structured `CohortDefinition` JSON. Ship P2-4 first as it improves agent behavior immediately; CB-4 supersedes it when the full builder ships. Similarly, P2-2 (`find_joinable_tables`) and P2-3 (`find_concept_across_tables`) are reused by CB-9 (query on cohort) — the agent tools built in P2 become the cohort-scoped agent's tools in CB-9.

---

### P6 — FHIR Export & Lineage (future)

#### P6-1: FHIR export (StructureDefinition generation)

Generate FHIR `StructureDefinition` + `ElementDefinition` resources from profiling output. Enables publishing profiled tables to the FHIR IG.

Maps:
- `SemanticTableProfile` → `StructureDefinition` (L3 fields)
- `SemanticColumnProfile` → `ElementDefinition` (L4 fields)
- `concept_binding` → `ED.mapping[vfig]`
- `code_system_binding` → `ED.mapping[vfig]`
- `value_set_binding` → `ED.binding.valueSet`
- `measurement_method` → `ext[VerilyAttributeSemanticMetadata]`
- `structural_links` → `ext[verily-structural-link]`
- `primary_key` → `ext[verily-primary-identity]`

**Effort**: Large

#### P6-2: Data lineage integration

Track upstream transformations (CRF → analysis views). Validate join paths against actual data flow.

**Effort**: Very Large

---

## Priority Summary

| Priority | Features | Theme | Effort |
|----------|----------|-------|--------|
| **P0** | P0-1, P0-2, P0-3 | Fix bulk profiling + validate | Small–Medium |
| **P1** | P1-1, P1-2, P1-3, P1-4 | FHIR-aligned concept bindings + structural links | Small–Medium |
| **P2** | P2-1, P2-2, P2-3, P2-4 | Agent uses FHIR metadata for cohort queries | Small–Medium |
| **P3** | P3-1 through P3-7 | Frontend displays concept bindings, value sets, structural links | Small–Large |
| **P4** | P4-1 through P4-5 | Profiling quality, entity reconciliation, concept index | Medium–Large |
| **P5** | CB-1 through CB-10 | AI-assisted cohort builder (NL, suggestions, filter UI, persistence) | Medium–Large |
| **P6** | P6-1, P6-2 | FHIR export, data lineage | Large–Very Large |

## Suggested Execution Order

```
Week 1:  P0-1 → P0-2 → P0-3        Fix bulk profiling, validate
Week 2:  P1-1 → P1-2 → P1-3        FHIR concept bindings + measurement method
Week 3:  P1-4 → P2-1 → P2-2        Structural links + agent prompt
Week 4:  P2-3 → P2-4               Concept search tool + cohort templates
Week 5:  P3-1 → P3-2 → P3-3 → P3-4  Frontend types + concept/value set display
Week 6:  P3-5 → P3-6               Concept summary panel + clickable joins
Week 7:  P4-1 → P4-4 → P4-5        Join index + concept-level catalog
Week 8:  CB-1 → CB-2 → CB-3 → CB-10  Cohort builder foundation (model, SQL, API, routing)
Week 9:  CB-4 → CB-5               NL → cohort + AI-suggested cohorts
Week 10: CB-6 → CB-7               Filter builder UI + results with AI insights
Week 11: CB-8 → CB-9               Cohort persistence + query-on-cohort
Week 12+: P4-2, P4-3, P3-7, P6-*   Entity reconciliation, graph viz, FHIR export, lineage
```

---

## Dependencies

```
P0-1 (bulk context) ──→ P0-2 (two-pass) ──→ P0-3 (validate)
                                                    │
P1-1 (concept bindings) ──→ P2-1 (agent prompt) ───┘
P1-2 (value set binding) ──→ P2-4 (cohort templates)
P1-3 (measurement method) ──→ P2-1 (agent prompt)
P1-4 (structural links) ──→ P2-2 (join tool)
                          ──→ P4-1 (join index) ──→ P3-4 (graph viz)
P1-1 ──→ P2-3 (concept search tool)
P1-1 ──→ P4-5 (concept-level catalog)
P1-* ──→ P6-1 (FHIR export)

Cohort builder (P5):
P1-1 (concept bindings)   ──→ CB-4 (NL → cohort uses concepts to find columns)
P1-2 (value set binding)  ──→ CB-6 (filter builder populates dropdowns from value sets)
P1-3 (measurement method) ──→ CB-7 (insights warn about cross-dataset comparability)
P1-4 (structural links)   ──→ CB-2 (SQL generator uses structural links for JOINs)
                           ──→ CB-6 (join picker offers tables from structural links)
P2-2 (find_joinable_tables) ──→ CB-9 (query-on-cohort agent reuses tool)
P2-3 (find_concept_across)  ──→ CB-9 (query-on-cohort agent reuses tool)
P2-4 (cohort templates)     ──→ CB-4 (NL → cohort supersedes simple templates)
CB-1 (model) ──→ CB-2 (SQL) ──→ CB-3 (API) ──→ CB-6 (builder UI)
CB-3 (API) ──→ CB-4 (NL) ──→ CB-5 (AI suggestions)
CB-3 (API) ──→ CB-8 (persistence) ──→ CB-9 (query on cohort)
CB-6 (builder UI) ──→ CB-7 (results + insights)

P3-1 (TS types) ──→ P3-2 (entity display)
P1-1 (concept bindings) ──→ P3-1 (TS types) ──→ P3-3 (concept binding display)
P1-2 (value set binding) ──→ P3-1 (TS types) ──→ P3-4 (value set display)
P3-3 + P3-4 ──→ P3-5 (concept summary panel)
P1-4 (structural links) ──→ P3-1 (TS types) ──→ P3-6 (clickable joins)
P4-1 (join index) ──→ P3-7 (graph viz)
P4-2 (entity reconciliation) ──→ P4-3 (smart context)
```

---

## Mapping: Profiler Output → Consumption-Driven Framework → FHIR

| Profiler Field | Framework Field | FHIR Element | Status |
|---------------|----------------|--------------|--------|
| `business_name` + `table_definition` | L3 Entity Definition | `SD.title` + `SD.description` | Done |
| `granularity` | L3 Granularity Definition | `SD.purpose` | Done |
| `primary_key` | L3 Primary Identity | `ext[verily-primary-identity]` | Done |
| `structural_links` (NEW) | L3 Structural Links | `ext[verily-structural-link]` | **P1-4** |
| `semantic_domain` | — (closest: L1 Domain Archetype) | `SD.useContext` | Done |
| `entity_anchor` + `entity_type` | L3 Entity Definition (refined) | `SD.title` + `SD.description` | Done |
| `definition` | L4 Steward Definition | `ED.definition` | Done |
| `concept_binding` (NEW) | L4 Fixed Concept Binding | `ED.mapping[vfig]` | **P1-1** |
| `code_system_binding` (NEW) | L4 Code System Binding | `ED.mapping[vfig]` | **P1-1** |
| `unit_of_measure` | L4 Unit of Measure | `ED.type[Quantity]` + UCUM | Done |
| `sensitivity` | L4 Security Label | DS4P `ext-inline-sec-label` | Done |
| `value_set_binding` (NEW) | L4 Value Set Binding | `ED.binding.valueSet` | **P1-2** |
| `measurement_method` (NEW) | L4 Measurement Method | `ext[VerilyAttributeSemanticMetadata]` | **P1-3** |
| `null_percent` (tech) | L4 Completeness (stats) | `MeasureReport` | Done (tech) |
| — | L4 Completeness (rules) | `ED.min` / `ED.max` | **P1-2** (partial) |
| — | L3 Domain Context | `SD.contact[].telecom[url]` | Steward-only |
| — | L3 Lifecycle State | `SD.status` | Steward-only |
| — | L2 Business Intent | `Library.description` | Steward-only |
| — | L2 Component Inventory | `Library.relatedArtifact` | Auto-discoverable |
| — | L2 Source Domain | `Library.contact` | Steward-only |
| — | L2 Product Type | `Library.type` | Steward-only |
| — | L1 Domain Archetype | `Organization.type` | Steward-only |
| — | L1 Hierarchy Position | `Organization.partOf` | Steward-only |
| — | L1 Environment Intent | `Organization.meta.tag` | Steward-only |
