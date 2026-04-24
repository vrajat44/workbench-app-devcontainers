# AI-Assisted Cohort Builder — Feature Spec

**Date**: 2026-04-24
**Parent**: `research/cohort_and_joins_feature_plan.md` (P5-1, P5-2, P5-3)
**Depends on**: P1 (FHIR concept bindings, value set bindings, structural links)

---

## Vision

The cohort builder lets users define, explore, and reuse patient/subject cohorts across the data catalog — through three complementary interaction modes:

1. **Natural Language** — "Show me diabetic females over 50 with elevated PHQ-9 scores"
2. **AI-Suggested Cohorts** — System proactively suggests interesting cohorts with rationale
3. **Filter-Based Builder** — Visual filter UI using cohort dimensions and value set bindings

After creation, cohorts become first-class objects: queryable, shareable, and composable.

---

## User Journeys

### Journey 1: Natural Language Cohort

```
User:  "Find participants with diabetes and depression scores above 10"

AI:    I found relevant data across 2 tables:
       - analysis.DIAGNOSES: der_hx_diabetes (binary flag, 1=yes)
       - analysis.PHQ9A_SCORES: score (PHQ-9 total, LOINC 44261-6)

       These tables join on SUBJID → USUBJID (entity_key, high confidence).

       Proposed cohort:
       ┌─────────────────────────────────────────────┐
       │  Diabetic Participants with Elevated PHQ-9   │
       │                                             │
       │  Base: analysis.DIAGNOSES                   │
       │  Filter: der_hx_diabetes = 1                │
       │  Join: analysis.PHQ9A_SCORES ON SUBJID      │
       │  Filter: score > 10                         │
       │                                             │
       │  Estimated: ~847 subjects                   │
       │  [Preview] [Save] [Modify Filters]          │
       └─────────────────────────────────────────────┘

       Note: PHQ-9 scores are self-reported (measurement_method).
       der_hx_diabetes is a calculated derived flag.

User:  clicks [Preview]

AI:    Shows first 50 rows + summary stats:
       - 847 unique subjects
       - Mean PHQ-9 score: 14.2 (SD 4.8)
       - 62% Female, 38% Male
       - Visits: Baseline (312), Month 3 (285), Month 6 (250)
```

### Journey 2: AI-Suggested Cohorts

```
User:  Opens the Cohort Builder tab on analysis.DIAGNOSES

System: Based on the profiled metadata, here are interesting cohorts:

       1. Cardiovascular Risk Cohort (est. 1,204 subjects)
          Filters: der_hx_cvd = 1 OR der_hx_cad = 1 OR der_hx_hypertension = 1
          Why: 3 binary cardiovascular flags with 15-40% prevalence each.
          Combined cohort captures broad CV risk. Joinable to ASCVD scores
          (structural_link: USUBJID) for risk stratification.

       2. Female Smokers (est. 389 subjects)
          Filters: sex = "Female" AND smoking_status = "Current"
          Why: Intersection of two high-value cohort dimensions.
          8.2% of the population — large enough for statistical analysis.
          Joinable to PHQ9A_SCORES for mental health correlation.

       3. Multi-Visit Longitudinal Cohort (est. 1,891 subjects)
          Filters: VISIT IN ("Baseline", "Month 3", "Month 6", "Month 12")
          Join: PHQ9A_SCORES ON USUBJID
          Why: Subjects with 4+ timepoints enable longitudinal analysis.
          PHQ-9 score trends over time.

       [Use This] [Modify] [Dismiss] for each suggestion
```

### Journey 3: Filter-Based Builder

```
User:  Opens Cohort Builder, selects base table: analysis.DIAGNOSES

System: Shows cohort dimensions with value sets:

       ┌─ Cohort Dimensions ─────────────────────────┐
       │                                             │
       │  sex          [Male ▾] [Female] [Other]     │
       │  smoking_status [Current] [Former] [Never]  │
       │  der_hx_cvd    [0] [1]                      │
       │  der_hx_diabetes [0] [1]                    │
       │  der_hx_cad    [0] [1]                      │
       │  der_hx_hypertension [0] [1]                │
       │                                             │
       │  + Add filter                               │
       │  + Add join to another table                │
       │                                             │
       │  ──────────────────────────────────────────  │
       │  Matching: ~2,502 of 2,502 subjects         │
       │  [Preview Results] [Save Cohort]            │
       └─────────────────────────────────────────────┘

User:  Clicks sex=Female, der_hx_diabetes=1
       Live counter updates: "Matching: ~421 of 2,502 subjects"

User:  Clicks "+ Add join to another table"
       System shows joinable tables from structural_links:
       - analysis.PHQ9A_SCORES (via SUBJID → USUBJID, entity_key)
       - analysis.ASCVD (via USUBJID, entity_key)
       - analysis.ENRDT (via SUBJID, entity_key)

User:  Selects PHQ9A_SCORES, adds filter: score > 10
       Live counter: "Matching: ~189 subjects (across 2 tables)"
```

---

## Feature Breakdown

### CB-1: Cohort Definition Model

The core data structure that all three modes produce.

**Backend model** (`backend/cohort_models.py`):
```python
@dataclass
class CohortFilter:
    column: str
    operator: str       # ==, !=, >, >=, <, <=, in, not_in, between, is_null, is_not_null
    value: Any
    data_type: str      # string, numeric, date, boolean

@dataclass
class CohortJoin:
    target_table: str           # fq table name
    source_column: str          # column in base table
    target_column: str          # column in target table
    link_type: str              # entity_key, foreign_key, shared_dimension
    filters: list[CohortFilter] # filters on the joined table

@dataclass
class CohortDefinition:
    cohort_id: str                  # UUID
    name: str
    description: str
    base_table: str                 # fq table name
    filters: list[CohortFilter]    # filters on base table
    joins: list[CohortJoin]        # joined tables with their filters
    logic: str                     # AND (default) — top-level filter combination
    status: str                    # draft, validated, executed, saved
    created_at: str
    modified_at: str
    result_count: Optional[int]
    result_sql: Optional[str]
    insights: Optional[dict]       # AI-generated summary stats
```

**Frontend type** (`frontend/src/types/cohort.ts`):
```typescript
interface CohortFilter {
  column: string;
  operator: "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not_in" | "between" | "is_null" | "is_not_null";
  value: any;
  dataType: "string" | "numeric" | "date" | "boolean";
}

interface CohortJoin {
  targetTable: string;
  sourceColumn: string;
  targetColumn: string;
  linkType: "entity_key" | "foreign_key" | "shared_dimension";
  filters: CohortFilter[];
}

interface CohortDefinition {
  cohortId: string;
  name: string;
  description: string;
  baseTable: string;
  filters: CohortFilter[];
  joins: CohortJoin[];
  status: "draft" | "validated" | "executed" | "saved";
  createdAt: string;
  modifiedAt: string;
  resultCount?: number;
  resultSql?: string;
  insights?: CohortInsights;
}

interface CohortInsights {
  totalSubjects: number;
  percentOfBase: number;
  dimensionBreakdowns: Record<string, Record<string, number>>;
  aiSummary: string;
}
```

**Effort**: Small
**Files**: New `backend/cohort_models.py`, new `frontend/src/types/cohort.ts`

---

### CB-2: Cohort SQL Generator

Translates `CohortDefinition` to BigQuery SQL. Reuses patterns from `gw_computation.py`.

**Core function** (`backend/cohort_engine.py`):
```python
def cohort_to_sql(cohort: CohortDefinition, mode: str = "preview") -> str:
    """
    mode="count"   → SELECT COUNT(DISTINCT entity_anchor) ...
    mode="preview"  → SELECT * ... LIMIT 500
    mode="full"     → SELECT * ... (no limit, for saved cohort)
    mode="summary"  → SELECT cohort_dim, COUNT(*) ... GROUP BY cohort_dim
    """
```

**SQL generation rules**:
- Base table WHERE clause from `filters`
- Each `CohortJoin` becomes a JOIN clause using `source_column` = `target_column`
- Join filters added to the JOIN's ON clause or a sub-WHERE
- Entity anchor column used for COUNT(DISTINCT) in count mode
- Cross-dataset joins use fully-qualified table names

**Example output**:
```sql
-- mode="preview"
SELECT d.*, s.score, s.VISIT AS score_visit
FROM `wb-beamish-acorn-6393.analysis.DIAGNOSES` d
JOIN `wb-beamish-acorn-6393.analysis.PHQ9A_SCORES` s
  ON d.SUBJID = s.USUBJID
WHERE d.sex = 'Female'
  AND d.der_hx_diabetes = 1
  AND s.score > 10
LIMIT 500

-- mode="count"
SELECT COUNT(DISTINCT d.SUBJID) AS cohort_size
FROM `wb-beamish-acorn-6393.analysis.DIAGNOSES` d
JOIN `wb-beamish-acorn-6393.analysis.PHQ9A_SCORES` s
  ON d.SUBJID = s.USUBJID
WHERE d.sex = 'Female'
  AND d.der_hx_diabetes = 1
  AND s.score > 10
```

**Effort**: Medium
**Files**: New `backend/cohort_engine.py`

---

### CB-3: Cohort API Endpoints

**Backend endpoints** (added to `main.py`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/cohorts/validate` | Validate filters, estimate size, return SQL |
| `POST` | `/api/cohorts/execute` | Execute cohort query, return sample rows |
| `POST` | `/api/cohorts/suggest` | AI suggests interesting cohorts for a table |
| `POST` | `/api/cohorts/from-natural-language` | Parse NL description into CohortDefinition |
| `POST` | `/api/cohorts/save` | Persist cohort definition to GCS |
| `GET`  | `/api/cohorts` | List all saved cohorts for the project |
| `GET`  | `/api/cohorts/{cohort_id}` | Get a saved cohort definition |
| `DELETE` | `/api/cohorts/{cohort_id}` | Delete a saved cohort |
| `POST` | `/api/cohorts/{cohort_id}/query` | Run arbitrary SQL scoped to the cohort |

**Validate response**:
```json
{
  "valid": true,
  "estimated_rows": 847,
  "sql": "SELECT COUNT(DISTINCT ...) ...",
  "warnings": ["PHQ-9 scores are self-reported (measurement_method)"],
  "errors": []
}
```

**Execute response**:
```json
{
  "cohort_id": "c-uuid-1234",
  "result_count": 847,
  "sample_rows": [...],
  "sql": "SELECT d.*, s.score ...",
  "column_metadata": [...],
  "executed_at": "2026-04-24T10:30:00Z"
}
```

**Effort**: Medium
**Files**: `backend/main.py`, `backend/cohort_engine.py`

---

### CB-4: Natural Language → Cohort (AI Translation)

Uses the existing agent infrastructure to parse natural language into a `CohortDefinition`.

**How it works**:

1. User types: "Diabetic females with PHQ-9 above 10"
2. Backend sends to Gemini with a specialized system prompt:
   - Catalog context (tables, columns, concept bindings, value sets, structural links)
   - Instructions to produce a JSON `CohortDefinition`
   - Examples of NL → filter translation
3. LLM returns structured JSON
4. Backend validates: columns exist, operators match data types, joins are valid structural links
5. Returns `CohortDefinition` with `status: "draft"` for user review

**System prompt additions**:
```
You are a cohort builder assistant. Given a natural language description of a patient
cohort, produce a JSON CohortDefinition using ONLY columns, tables, and joins that
exist in the catalog context.

Rules:
- Use concept_binding to find the right columns (e.g., "PHQ-9" → LOINC 44261-6 →
  columns with that fixed concept binding)
- Use value_set_binding for categorical filters (e.g., "female" → sex column where
  "Female" is in the value set)
- Use structural_links for joins (only suggest joins with link_type=entity_key or
  foreign_key)
- Use entity_anchor for COUNT(DISTINCT) in cohort sizing
- Note measurement_method when comparing values across tables
```

**Effort**: Medium
**Files**: New `backend/cohort_ai.py`

---

### CB-5: AI-Suggested Cohorts

Proactively generates interesting cohort suggestions when the user opens the cohort builder on a table.

**How it works**:

1. User opens Cohort Builder tab on `analysis.DIAGNOSES`
2. Backend loads:
   - Semantic profile (cohort_dimensions, value_set_bindings, concept_bindings)
   - Technical profile (value_counts, distinct_counts for cardinality)
   - Structural links (joinable tables)
3. Sends to Gemini with prompt:

```
Given this table's metadata, suggest 3-5 interesting patient cohorts. For each:
- Name and one-sentence description
- Specific filters using cohort_dimensions and their value_set_bindings
- Optional join to a related table (using structural_links)
- Why this cohort is interesting:
  - Clinical or research significance
  - Approximate size (use value_counts to estimate)
  - What analysis it enables (e.g., "longitudinal PHQ-9 trends", "CV risk stratification")
- Confidence: high/medium/low

Prioritize cohorts that:
1. Use multiple cohort dimensions (intersection = more specific)
2. Leverage cross-table joins (multi-table cohorts are harder to discover manually)
3. Have clinically meaningful subgroups (not just random combinations)
4. Are large enough for statistical analysis (est. >50 subjects)
```

4. Returns list of `CohortSuggestion` objects with rationale

**Suggestion model**:
```python
@dataclass
class CohortSuggestion:
    name: str
    description: str
    rationale: str              # why this cohort is interesting
    cohort: CohortDefinition    # ready-to-use definition
    estimated_size: int
    confidence: str             # high, medium, low
    analysis_opportunities: list[str]  # what you could do with this cohort
```

**UI**: Rendered as cards in the Cohort Builder tab, each with [Use This], [Modify], [Dismiss].

**Effort**: Medium
**Files**: `backend/cohort_ai.py`

---

### CB-6: Filter-Based Cohort Builder UI

The visual filter builder — the primary UI for the cohort builder tab.

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Cohort Builder                                   [NL Mode] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Base Table: analysis.DIAGNOSES (2,502 rows)                │
│  Entity: SUBJID (subject)                                   │
│                                                             │
│  ┌─ Filters ──────────────────────────────────────────────┐ │
│  │  sex          [==] [Female        ▾]        [x]        │ │
│  │  AND                                                   │ │
│  │  der_hx_diabetes [==] [1          ▾]        [x]        │ │
│  │                                                        │ │
│  │  [+ Add Filter]                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Joins ────────────────────────────────────────────────┐ │
│  │  + analysis.PHQ9A_SCORES (via SUBJID → USUBJID)       │ │
│  │    score      [>]  [10           ]          [x]        │ │
│  │                                                        │ │
│  │  [+ Join Another Table]                                │ │
│  │  Available: ASCVD, ENRDT, MTQC (from structural_links)│ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ──────────────────────────────────────────────────────────  │
│  Matching: ~189 of 2,502 subjects    [Live SQL ▾]          │
│                                                             │
│  [Validate & Count]  [Preview Results]  [Save Cohort]      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  AI Suggestions (3)                          [Refresh ↻]   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ CV Risk      │ │ Female       │ │ Longitudinal │       │
│  │ Cohort       │ │ Smokers      │ │ Multi-Visit  │       │
│  │ est. 1,204   │ │ est. 389     │ │ est. 1,891   │       │
│  │ [Use] [More] │ │ [Use] [More] │ │ [Use] [More] │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Filter column selection**:
- Cohort dimension columns shown first (highlighted, with value set dropdowns)
- Other columns available via search (free-form input instead of dropdown)
- Column metadata shown on hover: definition, data type, null%, distinct count

**Value input**:
- Categorical columns (from `value_set_binding`): multi-select dropdown with value counts
- Numeric columns: text input with min/max range from tech profile
- Date columns: date picker with range support
- Boolean columns: toggle switch

**Join selection**:
- Only tables from `structural_links` are offered
- Grouped by `link_type`: entity_key first, then foreign_key, then shared_dimension
- After selecting a joined table, its cohort dimensions become available as filters

**Live count**:
- After each filter change, debounced (500ms) call to `/api/cohorts/validate`
- Shows estimated count: "~189 of 2,502 subjects"
- Warnings shown inline (e.g., "PHQ-9 is self-reported")

**Effort**: Large
**Files**: New `frontend/src/components/CohortBuilder.tsx`, `CohortFilterRow.tsx`, `CohortJoinPanel.tsx`

---

### CB-7: Cohort Results & Preview

After execution, show cohort results in a table with summary insights.

**Results view**:
```
┌─────────────────────────────────────────────────────────────┐
│  Cohort: Diabetic Females with Elevated PHQ-9               │
│  189 subjects | 2 tables | Executed 2026-04-24 10:30        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─ Insights ──────────────────────────────────────────── ┐ │
│  │  This cohort captures 7.6% of the DIAGNOSES population │ │
│  │  (189 of 2,502 subjects).                              │ │
│  │                                                        │ │
│  │  Key characteristics:                                  │ │
│  │  - Mean PHQ-9 score: 14.2 (moderate-severe depression) │ │
│  │  - 72% have comorbid hypertension                      │ │
│  │  - Most common visits: Baseline (189), Month 3 (162)   │ │
│  │                                                        │ │
│  │  Compared to the full population:                      │ │
│  │  - 2.1x higher rate of der_hx_cvd (cardiovascular)    │ │
│  │  - Mean age 6.3 years older                            │ │
│  │  - PHQ-9 scores 4.8 points higher on average           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ SQL ──────────────────────────────────────────────────┐ │
│  │  SELECT d.SUBJID, d.sex, d.der_hx_diabetes, ...       │ │
│  │  FROM `...DIAGNOSES` d                                 │ │
│  │  JOIN `...PHQ9A_SCORES` s ON d.SUBJID = s.USUBJID     │ │
│  │  WHERE d.sex = 'Female' AND d.der_hx_diabetes = 1     │ │
│  │    AND s.score > 10                                    │ │
│  │                                  [Copy SQL] [Edit SQL] │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ Results (first 500 of 189 subjects) ──────────────────┐ │
│  │  SUBJID  | sex    | der_hx_diabetes | score | VISIT    │ │
│  │  S-0012  | Female | 1               | 15    | Baseline │ │
│  │  S-0012  | Female | 1               | 12    | Month 3  │ │
│  │  S-0045  | Female | 1               | 18    | Baseline │ │
│  │  ...                                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  [Save Cohort] [Export CSV] [Query This Cohort] [New Query] │
└─────────────────────────────────────────────────────────────┘
```

**Insights generation**:
- After execution, backend runs summary queries:
  - COUNT(DISTINCT entity_anchor) for cohort size
  - GROUP BY each cohort_dimension for breakdowns
  - Comparison to full base table (same dimensions, no filters)
- Sends summary stats to Gemini for natural language insight generation
- Highlights: over/under-representation vs base, clinically notable patterns

**Effort**: Medium
**Files**: New `frontend/src/components/CohortResults.tsx`, `backend/cohort_engine.py`

---

### CB-8: Cohort Persistence & Library

Saved cohorts are stored in GCS and browsable from the catalog.

**Storage** (`gs://{bucket}/cohorts/{project_id}/`):
```
cohorts/
  {project_id}/
    _cohort_index.json          # list of all saved cohorts (summary)
    {cohort_id}/
      definition.json           # full CohortDefinition
      last_result.json          # cached execution results (optional)
```

**Cohort index** (`_cohort_index.json`):
```json
{
  "cohorts": [
    {
      "cohort_id": "c-uuid-1234",
      "name": "Diabetic Females with Elevated PHQ-9",
      "description": "Female subjects with der_hx_diabetes=1 and PHQ-9 > 10",
      "base_table": "wb-beamish-acorn-6393.analysis.DIAGNOSES",
      "tables_used": ["analysis.DIAGNOSES", "analysis.PHQ9A_SCORES"],
      "filter_count": 3,
      "last_result_count": 189,
      "created_at": "2026-04-24T10:30:00Z",
      "modified_at": "2026-04-24T10:30:00Z"
    }
  ]
}
```

**Cohort library UI** (new page or section in CatalogPage):
```
┌─────────────────────────────────────────────────────────────┐
│  Saved Cohorts (4)                          [+ New Cohort]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌ Diabetic Females w/ Elevated PHQ-9 ──────────────────┐  │
│  │ 189 subjects | DIAGNOSES + PHQ9A_SCORES | Apr 24      │  │
│  │ 3 filters: sex=Female, der_hx_diabetes=1, score>10   │  │
│  │ [Open] [Re-execute] [Delete]                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌ CV Risk Cohort ──────────────────────────────────────┐  │
│  │ 1,204 subjects | DIAGNOSES + ASCVD | Apr 24          │  │
│  │ 3 filters: der_hx_cvd=1 OR der_hx_cad=1 OR ...      │  │
│  │ [Open] [Re-execute] [Delete]                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Effort**: Medium
**Files**: `backend/cohort_storage.py`, new `frontend/src/pages/CohortsPage.tsx`, `frontend/src/components/CohortCard.tsx`

---

### CB-9: Query on Cohort

After a cohort is defined, users can run additional queries scoped to that cohort.

**How it works**:

1. User opens a saved cohort and clicks [Query This Cohort]
2. Opens a chat-like interface scoped to the cohort
3. Agent system prompt includes the cohort definition and its SQL
4. User asks: "What's the average PHQ-9 score by visit for this cohort?"
5. Agent generates SQL using the cohort as a CTE:

```sql
WITH cohort AS (
  SELECT d.SUBJID
  FROM `...DIAGNOSES` d
  JOIN `...PHQ9A_SCORES` s ON d.SUBJID = s.USUBJID
  WHERE d.sex = 'Female' AND d.der_hx_diabetes = 1 AND s.score > 10
)
SELECT s.VISIT, AVG(s.score) AS avg_phq9, COUNT(DISTINCT s.USUBJID) AS n
FROM `...PHQ9A_SCORES` s
WHERE s.USUBJID IN (SELECT SUBJID FROM cohort)
GROUP BY s.VISIT
ORDER BY s.VISIT
```

**Implementation**: Extend the existing agent with cohort-aware context:
- Add cohort SQL as a CTE prefix the agent can reference
- Add "cohort" tool: `get_cohort_members()` → returns the cohort CTE SQL
- Agent prompt: "The user has an active cohort. Use the `cohort` CTE to scope all queries to this cohort's subjects."

**Effort**: Medium
**Files**: `packages/verily-chat/src/verily_chat/agent.py`, `backend/chat_handler.py`

---

### CB-10: Cohort Page & Routing

**New routes**:
```
/cohorts                    → CohortsPage (library of saved cohorts)
/cohorts/:cohortId          → CohortDetailPage (view/edit/query a cohort)
```

**Integration with existing pages**:
- **TablePage**: New "Cohort Builder" tab (6th tab) between Semantic and Key Insights
- **CatalogPage**: "Cohorts (N)" link in sidebar or header
- **ChatPanel**: New mode `"cohort"` alongside Q&A and Agent

**Effort**: Small
**Files**: `frontend/src/App.tsx`, new `frontend/src/pages/CohortsPage.tsx`, `frontend/src/pages/CohortDetailPage.tsx`

---

## Priority & Execution Order

| Priority | Feature | What It Delivers | Effort |
|----------|---------|------------------|--------|
| **CB-1** | Cohort Definition Model | Data structures (backend + frontend) | Small |
| **CB-2** | Cohort SQL Generator | Filter → SQL translation | Medium |
| **CB-3** | Cohort API Endpoints | Backend CRUD + validate/execute | Medium |
| **CB-4** | NL → Cohort (AI) | "Find diabetic females..." → structured cohort | Medium |
| **CB-5** | AI-Suggested Cohorts | Proactive suggestions with rationale | Medium |
| **CB-6** | Filter-Based Builder UI | Visual filter builder with value sets | Large |
| **CB-7** | Cohort Results & Preview | Results table + AI insights | Medium |
| **CB-8** | Cohort Persistence & Library | Save/list/delete cohorts in GCS | Medium |
| **CB-9** | Query on Cohort | Chat/agent scoped to a cohort | Medium |
| **CB-10** | Cohort Page & Routing | Navigation and page structure | Small |

**Suggested order**:
```
Phase 1 (foundation):  CB-1 → CB-2 → CB-3 → CB-10
Phase 2 (AI modes):    CB-4 → CB-5
Phase 3 (builder UI):  CB-6 → CB-7
Phase 4 (persistence): CB-8 → CB-9
```

---

## Dependencies on Other Features

```
P1-1 (concept bindings)  ──→ CB-4 (NL uses concept bindings to find columns)
P1-2 (value set binding) ──→ CB-6 (filter builder shows value set dropdowns)
P1-3 (measurement method)──→ CB-7 (insights warn about comparability)
P1-4 (structural links)  ──→ CB-6 (join selection from structural links)
P3-1 (TS types)          ──→ CB-1 (cohort types extend profile types)
```

The cohort builder is most powerful after P1 ships — concept bindings enable cross-table concept matching, value set bindings populate filter dropdowns, structural links power the join picker, and measurement method enables comparability warnings.

Without P1, the cohort builder still works but degrades to:
- Column names instead of concept codes for NL matching
- Free-text filter values instead of value set dropdowns
- Free-text join_paths instead of typed structural links

---

## Key Design Decisions

### 1. Cohort = filter definition, not materialized table

Cohorts are stored as filter definitions (JSON), not as materialized BigQuery tables or views. The SQL is regenerated and executed on demand. This avoids:
- Storage cost for materialized cohorts
- Staleness when underlying data changes
- Permission issues with creating tables

### 2. Three interaction modes, one output format

NL, AI suggestions, and filter builder all produce the same `CohortDefinition` JSON. Users can start in any mode and switch: NL generates a draft → user refines in filter UI → saves.

### 3. Insights are AI-generated, not pre-computed

Cohort insights (key characteristics, comparisons to base) are generated by Gemini after execution, not pre-computed for all possible cohorts. This keeps the feature lightweight — no background jobs for cohort analytics.

### 4. GCS storage, not database

Cohort definitions stored in GCS alongside profiling output. Consistent with the app's existing storage pattern. No database dependency. Trade-off: no concurrent editing, no query optimization. Acceptable for a single-user or small-team tool.

### 5. Agent-powered query-on-cohort, not a custom query builder

Instead of building a second query UI for "query this cohort," we scope the existing chat agent to the cohort via CTE injection. This gives users the full power of NL-to-SQL without building another UI.
