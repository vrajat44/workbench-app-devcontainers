# Test Cases: Cohort Building & Cross-Table Joins Features

**Scope**: 5-10 test cases per feature, designed to validate behavior across any dataset — not tied to a specific project or table structure.

**Conventions**:
- "Table A/B/C" = any profiled tables in the project
- "ID column" = any column identified as `entity_anchor`
- "Cohort column" = any column in `cohort_dimensions`
- Pass criteria are boolean assertions unless noted otherwise

---

## P0 — Fix Critical Blockers

### P0-1: Bulk profiler passes neighbor context

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Bulk-profile a dataset with 3+ tables that share a column name (e.g., a subject ID) | At least one table's semantic profile has non-empty `join_paths` for the shared column |
| 2 | Bulk-profile when `_catalog_context.md` already exists from a prior run | `neighbor_context` parameter passed to `profile_semantic()` is not None (log or assert) |
| 3 | Bulk-profile when `_catalog_context.md` does NOT exist (first-ever run) | Profiling completes without error; `neighbor_context` is None but no crash |
| 4 | Bulk-profile a dataset with 1 table only | Profiling completes; `join_paths` may be empty (no neighbors to reference) — no error |
| 5 | Bulk-profile with `--mode=semantic` (semantic-only, tech profiles pre-existing) | `neighbor_context` is loaded and passed; join_paths populated for shared columns |
| 6 | Bulk-profile with `--mode=both` (pipeline: tech then semantic) | After tech pass completes for all tables, semantic pass receives neighbor context |
| 7 | Verify catalog context is regenerated after bulk batch completes | `_catalog_context.md` in GCS has `profiled_at` timestamp after batch end time |

### P0-2: Two-pass semantic profiling

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | First-ever bulk profile of a 5-table dataset with shared ID column | After two-pass, ALL 5 tables have non-empty `join_paths` on the shared ID column (not just tables profiled last) |
| 2 | Table profiled first in batch (alphabetically or by order) has join_paths | The first table's semantic profile references at least one other table in `join_paths` |
| 3 | Table profiled last in batch has join_paths | The last table's semantic profile references at least one other table |
| 4 | Two-pass only re-profiles tables that need it | Tables that already have populated `join_paths` on entity_anchor columns are NOT re-profiled in the second pass |
| 5 | Two-pass with `--two-pass=false` (opt-out) | Only one pass runs; early tables may have empty join_paths (expected) |
| 6 | Two-pass on a dataset where no tables share columns | Second pass finds nothing to re-profile; completes quickly without LLM calls |
| 7 | After two-pass, catalog context includes join_paths from both passes | `_catalog_context.md` shows join_paths for tables profiled in pass 1 and pass 2 |
| 8 | Two-pass does not duplicate existing join_paths | If a table already had `join_paths = ["B.id"]` from pass 1, pass 2 does not produce `["B.id", "B.id"]` |

### P0-3: Execute and validate P0 test cases

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Test runner reads semantic profiles from GCS for all specified tables | All profiles loaded successfully; missing profiles reported as SKIP, not FAIL |
| 2 | Test runner validates entity_anchor is a real column in the table | If `entity_anchor = "SUBJID"`, column "SUBJID" exists in the tech profile |
| 3 | Test runner validates entity_type is from the allowed set | `entity_type` is one of: subject, participant, patient, encounter, observation, claim, specimen, sample, provider, organization, device, adverse_event, reference, metadata, or empty |
| 4 | Test runner validates cohort_dimensions excludes IDs and constants | No column in `cohort_dimensions` has `distinct_count = 1` (constant) or `distinct_count = row_count` (unique ID) |
| 5 | Test runner validates join_paths reference real tables | Every `join_paths` entry like "TABLE.COL" references a table that exists in the project |
| 6 | Test runner produces a summary report with pass/fail/skip counts | Report includes: total TCs, passed, failed, skipped, with details for each failure |
| 7 | Test runner works on any project/dataset, not just the test project | Runner accepts `--project`, `--dataset`, `--bucket` args and adapts assertions to the actual data |

---

## P1 — FHIR-Aligned Concept Bindings

### P1-1: Split terminology_bindings into Fixed Concept Binding + Code System Binding

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Column that always represents one concept (e.g., a score column named after a specific instrument) | `concept_binding` is populated with system + code + display; `code_system_binding` is None |
| 2 | Column that contains codes from a terminology (e.g., a column with ICD/CPT/LOINC codes as values) | `code_system_binding` is populated with system + display; `concept_binding` is None |
| 3 | Column with no terminology relevance (e.g., a free-text notes column) | Both `concept_binding` and `code_system_binding` are None/empty |
| 4 | Backward compatibility: `terminology_bindings` list is still populated | `terminology_bindings` contains entries matching `concept_binding` and/or `code_system_binding` |
| 5 | Concept binding system is a valid URI | `concept_binding.system` is one of the standard systems (LOINC, SNOMED, ICD-10, CPT, RxNorm, NDC) or `urn:verily:custom` |
| 6 | Code system binding system is a valid URI | Same validation as above |
| 7 | Two columns in different tables with the same fixed concept binding are matchable | If Table A col X and Table B col Y both have `concept_binding.code = "44261-6"` and same system, a search for that code returns both |
| 8 | Column with both a fixed concept AND codes (rare edge case: e.g., a score column where certain values map to severity codes) | Profiler picks the dominant pattern — either fixed or code system, not both |
| 9 | Re-profiling a table preserves concept/code system binding distinction | Profile table, modify nothing, re-profile — bindings remain consistent (not flipped between fixed/code system) |
| 10 | JSON serialization includes both new fields | `semantic_profile.json` contains `concept_binding` and `code_system_binding` at the column level |

### P1-2: Value Set Binding for cohort dimensions

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Binary column in cohort_dimensions (e.g., a 0/1 flag) | `value_set_binding` = `["0", "1"]` or equivalent meaningful labels |
| 2 | Low-cardinality categorical column (e.g., status with 3-5 values) | `value_set_binding` contains all distinct values from `top_values` |
| 3 | Column NOT in cohort_dimensions | `value_set_binding` is empty (not populated for non-cohort columns) |
| 4 | Sentinel values excluded (e.g., "99999", "N/A", "Unknown") | `value_set_binding` does not contain obvious sentinel/placeholder values |
| 5 | Null values excluded from value set | `value_set_binding` does not contain `null`, `None`, or empty string |
| 6 | High-cardinality column excluded from cohort_dimensions | Column with `distinct_count > 50` is NOT in `cohort_dimensions` and has no `value_set_binding` |
| 7 | Value set matches tech profile's top_values | Every value in `value_set_binding` appears in the tech profile's `top_values` or `value_counts` |
| 8 | Value set ordering follows frequency | Values in `value_set_binding` are ordered by frequency (most common first) or alphabetically — consistent ordering |
| 9 | JSON round-trip preserves value set | Write semantic profile to GCS, read back — `value_set_binding` is identical |
| 10 | Catalog context markdown includes value sets | `_catalog_context.md` shows value set for cohort dimension columns |

### P1-3: Measurement Method

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Survey/questionnaire score column | `measurement_method` = `"self-reported"` |
| 2 | Lab result column (e.g., blood glucose, hemoglobin) | `measurement_method` = `"lab-measured"` |
| 3 | Derived/calculated column (e.g., risk score, BMI) | `measurement_method` = `"calculated"` |
| 4 | Administrative/billing column (e.g., billing status, enrollment flag) | `measurement_method` = `"administrative"` |
| 5 | Device/sensor column (e.g., heart rate from wearable, step count) | `measurement_method` = `"device-collected"` |
| 6 | Non-measurement column (e.g., patient ID, date, free text) | `measurement_method` = `""` (empty) |
| 7 | Measurement method is from the allowed set | Value is one of: `self-reported`, `lab-measured`, `device-collected`, `calculated`, `administrative`, or empty |
| 8 | JSON serialization includes measurement_method | `semantic_profile.json` column entries include `measurement_method` field |
| 9 | Catalog context includes measurement method | `_catalog_context.md` column lines include measurement method when populated |
| 10 | Re-profiling produces consistent measurement method | Profile same table twice — measurement_method values are identical (LLM consistency) |

### P1-4: Structural Links (typed join_paths)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Two tables sharing an entity anchor column name | `structural_links` contains a link with `link_type = "entity_key"` between them |
| 2 | Two tables sharing a categorical dimension column (e.g., VISIT) | `structural_links` contains a link with `link_type = "shared_dimension"` |
| 3 | Structural link has valid cardinality | `cardinality` is one of: `one_to_one`, `one_to_many`, `many_to_one`, `many_to_many` |
| 4 | Structural link has confidence rating | `confidence` is one of: `high`, `medium`, `low` |
| 5 | Structural link target_table is fully qualified | `target_table` matches pattern `project.dataset.table` or `dataset.table` |
| 6 | Structural link target_column exists in the target table | The referenced column actually exists in the target table's profile |
| 7 | Backward compatibility: `join_paths` still populated | Column-level `join_paths` list is still populated alongside table-level `structural_links` |
| 8 | No self-referential links | No structural link has `target_table` = the table's own name |
| 9 | Entity_key links use entity_anchor columns | If `link_type = "entity_key"`, `source_column` matches `entity_anchor` of the source table or `target_column` matches `entity_anchor` of the target table |
| 10 | Cross-dataset links include dataset in target_table | If source is in dataset A and target is in dataset B, `target_table` includes dataset B's name |

---

## P2 — Chat & Agent Integration

### P2-1: Expose entity, cohort, concept bindings in chat prompt

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | System prompt includes entity_anchor for a profiled table | Prompt text contains "Entity:" or "entity_anchor:" followed by the column name |
| 2 | System prompt includes cohort_dimensions | Prompt text contains "Cohort dimensions:" followed by column names |
| 3 | System prompt includes concept_binding for columns that have one | Prompt text contains the LOINC/SNOMED/ICD code for fixed-bound columns |
| 4 | System prompt includes structural_links | Prompt text contains join information with link_type |
| 5 | System prompt includes measurement_method where populated | Prompt text contains "self-reported" / "lab-measured" etc. for relevant columns |
| 6 | Agent correctly uses entity_anchor for COUNT DISTINCT in a query | When asked "how many patients", agent uses `COUNT(DISTINCT entity_anchor_column)` |
| 7 | Agent uses cohort_dimensions for WHERE clauses when asked to filter | When asked "filter by sex", agent uses a cohort dimension column in WHERE |
| 8 | Prompt does not exceed token limit after adding new fields | System prompt with all new fields is under 30,000 tokens for a 50-table project |

### P2-2: `find_joinable_tables` agent tool

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Call with a table that has structural_links | Returns at least one joinable table with column mapping |
| 2 | Call with a table that has no structural_links | Returns empty result, no error |
| 3 | Results grouped by link_type | Output separates entity_key joins from shared_dimension joins |
| 4 | Bidirectional discovery | If Table A links to Table B, calling for Table B also returns Table A |
| 5 | Partial table name matching works | Calling with just "DIAGNOSES" finds "project.analysis.DIAGNOSES" |
| 6 | Cross-dataset joins included | If Table A (dataset X) links to Table B (dataset Y), both appear in results |
| 7 | Agent uses tool when user asks "what can I join to this table?" | Agent invokes `find_joinable_tables` tool, not `query_bigquery` |

### P2-3: `find_concept_across_tables` agent tool

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Search by concept code (e.g., "LOINC:44261-6") | Returns all columns across all tables with matching `concept_binding.code` |
| 2 | Search by display name (e.g., "PHQ-9") | Returns columns with display name containing the search term |
| 3 | Search by column name pattern (e.g., "phq9") | Returns columns whose name matches the pattern |
| 4 | No matches found | Returns empty result with helpful message, no error |
| 5 | Results include table name, column name, binding type, measurement method | Each result row has enough context for the agent to reason about cross-table matching |
| 6 | Code system bindings matched separately from fixed bindings | Searching for "ICD-10" returns columns with `code_system_binding.system` matching ICD-10, not columns with a single fixed ICD-10 concept |
| 7 | Agent uses tool when user asks "find all depression data" | Agent invokes `find_concept_across_tables` with a relevant concept term |

### P2-4: Cohort query builder in agent prompt

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Agent generates a multi-table JOIN query when user asks for a cross-table cohort | SQL uses JOIN with correct ON clause matching structural_links |
| 2 | Agent uses cohort_dimensions in WHERE clause | Filter columns in WHERE match `cohort_dimensions` from the semantic profile |
| 3 | Agent uses entity_anchor for subject-level deduplication | Query includes `COUNT(DISTINCT entity_anchor)` or `GROUP BY entity_anchor` |
| 4 | Agent warns about measurement_method differences | When joining two tables with different measurement_methods for the same concept, agent response includes a comparability note |
| 5 | Agent uses value_set_binding for valid filter values | WHERE clause uses values from `value_set_binding`, not hallucinated values |
| 6 | Agent generates correct cross-dataset JOIN syntax | Fully-qualified table names used: `` `project.dataset.table` `` |
| 7 | Query executes without BigQuery errors | Agent-generated SQL runs successfully via `query_bigquery` tool |

---

## P3 — Frontend & UI

### P3-1: Add new fields to TypeScript types

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | `SemProfile` interface includes `entity_anchor: string` | TypeScript compiles without error |
| 2 | `SemProfile` interface includes `entity_type: string` | TypeScript compiles |
| 3 | `SemProfile` interface includes `cohort_dimensions: string[]` | TypeScript compiles |
| 4 | `SemColumn` interface includes `concept_binding` | TypeScript compiles; type matches backend JSON shape |
| 5 | `SemColumn` interface includes `code_system_binding` | TypeScript compiles |
| 6 | `SemColumn` interface includes `measurement_method: string` | TypeScript compiles |
| 7 | `SemColumn` interface includes `value_set_binding: string[]` | TypeScript compiles |
| 8 | `SemProfile` interface includes `structural_links` array | TypeScript compiles; link type matches backend JSON shape |
| 9 | Frontend fetches and parses a real semantic profile without errors | GET `/api/.../profile/semantic` response deserialized into new types without runtime errors |
| 10 | Missing fields in older profiles default gracefully | Profile without `concept_binding` (pre-P1) renders without crash; field shows as empty |

### P3-2: Entity & concept metadata display

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Entity anchor badge visible on table detail page | Badge shows column name and entity type (e.g., "SUBJID (subject)") |
| 2 | Entity anchor is empty for a reference/admin table | No entity badge shown (not "Entity: (empty)") |
| 3 | Measurement method indicator visible on applicable columns | Columns with `measurement_method` show an indicator (e.g., "self-reported" tag) |
| 4 | Measurement method not shown for columns where it's empty | No empty badge or "(none)" text |
| 5 | Structural links section shows linked tables | Section lists target tables with link type badges |

### P3-3: Concept Binding display (per column)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Column with fixed concept binding shows badge | Badge displays system icon + code + display name (e.g., "LOINC 44261-6 — PHQ-9 Total Score") |
| 2 | Column with code system binding shows different badge | Badge displays system name with "Code System" indicator, visually distinct from fixed |
| 3 | Column with no binding shows no badge | Cell is empty, no placeholder text |
| 4 | Fixed and code system badges have different colors/icons | Visual distinction is immediately obvious without reading text |
| 5 | Clicking a standard terminology code opens external reference | Clicking LOINC code opens loinc.org (or similar), or shows a tooltip with the full URI |
| 6 | Custom bindings (urn:verily:custom) render without external link | Badge shows custom code and display, no broken external link |
| 7 | Multiple bindings on one column render cleanly | If a column has both (rare), both badges shown without overlap |

### P3-4: Value Set Binding display (per cohort dimension)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Cohort dimension column shows "Cohort Filter" tag | Tag visually distinguishes cohort columns from non-cohort columns |
| 2 | Clicking/hovering tag shows value set popover | Popover lists allowed values from `value_set_binding` |
| 3 | Value set with counts shows counts | If `value_counts` available from tech profile, display: "Male (1,247) \| Female (1,255)" |
| 4 | Value set with >15 values truncates | Shows top 15 with "... and N more" indicator |
| 5 | Empty value set (column in cohort_dimensions but no values) | Tag shown but popover says "No value set available" or similar |
| 6 | Non-cohort columns don't show the tag | Columns not in `cohort_dimensions` have no "Cohort Filter" tag |
| 7 | Binary columns show meaningful labels | For 0/1 columns, shows "0, 1" not empty |

### P3-5: Terminology & concept bindings summary panel

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Panel renders on a table with fixed concept bindings | "Fixed Concept Bindings" section lists bindings with their column mappings |
| 2 | Panel renders on a table with code system bindings | "Code System Bindings" section lists systems with their column mappings |
| 3 | Panel renders cohort dimensions with value sets | "Cohort Dimensions" section lists dimensions with their allowed values |
| 4 | Panel is collapsible | User can expand/collapse the panel |
| 5 | Table with no bindings shows empty state | Panel shows "No concept bindings found" or similar, not a blank section |
| 6 | Panel counts are accurate | "Cohort Dimensions (6)" matches actual count of cohort_dimensions |
| 7 | Panel renders without errors on a table with only tech profile (no semantic) | Panel shows "Semantic profile required" or is hidden entirely |

### P3-6: Join paths as clickable links

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Structural link renders as a clickable link | Click navigates to `/table/{project}/{dataset}/{table}` for the target table |
| 2 | Link shows target table name and column | Display text includes table and column: "ENRDT.SUBJID" |
| 3 | Link type badge shown alongside | "entity_key" / "foreign_key" / "shared_dimension" badge visible |
| 4 | Cross-dataset link includes dataset in display | Shows "dataset.TABLE.COLUMN" for cross-dataset links |
| 5 | Link to a table that doesn't exist (stale join_path) | Link renders but navigates to a 404 or shows "table not found" gracefully |

### P3-7: Join graph visualization

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Graph renders with 5+ tables and their connections | Nodes and edges visible, no overlap that obscures labels |
| 2 | Tables with no joins shown as disconnected nodes | Isolated nodes visible at graph periphery |
| 3 | Entity type determines node color | Different entity types (subject, specimen, reference) have different colors |
| 4 | Link type determines edge style | Entity_key, foreign_key, shared_dimension edges are visually distinct |
| 5 | Clicking a node navigates to table detail page | Click on "DIAGNOSES" node goes to its TablePage |
| 6 | Graph handles 50+ tables without performance degradation | Render time < 2 seconds; interaction (pan/zoom) is smooth |
| 7 | Graph handles a table with 0 structural_links | Table appears as isolated node, graph doesn't crash |
| 8 | Graph tooltip shows link details on edge hover | Hovering an edge shows source_column → target_column and cardinality |

---

## P4 — Profiling Quality & Scale

### P4-1: Structured join index (`_join_index.json`)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Join index generated after profiling completes | `_join_index.json` exists in GCS at `profiling/{project}/` |
| 2 | Every structural_link in every semantic profile appears in the index | Count of joins in index >= count of unique structural_links across all profiles |
| 3 | Join index entries are bidirectional | If A→B exists, B→A also exists (or a single entry with both directions) |
| 4 | Join index is valid JSON | Parses without error; matches expected schema |
| 5 | Join index regenerated after re-profiling a single table | Updated entries for the re-profiled table; other entries unchanged |
| 6 | Orphan detection: tables with no joins listed separately | Index includes a section or flag for tables with zero structural_links |
| 7 | Agent can load join index instead of scanning all profiles | `find_joinable_tables` tool uses `_join_index.json` and returns results in < 500ms |

### P4-2: Entity reconciliation across tables

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Tables with identical entity_anchor column names grouped | Tables with `entity_anchor = "SUBJID"` all appear in the same entity group |
| 2 | Tables with different names for same entity recognized | "SUBJID", "USUBJID", "participant_id" in the same entity group if they share data or join_paths |
| 3 | Entity registry includes canonical name and aliases | Registry entry has `canonical_name` and `aliases` list |
| 4 | Tables listed under each entity group | Each group includes the list of tables that use this entity |
| 5 | Reconciliation injected into profiling prompt | When re-profiling after reconciliation, the entity registry is included in the prompt |
| 6 | Non-entity columns not falsely reconciled | A "visit" column in Table A and a "visit" column in Table B are not treated as entity reconciliation (they're shared dimensions, not entity anchors) |
| 7 | Entity registry persisted to GCS | `_entity_registry.json` exists in GCS after reconciliation |
| 8 | Entity registry survives re-profiling | Re-profiling one table doesn't destroy the registry; it updates the relevant entries |

### P4-3: Smart neighbor context (prioritized)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Same-dataset tables get full detail in neighbor context | When profiling Table A in dataset X, other tables in dataset X appear with column-level detail |
| 2 | Cross-dataset tables with shared columns get summary | Tables in dataset Y that share column names with Table A appear with name + key columns |
| 3 | Unrelated tables get name-only listing | Tables with no column overlap appear as just a table name |
| 4 | Total context stays within budget (16KB) | `len(neighbor_context) <= 16384` characters |
| 5 | Context prioritization improves join_paths quality | Tables profiled with smart context produce more accurate join_paths than with truncated context |
| 6 | Project with 100+ tables doesn't crash or timeout | Neighbor context generation completes in < 5 seconds for large projects |
| 7 | Context includes entity_anchor and entity_type for neighbor tables | Neighbor summaries include entity info so LLM can match entity types |

### P4-4: Cross-dataset join format standardization

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Intra-dataset join uses `TABLE.COLUMN` format | `join_paths` entry for same-dataset join: "ENRDT.SUBJID" (no dataset prefix) |
| 2 | Cross-dataset join uses `DATASET.TABLE.COLUMN` format | `join_paths` entry for cross-dataset join: "analysis.ENRDT.SUBJID" |
| 3 | Post-processing validates format | If LLM produces "enrollment_table.id" (not a real table name), it's flagged or removed |
| 4 | Structural link target_table is always fully qualified | `structural_link.target_table` matches `project.dataset.table` pattern |
| 5 | No vague/generic table names in join_paths | No entries like "patient_table.id" or "other_table.subject_id" |

### P4-5: Concept-level catalog context

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Concept index generated alongside `_catalog_context.md` | `_concept_index.md` or concept section in catalog context exists after profiling |
| 2 | All fixed concept bindings appear in concept index | Every unique `concept_binding.code` across all tables has an entry |
| 3 | Concept index groups columns by concept | LOINC 44261-6 entry lists all columns from all tables with that binding |
| 4 | Concept index includes measurement method | Each column entry shows its measurement_method |
| 5 | Entity identifiers grouped in concept index | Subject ID columns (from entity_anchor) grouped under "Subject Identifier" |
| 6 | LLM uses concept index for cross-table matching | When profiling with concept index injected, join_paths accuracy improves for cross-dataset tables |
| 7 | Concept index handles project with no concept bindings | Empty but valid output, no error |

---

## P5 — AI-Assisted Cohort Builder

### CB-1: Cohort Definition Model

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | CohortDefinition serializes to valid JSON | `json.dumps(cohort.to_json_dict())` produces valid JSON |
| 2 | CohortDefinition deserializes from JSON | `CohortDefinition.from_dict(json_data)` reconstructs the object correctly |
| 3 | CohortFilter supports all operator types | Each operator (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `between`, `is_null`, `is_not_null`) serializes and deserializes |
| 4 | CohortJoin references a valid structural_link | `source_column` and `target_column` correspond to a known structural_link |
| 5 | Empty filters list is valid (select-all cohort) | CohortDefinition with `filters = []` is valid, represents "all rows" |
| 6 | Empty joins list is valid (single-table cohort) | CohortDefinition with `joins = []` is valid |
| 7 | Status transitions are valid | draft → validated → executed → saved; no skipping states |

### CB-2: Cohort SQL Generator

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Single-table cohort with one filter | SQL: `SELECT * FROM table WHERE col = 'value' LIMIT 500` |
| 2 | Single-table cohort with multiple AND filters | SQL: `WHERE col1 = 'v1' AND col2 > 10` |
| 3 | Multi-table cohort with one join | SQL includes `JOIN target ON source.col = target.col` |
| 4 | Multi-table cohort with join + filters on both tables | SQL has WHERE conditions on both base and joined table columns |
| 5 | Count mode produces COUNT DISTINCT on entity_anchor | SQL: `SELECT COUNT(DISTINCT entity_anchor) FROM ...` |
| 6 | Preview mode has LIMIT 500 | SQL ends with `LIMIT 500` |
| 7 | `in` operator produces correct SQL | `WHERE col IN ('a', 'b', 'c')` |
| 8 | `between` operator produces correct SQL | `WHERE col BETWEEN 10 AND 20` |
| 9 | `is_null` operator produces correct SQL | `WHERE col IS NULL` |
| 10 | Generated SQL executes without BigQuery syntax errors | SQL runs against BigQuery without errors for any valid CohortDefinition |

### CB-3: Cohort API Endpoints

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | POST `/api/cohorts/validate` with valid filters returns estimated count | Response includes `estimated_rows` > 0 and `valid = true` |
| 2 | POST `/api/cohorts/validate` with invalid column name returns error | Response includes `valid = false` and `errors` describing the invalid column |
| 3 | POST `/api/cohorts/execute` returns sample rows | Response includes `sample_rows` with at least 1 row (assuming data exists) |
| 4 | POST `/api/cohorts/execute` returns the generated SQL | Response includes `sql` field with the full query |
| 5 | GET `/api/cohorts` returns empty list when no cohorts saved | Response: `{"cohorts": []}` |
| 6 | POST `/api/cohorts/save` then GET `/api/cohorts` returns the saved cohort | Saved cohort appears in the list with correct name and filter count |
| 7 | DELETE `/api/cohorts/{id}` removes the cohort | Subsequent GET returns 404 |
| 8 | POST `/api/cohorts/validate` with no filters returns total row count | Estimated rows = total rows in the base table |

### CB-4: Natural Language → Cohort (AI Translation)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Simple NL: "female patients" | CohortDefinition with filter on sex/gender column using value from value_set_binding |
| 2 | NL with numeric threshold: "patients with score above 10" | CohortDefinition with filter: `operator = ">"`, `value = 10` on a score column |
| 3 | NL requiring a join: "patients with lab results above normal" | CohortDefinition with a CohortJoin to a lab/scores table |
| 4 | NL with concept reference: "all PHQ-9 data" | AI uses concept_binding to find the right columns (not just column name grep) |
| 5 | NL with ambiguous term: "diabetes" | AI correctly maps to the right column(s) — a diagnosis flag, not a free-text field |
| 6 | NL that can't be resolved: "patients who like chocolate" | Returns an error or empty cohort with explanation, not a hallucinated filter |
| 7 | NL result is a valid CohortDefinition | Output parses into CohortDefinition without validation errors |
| 8 | NL uses only columns that exist in the profiled tables | No hallucinated column names |

### CB-5: AI-Suggested Cohorts

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Suggestions returned for a table with 3+ cohort dimensions | At least 3 suggestions returned |
| 2 | Each suggestion has a name, description, and rationale | All three fields are non-empty strings |
| 3 | Each suggestion has an estimated size | `estimated_size > 0` |
| 4 | Suggestions use real column names and real values | Every filter column exists in the table; every value is from `value_set_binding` or `top_values` |
| 5 | At least one suggestion involves a cross-table join | If the table has structural_links, at least one suggestion uses a CohortJoin |
| 6 | Suggestions are diverse (not all using the same dimension) | At least 2 of the suggestions use different primary filter columns |
| 7 | Table with no cohort_dimensions produces no suggestions (or minimal) | Returns empty list or a message indicating no cohort dimensions available |
| 8 | Each suggestion includes analysis_opportunities | List of what analysis this cohort enables |

### CB-6: Filter-Based Cohort Builder UI

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Cohort dimension columns appear first in filter dropdown | Cohort dimension columns are prioritized/highlighted above non-cohort columns |
| 2 | Selecting a categorical cohort column shows value set dropdown | Dropdown populated from `value_set_binding` |
| 3 | Selecting a numeric column shows text input with min/max hint | Input shows placeholder from tech profile's `min_value` / `max_value` |
| 4 | Adding a filter updates the live count | Count changes after selecting a filter (debounced call to validate endpoint) |
| 5 | "Add join" button shows only tables from structural_links | Dropdown lists only tables with known joins, not all project tables |
| 6 | Adding a join shows the joined table's cohort dimensions as filters | New filter rows appear for the joined table's columns |
| 7 | Removing a filter updates the live count | Count increases when a restrictive filter is removed |
| 8 | Empty filter state shows total row count | "Matching: ~2,502 of 2,502 subjects" |
| 9 | Live SQL display updates with each filter change | SQL panel shows the current query reflecting all active filters |
| 10 | "Validate & Count" button triggers backend validation | POST to `/api/cohorts/validate` and displays estimated rows |

### CB-7: Cohort Results & Preview

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Results table shows columns from all joined tables | Both base table and joined table columns visible |
| 2 | Result count matches validated estimate (within tolerance) | `result_count` within 10% of `estimated_rows` from validation |
| 3 | AI insights generated after execution | Insights section shows natural language summary |
| 4 | Insights include cohort-vs-base comparison | At least one comparison metric (e.g., "2x higher rate of X") |
| 5 | SQL is visible and copyable | SQL section shows the executed query with a copy button |
| 6 | Results paginate for large cohorts | First 500 rows shown with indication of total count |
| 7 | Execution on an empty cohort (0 results) produces a useful message | Shows "No subjects match these criteria" instead of an empty table |

### CB-8: Cohort Persistence & Library

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Save a cohort to GCS | `cohorts/{project}/{cohort_id}/definition.json` exists in GCS |
| 2 | `_cohort_index.json` updated after save | Index includes the new cohort with name, base_table, filter_count |
| 3 | Load a saved cohort from GCS | `definition.json` deserializes into a valid CohortDefinition |
| 4 | Delete a cohort removes files from GCS | Both `definition.json` and index entry are removed |
| 5 | Cohort library page lists all saved cohorts | GET `/api/cohorts` returns all saved cohorts with summary metadata |
| 6 | Re-execute a saved cohort produces fresh results | Re-execution uses the saved filters against current data (not cached results) |
| 7 | Saving a cohort with the same name as an existing one creates a new entry | No overwrite; both cohorts exist with different IDs |

### CB-9: Query on Cohort

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Agent receives cohort CTE in system prompt | System prompt includes `WITH cohort AS (...)` SQL block |
| 2 | Agent scopes queries to the cohort | Agent-generated SQL uses `WHERE id IN (SELECT id FROM cohort)` or joins to the CTE |
| 3 | User asks "what's the average score for this cohort" | Agent returns correct average computed only over cohort members |
| 4 | User asks a question unrelated to the cohort | Agent still answers correctly (doesn't force cohort CTE onto every query) |
| 5 | Agent can reference cohort definition in natural language | Agent says "your cohort of 189 diabetic females" (not "the CTE") |
| 6 | Query on cohort with a joined table works | CTE includes the JOIN; subsequent queries can reference joined columns |
| 7 | Cohort CTE doesn't break on tables with reserved-word column names | Column names properly backtick-escaped in CTE SQL |

### CB-10: Cohort Page & Routing

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | `/cohorts` route renders the cohort library page | Page loads without error; shows list or empty state |
| 2 | `/cohorts/:cohortId` route renders cohort detail | Page loads the saved cohort and displays filters, SQL, results |
| 3 | "Cohort Builder" tab appears in TablePage | 6th tab visible between Semantic and Key Insights |
| 4 | Clicking "Cohort Builder" tab opens the filter UI | Filter builder renders with the current table's cohort dimensions |
| 5 | Navigation from cohort library to cohort detail works | Clicking a cohort card in the library navigates to `/cohorts/:id` |
| 6 | Back navigation from cohort detail returns to library | Browser back button returns to `/cohorts` |

---

## P6 — FHIR Export & Lineage

### P6-1: FHIR export (StructureDefinition generation)

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Export produces valid FHIR StructureDefinition JSON | Output validates against FHIR R4 StructureDefinition schema |
| 2 | `SD.title` matches `business_name` | Title field populated from semantic profile |
| 3 | `SD.purpose` matches `granularity` | Purpose field populated |
| 4 | `ext[verily-primary-identity]` matches `primary_key` | Extension contains PK column names |
| 5 | `ext[verily-structural-link]` matches `structural_links` | One extension per structural link |
| 6 | ElementDefinitions generated for each column | Count of EDs = count of columns in semantic profile |
| 7 | `ED.definition` matches column `definition` | Definition text matches |
| 8 | `ED.mapping[vfig]` matches `concept_binding` or `code_system_binding` | Mapping entries contain correct system + code |
| 9 | `ED.binding.valueSet` matches `value_set_binding` | Binding references a ValueSet with the correct codes |
| 10 | Export works for a table with no semantic profile | Returns error or empty output, no crash |

### P6-2: Data lineage integration

| # | Test Case | Pass Criteria |
|---|-----------|---------------|
| 1 | Lineage tracks source → derived table relationships | If Table B is derived from Table A, lineage shows A → B |
| 2 | Lineage validates join_paths against actual data flow | If a join_path references a table not in the lineage graph, it's flagged |
| 3 | Lineage handles multi-hop derivations | A → B → C lineage renders as a chain |
| 4 | Lineage handles tables with no known upstream | Table appears as a root node |
| 5 | Lineage persists to GCS | `_lineage.json` exists in GCS after lineage analysis |
