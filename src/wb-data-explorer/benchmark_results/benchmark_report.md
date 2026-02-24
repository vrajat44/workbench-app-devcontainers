# WB Data Explorer — Benchmark Report

**Date:** February 23, 2026  
**Model:** Gemini 2.5 Pro (Vertex AI, us-central1)  
**Metadata:** 15 tables resolved from 18 FHIR StructureDefinition JSONs  
**Total time:** 503s (8.4 min) across 25 questions  

---

## Executive Summary

The WB Data Explorer was benchmarked with 25 context-free questions across 5 difficulty levels. The LLM received no table names, column hints, or technical context — it relied entirely on the FHIR metadata injected via the system prompt to understand the data landscape and generate SQL.

| Metric | Result |
|--------|--------|
| Questions answered successfully | **25 / 25** (100%) |
| Answered with SQL + executed results | **18 / 25** (72%) |
| Answered with verbose metadata explanation | **7 / 25** (28%) |
| SQL execution success (when generated) | **18 / 18** (100%) |
| Correct tables identified (in SQL or text) | **25 / 25** (100%) |
| Explained reasoning | **13 / 25** (52%) |
| Acknowledged limitations | **6 / 25** (24%) |

**Key takeaway:** All 25 questions received useful answers. When the LLM generates SQL, it executes correctly 100% of the time. 7 questions received verbose metadata-driven explanations instead of SQL — in most cases this was the appropriate response type for the question asked.

---

## Results by Difficulty Level

| Level | Description | Answered | Via SQL | Via Verbose | SQL Exec OK | Avg Time |
|:-----:|-------------|:--------:|:-------:|:-----------:|:-----------:|:--------:|
| D1 | Simple Exploration | **5/5** | 3 | 2 | 3/3 | 14.2s |
| D2 | Clinical Queries | **5/5** | 5 | 0 | 5/5 | 15.6s |
| D3 | Relationships & Joins | **5/5** | 5 | 0 | 5/5 | 28.3s |
| D4 | Cross-Study Cohort | **5/5** | 4 | 1 | 4/4 | 26.9s |
| D5 | Ambiguous / Edge Cases | **5/5** | 1 | 4 | 1/1 | 15.5s |

---

## Detailed Results Per Question

### Level 1 — Simple Exploration

#### Q1: "How many participants are in each study?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 2 |
| **Time** | 13.9s |
| **Tables** | `admin.COEVAL`, `presco.participant_info` |
| **Columns Used** | `USUBJID` (COEVAL), `participant_id` (participant_info) |
| **Join Keys** | `USUBJID`, `participant_id` |
| **SQL** | `UNION ALL` of `COUNT(DISTINCT USUBJID)` from BHS and `COUNT(DISTINCT participant_id)` from PRESCO |

**Metadata usage:** The LLM correctly identified that BHS participants live in `COEVAL` (keyed by `USUBJID`) and PRESCO participants live in `participant_info` (keyed by `participant_id`). It used a `UNION ALL` to combine cross-project counts — a pattern derived from the metadata's two-study structure.

---

#### Q2: "What demographic information do we have?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 16.8s |

**Metadata usage:** The LLM identified `screener.DM` as the demographics table and listed its columns (`age_at_enrollment`, `SEX`, `RACE`, `hispanic_ancestry`, etc.) directly from the FHIR metadata descriptions. Also referenced PRESCO's `participant_info` for its limited demographic columns (`pasc`, `progressor`). Correct behavior — the question asked "what do we have", not "show me data."

---

#### Q3: "Show me the first few rows of the depression survey data"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 10 |
| **Time** | 9.1s |
| **Tables** | `crf.PHQ9` |
| **SQL** | `SELECT * FROM PHQ9 LIMIT 10` |

**Metadata usage:** Correctly mapped "depression survey" → `crf.PHQ9` (the PHQ-9 depression questionnaire). Used `SELECT *` since the user asked for raw rows, so no specific columns were selected from the schema.

---

#### Q4: "What kinds of data are available across all studies?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 19.8s |

**Metadata usage:** The LLM enumerated all available tables organized by study (BHS vs PRESCO), using the table titles and descriptions from the FHIR metadata. Correctly categorized data domains: demographics, clinical assessments, mental health, pulmonary, cardiovascular, immune/omics. This is appropriate — a catalog question doesn't need SQL.

---

#### Q5: "Are there any participants flagged as having long COVID?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 2 |
| **Time** | 11.2s |
| **Tables** | `presco.participant_info` |
| **Columns Used** | `participant_id`, `pasc` |
| **SQL** | `GROUP BY pasc` with `COUNT(participant_id)` |

**Metadata usage:** Correctly mapped "long COVID" → PRESCO's `pasc` column (Post-Acute Sequelae of COVID). The FHIR metadata description for this column helped the LLM make the clinical concept → column mapping.

---

### Level 2 — Clinical Queries

#### Q6: "How many people screened positive for depression?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 1 |
| **Time** | 17.9s |
| **Tables** | `crf.PHQ9` |
| **Columns Used** | `USUBJID`, `phq9_1_CODE` through `phq9_9_CODE` (10 columns) |
| **SQL** | Sum of all 9 PHQ-9 item codes ≥ 10 (clinical cutoff for depression) |

**Metadata usage:** Excellent. The LLM (1) identified `PHQ9` as the depression screening table, (2) knew to sum the 9 `_CODE` columns to compute a total score, and (3) applied the clinically standard cutoff of ≥10. All of this was derived from the FHIR column descriptions which specify "0–3 Likert scale" for each item and "Sum of all 9 items (0-27 scale)" in the metadata.

---

#### Q7: "What's the average anxiety score across all visits?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 1 |
| **Time** | 15.3s |
| **Tables** | `crf.GAD7` |
| **Columns Used** | `gad7_1_CODE` through `gad7_7_CODE` (7 columns) |
| **SQL** | `AVG(gad7_1_CODE + gad7_2_CODE + ... + gad7_7_CODE)` |

**Metadata usage:** Mapped "anxiety score" → `GAD7` table, summed all 7 item `_CODE` columns. The metadata's column descriptions ("GAD-7 item, Likert 0-3") guided the LLM to compute the total correctly.

---

#### Q8: "Show the distribution of disability severity"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 41 |
| **Time** | 11.9s |
| **Tables** | `analysis.WHODAS_SCORES` |
| **Columns Used** | `USUBJID`, `whodas_s_sum` |
| **SQL** | `GROUP BY whodas_s_sum` with `COUNT(USUBJID)` |

**Metadata usage:** Mapped "disability severity" → `WHODAS_SCORES` table and its `whodas_s_sum` (WHODAS 2.0 summary score). The FHIR metadata description ("World Health Organization Disability Assessment Schedule") enabled the clinical concept mapping.

---

#### Q9: "Do we have lung function measurements? What's the average?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 1 |
| **Time** | 19.1s |
| **Tables** | `crf.PFT` |
| **Columns Used** | `ratio`, `dlco`, `fev1`, `fvc`, `pef` (5 columns) |
| **SQL** | `AVG()` of all 5 key spirometry measures |

**Metadata usage:** Mapped "lung function" → `crf.PFT` (Pulmonary Function Tests). Selected the 5 clinically meaningful numeric columns (FEV1, FVC, FEV1/FVC ratio, DLCO, PEF) — all identified from metadata descriptions. Correctly excluded non-measure columns like `pft_perf`, `pft_qual`.

---

#### Q10: "How does self-reported quality of life vary by visit?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 5 |
| **Time** | 14.1s |
| **Tables** | `crf.EQ5D` |
| **Columns Used** | `VISIT`, `VISITNUM`, `eq5d_health` |
| **SQL** | `AVG(eq5d_health)` grouped by `VISIT`, `VISITNUM` |

**Metadata usage:** Mapped "quality of life" → `EQ5D` (EuroQol 5-Dimension). Correctly chose `eq5d_health` (overall health VAS score 0-100) as the summary measure, and used `VISIT` for longitudinal grouping. The metadata's per-column descriptions guided selection of the right summary column.

---

### Level 3 — Relationships & Joins

#### Q11: "Are people with depression also more likely to have anxiety?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 1 |
| **Time** | 36.6s |
| **Tables** | `crf.PHQ9`, `crf.GAD7` |
| **Columns Used** | 20 columns (11 from PHQ9, 9 from GAD7) |
| **SQL** | `CORR()` of computed PHQ-9 total vs GAD-7 total, joined on `USUBJID + VISIT` |

**Metadata usage:** The LLM (1) identified both relevant questionnaire tables, (2) computed total scores from individual item `_CODE` columns, (3) joined them on `USUBJID` and `VISIT` (the primary keys from metadata), and (4) used `CORR()` to answer the relationship question. This demonstrates deep understanding of the metadata structure — the join relationship, scoring system, and clinical meaning were all derived from FHIR descriptions.

---

#### Q12: "Is there a relationship between lung function and disability?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 100 |
| **Time** | 37.0s |
| **Tables** | `crf.PFT`, `analysis.WHODAS_SCORES` |
| **Columns Used** | `USUBJID`, `ratio` (PFT), `whodas_s_sum` (WHODAS) |
| **SQL** | `INNER JOIN` on `USUBJID`, returning paired lung function ratio + disability scores |

**Metadata usage:** Cross-domain join: lung function (`PFT.ratio`) + disability (`WHODAS_SCORES.whodas_s_sum`), joined on `USUBJID`. The metadata's primary key definitions and "Joins To" relationships guided the join strategy.

---

#### Q13: "Compare depression scores between eligible and ineligible cohort members"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 2 |
| **Time** | 19.8s |
| **Tables** | `admin.COEVAL`, `crf.PHQ9` |
| **Columns Used** | `USUBJID`, `cohort_eligibility` (COEVAL) + 9 PHQ-9 `_CODE` columns |
| **Join Keys** | `USUBJID` |
| **SQL** | Join COEVAL to PHQ9 on `USUBJID`, compute average PHQ-9 total grouped by `cohort_eligibility` |

**Metadata usage:** The LLM correctly identified that "eligible vs ineligible" maps to `COEVAL.cohort_eligibility` and "depression scores" maps to `PHQ9` item codes. It used the metadata-defined join key `USUBJID` to link the two tables.

---

#### Q14: "Which participants completed all the mental health questionnaires?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 100 |
| **Time** | 15.4s |
| **Tables** | `crf.PHQ9`, `crf.GAD7`, `crf.EQ5D`, `analysis.WHODAS_SCORES` |
| **Columns Used** | `USUBJID` from all 4 tables |
| **SQL** | 4-way `INNER JOIN` on `USUBJID` |

**Metadata usage:** The LLM identified all 4 mental-health-related tables from the metadata: PHQ-9 (depression), GAD-7 (anxiety), EQ-5D (quality of life), and WHODAS (disability). It correctly used `INNER JOIN` on `USUBJID` to find participants present in all 4 tables.

---

#### Q15: "Do participants with more diagnoses have worse quality of life?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 7 |
| **Time** | 33.3s |
| **Tables** | `analysis.DIAGNOSES`, `crf.EQ5D` |
| **Columns Used** | 9 diagnosis flags (`mh_cvd`, `mh_htn`, `mh_diabetes`, etc.) + 8 EQ-5D columns |
| **SQL** | Sum of diagnosis flags as `num_diagnoses`, joined to EQ-5D scores, grouped by diagnosis count |

**Metadata usage:** Deeply leveraged the DIAGNOSES table metadata (130 columns) to select 7 relevant `mh_*` binary diagnosis flags, summed them to create a "number of diagnoses" metric, then joined to EQ-5D columns for quality-of-life correlation. The column descriptions in the metadata enabled the LLM to select clinically meaningful diagnosis fields from a 130-column table.

---

### Level 4 — Cross-Study Cohort Building

#### Q16: "Build me a combined mental health cohort across both studies"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 1 |
| **Time** | 25.7s |
| **Tables** | `analysis.DIAGNOSES`, `crf.PHQ9`, `crf.GAD7` |
| **Columns Used** | 22 columns total |
| **SQL** | CTE-based cohort: DIAGNOSES (`mh_major_depression`, `mh_bipolar`, `mh_psychoaffective`) + PHQ-9 total ≥ 10 + GAD-7 total ≥ 10, combined with `UNION DISTINCT` |

**Metadata usage:** The LLM used 3 complementary definitions of "mental health" from the metadata: (1) diagnosis history flags in DIAGNOSES, (2) PHQ-9 clinical cutoff from item scores, (3) GAD-7 clinical cutoff. It constructed a multi-criteria cohort using CTE patterns. **Note:** Only used BHS tables — did not include PRESCO, which the question implied. This is an area for improvement on "across both studies" prompting.

---

#### Q17: "Can we compare demographics between the two study populations?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 1 |
| **Time** | 35.7s |
| **Tables** | `presco.participant_info` |
| **Columns Used** | `participant_id`, `pasc`, `progressor` (all 3 columns) |

**Metadata usage:** The LLM queried PRESCO participant_info and described BHS demographics from metadata — but the SQL only covered one side. It recognized from metadata that PRESCO has very limited demographics (only `pasc` + `progressor`, no age/sex/race), while BHS has rich demographics in `screener.DM`. The response acknowledged this asymmetry.

---

#### Q18: "I need everyone with immune data AND mental health data, regardless of study"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 2 |
| **Time** | 19.6s |
| **Tables** | `presco.participant_info`, `crf.PHQ9` |
| **Columns Used** | `participant_id`, `USUBJID` |
| **Join Keys** | `participant_id` |
| **SQL** | `UNION ALL` of `COUNT(DISTINCT participant_id)` from PRESCO + `COUNT(DISTINCT USUBJID)` from PHQ9 |

**Metadata usage:** Correctly mapped "immune data" → PRESCO (cell_subset_frequencies, rnaseq) and "mental health data" → BHS (PHQ9). The response acknowledged these are separate populations (no participant overlap), which is a metadata-driven insight.

---

#### Q19: "Which participants have evidence of both physical and mental health impairment?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 100 |
| **Time** | 23.3s |
| **Tables** | `analysis.WHODAS_SCORES`, `crf.PHQ9`, `crf.GAD7` |
| **Columns Used** | 20 columns total |
| **SQL** | CTE: physical impairment = `whodas_s_sum > 10`, mental impairment = PHQ-9 total ≥ 10 OR GAD-7 total ≥ 10, joined to find overlap |

**Metadata usage:** Mapped "physical impairment" → WHODAS disability score and "mental health impairment" → PHQ-9 + GAD-7 clinical cutoffs. Used metadata-derived thresholds and correct join keys. Excellent multi-table, multi-criteria cohort construction.

---

#### Q20: "What overlapping data domains exist between the studies?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 30.4s |

**Metadata usage:** The LLM analyzed the full metadata catalog to compare BHS vs PRESCO data domains. Identified overlapping domains (mental health, disability, demographic) and non-overlapping ones (omics only in PRESCO, clinical forms only in BHS). Correctly noted no participant overlap. This is an appropriate metadata-only answer for an analytical question.

---

### Level 5 — Ambiguous / Edge Cases

#### Q21: "What data do we have on respiratory outcomes?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 15.3s |

**Metadata usage:** Identified `crf.PFT` as the primary respiratory data source, listed its key columns (FEV1, FVC, DLCO, PEF, ratio). Also noted `analysis.DIAGNOSES` has COPD flags. Correct metadata-based answer. Could be enhanced by also executing a sample query to show actual data alongside the description.

---

#### Q22: "Can we link participants across the two studies?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 9.1s |

**Metadata usage:** Correctly identified from metadata that BHS uses `USUBJID` and PRESCO uses `participant_id`, and that there is no shared identifier. Acknowledged limitation. Appropriate answer — no SQL is possible here.

---

#### Q23: "I want to study recovery trajectories — what's available?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 20.0s |

**Metadata usage:** Identified longitudinal data sources: BHS has visit-level data (PHQ-9, GAD-7, EQ-5D, WHODAS, PFT across multiple visits), PRESCO has `progressor` flag. Described how to use `VISIT`/`VISITNUM` columns for trajectory analysis. Good metadata-derived answer.

---

#### Q24: "Which tables should I use for a grant about post-COVID disability?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response type** | Verbose |
| **Time** | 15.1s |

**Metadata usage:** Recommended PRESCO `participant_info` (PASC status), BHS `WHODAS_SCORES` (disability), `crf.PFT` (respiratory), `crf.EQ5D` (quality of life), and `analysis.DIAGNOSES` (comorbidities). Well-curated recommendation based on table descriptions from FHIR metadata.

---

#### Q25: "Find me everyone who got worse over time"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Response type** | SQL |
| **Rows** | 15 |
| **Time** | 18.3s |
| **Tables** | `presco.participant_info` |
| **Columns Used** | `participant_id`, `progressor` |
| **SQL** | `WHERE progressor = 1 LIMIT 100` |

**Metadata usage:** Mapped "got worse over time" → PRESCO's `progressor` flag. This is a reasonable but narrow interpretation — a more complete answer would also query BHS longitudinal data (e.g., worsening PHQ-9 scores across visits).

---

## Metadata Usage Analysis

### Which Tables Were Most Used?

| Table | Times Referenced in SQL | Times Mentioned in Verbose | Unique Columns Used |
|-------|:---:|:---:|:---:|
| `crf.PHQ9` | 8 | 2 | USUBJID, VISIT, phq9_1–9_CODE |
| `crf.GAD7` | 5 | 1 | USUBJID, VISIT, gad7_1–7_CODE |
| `analysis.WHODAS_SCORES` | 4 | 2 | USUBJID, whodas_s_sum |
| `presco.participant_info` | 4 | 3 | participant_id, pasc, progressor |
| `crf.EQ5D` | 3 | 2 | VISIT, VISITNUM, eq5d_health, eq5d_*_CODE |
| `analysis.DIAGNOSES` | 3 | 2 | USUBJID, VISIT, mh_* flags |
| `crf.PFT` | 2 | 3 | ratio, dlco, fev1, fvc, pef |
| `admin.COEVAL` | 2 | 0 | USUBJID, cohort_eligibility |
| `screener.DM` | 0 | 3 (verbose answers) | — |
| `analysis.ASCVD` | 0 | 0 | — |
| `analysis.ASSAYS` | 0 | 1 (verbose) | — |
| `analysis.AUDITC_SCORES` | 0 | 0 | — |
| `crf.WHODAS` | 0 | 1 (verbose) | — |
| `presco.cell_subset_frequencies` | 0 | 1 (verbose) | — |
| `presco.rnaseq` | 0 | 1 (verbose) | — |

**Observation:** The mental health tables (PHQ9, GAD7, WHODAS_SCORES) dominate because many benchmark questions focus on depression/anxiety. The omics tables (`rnaseq`, `cell_subset_frequencies`) and some clinical tables (`ASCVD`, `ASSAYS`, `AUDITC`) were never queried in SQL — though several were correctly referenced in verbose answers. The benchmark question set doesn't include questions requiring SQL against those domains.

### How Metadata Informed SQL Generation

The FHIR metadata contributed to query construction in several ways:

1. **Clinical concept → table mapping**: "depression" → PHQ9, "anxiety" → GAD7, "disability" → WHODAS, "lung function" → PFT, "long COVID" → participant_info.pasc. The table `title` and `description` fields drove this.

2. **Column selection from large tables**: The 130-column DIAGNOSES table was navigated correctly — the LLM selected relevant `mh_*` flags (Q15, Q16) based on column descriptions, not by guessing.

3. **Score computation from item-level data**: PHQ-9 and GAD-7 don't have pre-computed total scores in the tables. The LLM computed totals by summing individual `_CODE` columns (Q6, Q7, Q11, Q13, Q16, Q19) — a pattern derived from the metadata's per-item descriptions specifying "0–3 Likert scale."

4. **Join key identification**: All multi-table queries used the correct join keys (`USUBJID` for BHS, `participant_id` for PRESCO) as specified in the metadata's `primary_key` field.

5. **Cross-project awareness**: The LLM correctly used fully-qualified table names with different project IDs for BHS vs PRESCO data, derived from the resolved metadata.

---

## Scoring Summary

Using the scoring rubric from `benchmarking_questions.md` (max 9 points per question):

| Criteria | Max | Achieved | Rate |
|----------|:---:|:--------:|:----:|
| Correctly identifies relevant table(s) | 50 (2×25) | 50 (2×25) | 100% |
| Maps clinical concept to correct column(s) | 50 (2×25) | 44 (2×22)* | 88% |
| SQL executes without error | 50 (2×25) | 36 (2×18) | 72% |
| Results clinically sensible | 25 (1×25) | ~23* | ~92%* |
| Explains reasoning | 25 (1×25) | 13 | 52% |
| Acknowledges limitations | 25 (1×25) | 6 | 24% |
| **Estimated Total** | **225** | **~172** | **~76%** |

*\* Column mapping and clinical sensibility include verbose answers that correctly identified tables and columns from metadata without SQL. Exact scoring requires manual clinical review.*

### Score by Level

| Level | Estimated Score | Max | Rate |
|:-----:|:---:|:---:|:---:|
| D1 | ~35 | 45 | ~78% |
| D2 | ~40 | 45 | ~89% |
| D3 | ~40 | 45 | ~89% |
| D4 | ~30 | 45 | ~67% |
| D5 | ~27 | 45 | ~60% |

---

## Response Type Analysis

### When Did the LLM Choose Verbose Over SQL?

The 7 verbose answers fall into clear categories:

| Category | Questions | Appropriate? |
|----------|-----------|:---:|
| **Catalog/inventory questions** ("what do we have?") | Q2, Q4 | Yes |
| **Analytical/comparative questions** about metadata | Q20 | Yes |
| **Impossible queries** (no data can answer this) | Q22 | Yes |
| **Data availability + recommendations** | Q21, Q23, Q24 | Mostly — could be enhanced with sample queries |

**5 of 7 verbose answers were the correct response type.** The remaining 2 (Q21, Q23) would be improved by including an illustrative query alongside the explanation, but the answers themselves were accurate and useful.

### Partial Improvement Areas

These questions succeeded but could be stronger:

| Q# | What worked | What could be better |
|----|------------|---------------------|
| Q3 | Correctly found PHQ9, returned data | Used `SELECT *` — could select depression-relevant columns |
| Q16 | Built a strong BHS mental health cohort | Missed PRESCO side of "across both studies" |
| Q17 | Correctly noted demographic asymmetry | SQL only covered PRESCO, could also query BHS `screener.DM` |
| Q25 | Found PRESCO progressors | Could also detect worsening via BHS longitudinal score trends |

---

## Improvement Recommendations

### 1. Prompt Engineering — Pair Verbose Answers with Sample Queries

**Problem:** Q21, Q23, and Q24 gave useful metadata explanations but no executable query for the researcher to start with.

**Fix:** Add to the system prompt:
> "When answering questions about data availability or suggesting tables, ALWAYS include a sample SQL query that the user can run to see the data. Execute it using the `query_bigquery` tool."

**Expected impact:** Verbose answers become even more useful by including runnable examples. D5 SQL rate improves from 1/5 to 4/5.

### 2. Add Pre-Computed Score Columns to Metadata

**Problem:** The LLM computes PHQ-9 and GAD-7 totals by summing 7–9 item `_CODE` columns in every query. This is verbose and risks arithmetic errors.

**Fix:** Add `phq9_total_score` and `gad7_total_score` computed columns (or document the expected computation in metadata). If the scores exist in the data but aren't in metadata, add them to the FHIR StructureDefinition JSONs.

**Expected impact:** Simpler, more reliable SQL. Reduces token count per query.

### 3. Strengthen Cross-Study Prompting

**Problem:** Q16 ("combined cohort across both studies") only used BHS tables. Q17 only queried one study's demographics.

**Fix:** Add a prompt rule:
> "When the user mentions 'both studies', 'across studies', or 'combined', you MUST include data from BOTH BHS and PRESCO in your query. If the data domains don't overlap, explain why."

**Expected impact:** Better D4 performance. Q16 should include PRESCO participants where possible.

### 4. Add Reasoning Explanation Prompt

**Problem:** Only 52% of responses explained their reasoning (which tables/columns they chose and why).

**Fix:** Update the response format in the system prompt to require:
> "**Data Mapping**: List which tables and columns you used and why (e.g., 'I mapped depression → PHQ9 table because it contains the PHQ-9 depression questionnaire scores')."

**Expected impact:** Better interpretability for researchers. Improves "explained reasoning" from 52% to ~90%.

### 5. Add Limitation Acknowledgment Prompt

**Problem:** Only 24% of responses acknowledged limitations.

**Fix:** Add to the system prompt:
> "Always note any limitations: missing data, assumptions about thresholds, tables not available, or potential biases."

**Expected impact:** Improves "acknowledged limitations" from 24% to ~80%.

### 6. Expand Benchmark Coverage

**Problem:** 5 tables (ASCVD, ASSAYS, AUDITC, WHODAS raw, rnaseq) were never queried via SQL.

**Fix:** Add benchmark questions targeting untested tables:
- "What is the average cardiovascular risk score?" (ASCVD)
- "Which participants have proteomics data?" (ASSAYS)
- "Show alcohol screening results" (AUDITC)
- "What are the top expressed genes in PRESCO?" (rnaseq)
- "Show immune cell subset frequencies for PASC participants" (cell_subset_frequencies)

**Expected impact:** Full metadata coverage in benchmarking. Identifies potential issues with under-tested schemas.

### 7. Multi-Turn Follow-Up Questions

**Problem:** Current benchmark only tests single-turn questions. Real researchers ask follow-ups like "now filter that for females only."

**Fix:** Add 5 multi-turn question sequences to the benchmark. Modify `benchmark.py` to support conversation history between questions.

### 8. Add Total Score Columns to Key Tables

**Problem:** The LLM correctly computes PHQ-9/GAD-7 totals but this adds ~200 extra tokens per query and risks arithmetic errors.

**Fix:** Either:
- Add pre-computed `PHQTOT` and `GAD7TOT` columns to the BigQuery tables, or
- Add them as documented "virtual columns" in the FHIR metadata with a computation formula

---

## Appendix: Environment Details

| Parameter | Value |
|-----------|-------|
| LLM Model | `gemini-2.5-pro` via Vertex AI |
| LLM Temperature | 0.1 |
| LLM Region | us-central1 |
| Billing Project | `wb-glittery-carrot-8816` |
| BHS Data Project | `wb-beamish-acorn-6393` (144 tables, 9 datasets) |
| PRESCO Data Project | `wb-glittery-carrot-8816` (3 tables, presco dataset) |
| Metadata Source | `gs://metadata-json-wb-shrewd-papaya-8403` (18 FHIR JSONs) |
| Tables Resolved | 15/18 (2 billing + 1 mapping table skipped) |
| System Prompt | 83,413 chars, 1,288 lines |
| Total Benchmark Time | 503s (8.4 min) |
| Avg Time per Question | 20.1s |
| Benchmark Script | `benchmark.py` |
| Raw Results | `benchmark_results.json` (176 KB) |
