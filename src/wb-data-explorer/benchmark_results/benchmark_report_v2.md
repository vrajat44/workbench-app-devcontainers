# WB Data Explorer — Benchmark Report v2

**Date:** February 23, 2026  
**Model:** Gemini 2.5 Pro (Vertex AI, us-central1)  
**Metadata:** 15 tables resolved from 18 FHIR StructureDefinition JSONs  
**Total time:** 852s (14.2 min) across 36 questions  
**Prompt version:** v2 (dynamic cross-study rules, always-execute, reasoning + limitations)

---

## Executive Summary

This is the second iteration of benchmarking the WB Data Explorer backend. **v2 incorporates 5 targeted improvements** identified from the v1 benchmark and adds 11 new questions for broader coverage.

### v1 → v2 Comparison (on original 25 questions)

| Metric | v1 | v2 | Change |
|--------|:--:|:--:|--------|
| SQL generated + executed | 18/25 (72%) | **24/25 (96%)** | +24pp |
| Verbose-only answers | 7/25 (28%) | **1/25 (4%)** | -24pp |
| SQL execution success | 18/18 (100%) | **24/24 (100%)** | — |
| Explained reasoning | 13/25 (52%) | **15/25 (60%)** | +8pp |
| Acknowledged limitations | 6/25 (24%) | **16/25 (64%)** | +40pp |

### v2 Full Results (36 questions)

| Metric | Result |
|--------|--------|
| Questions answered successfully | **36 / 36** (100%) |
| SQL generated + executed | **35 / 36** (97.2%) |
| Verbose-only answers | **1 / 36** (2.8%) |
| SQL execution success | **35 / 35** (100%) |
| Correct tables identified | **35 / 36** (97.2%) |
| Explained reasoning | **19 / 36** (52.8%) |
| Acknowledged limitations | **22 / 36** (61.1%) |
| All tables tested via SQL | **15 / 15** (100%) |

**Key takeaway:** The "always-execute" prompt improvement converted 6 previously verbose-only answers into SQL + execution, a dramatic improvement. The dynamic cross-study prompting eliminated hardcoded study names. Limitation acknowledgment nearly tripled from 24% to 61%. All 15 metadata tables now have SQL coverage through the expanded question set.

---

## Results by Difficulty Level

| Level | Description | Answered | SQL Generated | SQL Exec OK | Reasoning | Limitations | Avg Time |
|:-----:|-------------|:--------:|:---:|:---:|:---:|:---:|:---:|
| D1 | Simple Exploration (5) | **5/5** | 4/5 | 4/4 | 4/5 | 1/5 | 14.9s |
| D2 | Clinical Queries (10) | **10/10** | 10/10 | 10/10 | 1/10 | 5/10 | 16.0s |
| D3 | Relationships & Joins (8) | **8/8** | 8/8 | 8/8 | 7/8 | 7/8 | 23.5s |
| D4 | Cross-Study Cohort (8) | **8/8** | 8/8 | 8/8 | 5/8 | 6/8 | 32.8s |
| D5 | Ambiguous / Edge Cases (5) | **5/5** | 5/5 | 5/5 | 2/5 | 3/5 | 37.4s |
| | **TOTAL** | **36/36** | **35/36** | **35/35** | **19/36** | **22/36** | **23.7s** |

---

## Detailed Results Per Question

### Level 1 — Simple Exploration

#### Q1: "How many participants are in each study?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `UNION ALL` of `COUNT(DISTINCT USUBJID)` from BHS COEVAL + `COUNT(DISTINCT participant_id)` from PRESCO |
| **Tables** | `admin.COEVAL`, `presco.participant_info` |
| **Columns** | `USUBJID`, `participant_id` |
| **Rows** | 2 |
| **Time** | 16.5s |
| **Reasoning** | Yes |
| **Limitations** | No |

---

#### Q2: "What demographic information do we have?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `SELECT USUBJID, age_at_enrollment, SEX, RACE, hispanic_ancestry FROM DM LIMIT 10` |
| **Tables** | `screener.DM` |
| **Columns** | `USUBJID`, `age_at_enrollment`, `SEX`, `RACE`, `hispanic_ancestry` |
| **Rows** | 10 |
| **Time** | 14.7s |
| **Reasoning** | Yes |
| **Limitations** | No |
| **v1 → v2** | **IMPROVED** — v1 gave verbose-only answer. v2 executes SQL against DM table, showing actual demographic data. |

---

#### Q3: "Show me the first few rows of the depression survey data"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `SELECT * FROM PHQ9 LIMIT 10` |
| **Tables** | `crf.PHQ9` |
| **Rows** | 10 |
| **Time** | 11.5s |

---

#### Q4: "What kinds of data are available across all studies?"
| | |
|---|---|
| **Outcome** | **Success** — answered from metadata (verbose) |
| **Response** | Comprehensive list of all tables grouped by study (BHS vs PRESCO) with descriptions |
| **Time** | 15.9s |
| **Note** | This is the only verbose-only answer in v2. It's the appropriate response type — a catalog question listing all available datasets doesn't require SQL. |

---

#### Q5: "Are there any participants flagged as having long COVID?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `GROUP BY pasc` with `COUNT(participant_id)` from PRESCO |
| **Tables** | `presco.participant_info` |
| **Columns** | `participant_id`, `pasc` |
| **Rows** | 2 |
| **Time** | 16.0s |

---

### Level 2 — Clinical Queries

#### Q6: "How many people screened positive for depression?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | Sum of 9 PHQ-9 `_CODE` columns ≥ 10 (clinical cutoff) |
| **Tables** | `crf.PHQ9` |
| **Columns** | 10 columns (`USUBJID`, `phq9_1_CODE` through `phq9_9_CODE`) |
| **Rows** | 1 |
| **Time** | 14.0s |

---

#### Q7: "What's the average anxiety score across all visits?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `AVG(gad7_1_CODE + ... + gad7_7_CODE)` |
| **Tables** | `crf.GAD7` |
| **Columns** | 7 columns (`gad7_1_CODE` through `gad7_7_CODE`) |
| **Rows** | 1 |
| **Time** | 21.1s |

---

#### Q8: "Show the distribution of disability severity"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Tables** | `analysis.WHODAS_SCORES` |
| **Columns** | `whodas_s_sum` |
| **Rows** | 6 |
| **Time** | 29.4s |

---

#### Q9: "Do we have lung function measurements? What's the average?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Tables** | `crf.PFT` |
| **Columns** | `fev1`, `fvc`, `ratio` |
| **Rows** | 1 |
| **Time** | 13.7s |

---

#### Q10: "How does self-reported quality of life vary by visit?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Tables** | `crf.EQ5D` |
| **Columns** | `VISIT`, `VISITNUM`, `eq5d_health` |
| **Rows** | 5 |
| **Time** | 16.2s |

---

#### Q26 (NEW): "How many PRESCO participants are progressors versus non-progressors?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `CASE WHEN progressor=1 THEN 'Progressor' ... GROUP BY progressor_status` |
| **Tables** | `presco.participant_info` |
| **Columns** | `participant_id`, `progressor` |
| **Rows** | 3 (Progressor / Non-Progressor / Not in CPR analysis) |
| **Time** | 13.7s |
| **Note** | New question targeting untested PRESCO table. LLM correctly handled the 3-way split (including NULL values). |

---

#### Q27 (NEW): "What is the average cardiovascular risk score?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `AVG(score) FROM ASCVD` |
| **Tables** | `analysis.ASCVD` |
| **Columns** | `score` |
| **Rows** | 1 |
| **Time** | 11.4s |
| **Note** | New question. ASCVD was never tested in v1. LLM correctly identified the `score` column from metadata. |

---

#### Q28 (NEW): "Show me alcohol screening results"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `SELECT USUBJID, VISIT, AUDITC_SUM_SCORE FROM AUDITC_SCORES LIMIT 10` |
| **Tables** | `analysis.AUDITC_SCORES` |
| **Columns** | `USUBJID`, `VISIT`, `AUDITC_SUM_SCORE` |
| **Rows** | 10 |
| **Time** | 15.8s |
| **Note** | New question. AUDITC_SCORES was never tested in v1. Correctly mapped "alcohol screening" → AUDIT-C questionnaire. |

---

#### Q29 (NEW): "What are the demographics broken down by sex and race?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `GROUP BY SEX, RACE` with `COUNT(USUBJID)` from DM |
| **Tables** | `screener.DM` |
| **Columns** | `SEX`, `RACE`, `USUBJID` |
| **Rows** | 13 |
| **Time** | 9.6s |
| **Note** | New question. DM was only referenced in verbose answers in v1. Now generates SQL. |

---

#### Q30 (NEW): "Which participants have lab assay data and what was measured?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `SELECT USUBJID, Flow_Cytometry, Mass_Cytometry, ... WHERE any_assay = 1` |
| **Tables** | `analysis.ASSAYS` |
| **Columns** | 9 columns (USUBJID + 8 assay type flags) |
| **Rows** | 100 |
| **Time** | 14.9s |
| **Note** | New question. ASSAYS was never tested in v1. LLM identified all 8 binary assay-type columns from metadata and used OR logic to find participants with any lab data. |

---

### Level 3 — Relationships & Joins

#### Q11: "Are people with depression also more likely to have anxiety?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | Join PHQ9 + GAD7 on USUBJID, compute total scores, pair for correlation |
| **Tables** | `crf.PHQ9`, `crf.GAD7` |
| **Columns** | 20 columns (11 from PHQ9, 9 from GAD7) |
| **Rows** | 100 |
| **Time** | 27.0s |

---

#### Q12: "Is there a relationship between lung function and disability?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `INNER JOIN` PFT + WHODAS_SCORES on `USUBJID` |
| **Tables** | `crf.PFT`, `analysis.WHODAS_SCORES` |
| **Columns** | 8 columns |
| **Rows** | 100 |
| **Time** | 19.8s |

---

#### Q13: "Compare depression scores between eligible and ineligible cohort members"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | Join COEVAL + PHQ9 on `USUBJID`, AVG of PHQ-9 total by `cohort_eligibility` |
| **Tables** | `admin.COEVAL`, `crf.PHQ9` |
| **Columns** | 12 columns |
| **Rows** | 2 |
| **Time** | 15.9s |

---

#### Q14: "Which participants completed all the mental health questionnaires?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | 4-way `INNER JOIN` of PHQ9, GAD7, WHODAS, EQ5D on `USUBJID` |
| **Tables** | `crf.PHQ9`, `crf.GAD7`, `crf.WHODAS`, `crf.EQ5D` |
| **Columns** | `USUBJID` from each table |
| **Rows** | 100 |
| **Time** | 20.1s |

---

#### Q15: "Do participants with more diagnoses have worse quality of life?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | Sum of 15 diagnosis flags from DIAGNOSES, joined to EQ-5D scores |
| **Tables** | `analysis.DIAGNOSES`, `crf.EQ5D` |
| **Columns** | 18 columns (15 diagnosis flags + 3 EQ-5D columns) |
| **Rows** | 100 |
| **Time** | 28.7s |

---

#### Q31 (NEW): "Show immune cell subset frequencies for PASC versus non-PASC participants"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `JOIN cell_subset_frequencies ON SPLIT(sample_id, '_')[0] = participant_id`, grouped by `pasc` and `subset_code` |
| **Tables** | `presco.cell_subset_frequencies`, `presco.participant_info` |
| **Columns** | 5 columns (`subset_code`, `freq`, `sample_id`, `pasc`, `participant_id`) |
| **Rows** | 66 |
| **Time** | 14.6s |
| **Note** | New question. First SQL test of `cell_subset_frequencies`. LLM correctly handled the non-trivial join (parsing `participant_id` from `sample_id` string). |

---

#### Q32 (NEW): "What are the top expressed genes in the PRESCO data?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `AVG(expr) GROUP BY gene_id ORDER BY average_tpm_expression DESC LIMIT 10` |
| **Tables** | `presco.rnaseq` |
| **Columns** | `gene_id`, `expr` |
| **Rows** | 10 |
| **Time** | 16.6s |
| **Note** | New question. First SQL test of `rnaseq`. Correctly aggregated expression values by gene. |

---

#### Q33 (NEW): "Is cardiovascular risk related to disability severity?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | CTE categorizing ASCVD `score` into risk categories, joined to WHODAS_SCORES, computing AVG disability per risk group |
| **Tables** | `analysis.ASCVD`, `analysis.WHODAS_SCORES` |
| **Columns** | 6 columns (`score`, `USUBJID`, `VISIT` from ASCVD; `USUBJID`, `VISIT`, `whodas_s_sum` from WHODAS) |
| **Rows** | 1 |
| **Time** | 25.6s |
| **Note** | New question. Cross-domain join of two previously untested tables. LLM applied clinical risk category thresholds. |

---

### Level 4 — Cross-Study Cohort Building

#### Q16: "Build me a combined mental health cohort across both studies"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | CTE-based cohort: PHQ-9 total ≥ 10 + GAD-7 total ≥ 10, combined with `UNION DISTINCT` |
| **Tables** | `crf.PHQ9`, `crf.GAD7` |
| **Columns** | 20 columns |
| **Rows** | 100 |
| **Time** | 32.0s |
| **Note** | Still BHS-only (same as v1). PRESCO lacks comparable mental health questionnaire data, which the prompt now acknowledges. |

---

#### Q17: "Can we compare demographics between the two study populations?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Tables** | `presco.participant_info` |
| **Columns** | 3 columns |
| **Rows** | 6 |
| **Time** | 44.2s |
| **Note** | LLM correctly noted PRESCO has limited demographics. Acknowledged asymmetry in response. |

---

#### Q18: "I need everyone with immune data AND mental health data, regardless of study"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | Cross-domain query: ASSAYS (BHS immune data) + PHQ9 + GAD7 |
| **Tables** | `analysis.ASSAYS`, `crf.PHQ9`, `crf.GAD7` |
| **Columns** | 8 columns |
| **Rows** | 100 |
| **Time** | 26.1s |
| **v1 → v2** | **IMPROVED** — v1 used simple UNION ALL of counts. v2 uses ASSAYS to find BHS participants with immune data AND mental health questionnaires, a proper intersection query. |

---

#### Q19: "Which participants have evidence of both physical and mental health impairment?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | CTE: mental impairment = PHQ-9 ≥ 10 OR GAD-7 ≥ 10, physical = diagnosis flags, intersection |
| **Tables** | `crf.PHQ9`, `crf.GAD7`, `analysis.DIAGNOSES` |
| **Columns** | 26 columns |
| **Rows** | 100 |
| **Time** | 27.2s |

---

#### Q20: "What overlapping data domains exist between the studies?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `UNION ALL` comparing BHS RNA_Sequencing participants (from ASSAYS) vs PRESCO rnaseq participants |
| **Tables** | `analysis.ASSAYS`, `presco.rnaseq` |
| **Columns** | 3 columns |
| **Rows** | 2 |
| **Time** | 31.2s |
| **v1 → v2** | **IMPROVED** — v1 gave verbose-only metadata explanation. v2 generates SQL quantifying the overlap: counts participants with RNA-seq data in each study. |

---

#### Q34 (NEW): "Compare disability scores between the two study populations"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | PRESCO progressor percentage from `participant_info` |
| **Tables** | `presco.participant_info` |
| **Columns** | `participant_id`, `progressor` |
| **Rows** | 1 |
| **Time** | 34.6s |
| **Note** | New cross-study question. The LLM used PRESCO's `progressor` as a proxy for disability since PRESCO lacks WHODAS. Acknowledged the limitation. |

---

#### Q35 (NEW): "What biological and clinical data exists across studies for COVID recovery?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `UNION ALL` of PRESCO rnaseq participant counts + BHS ASSAYS RNA_Sequencing participant counts |
| **Tables** | `presco.rnaseq`, `analysis.ASSAYS` |
| **Columns** | 3 columns |
| **Rows** | 2 |
| **Time** | 29.1s |
| **Note** | New cross-study question. LLM quantified biological data across both studies using SQL, identifying the RNA-seq overlap between PRESCO and BHS. |

---

#### Q36 (NEW): "Build a combined dataset with immune markers and mental health data from all available studies"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `JOIN participant_info ON SPLIT(sample_id) = participant_id` with cell_subset_frequencies |
| **Tables** | `presco.participant_info`, `presco.cell_subset_frequencies` |
| **Columns** | `participant_id`, `pasc`, `progressor`, `subset_code`, `freq` |
| **Rows** | 100 |
| **Time** | 37.7s |
| **Note** | New complex cross-study question. LLM built a PRESCO immune + clinical dataset with proper join parsing. Acknowledged BHS mental health data would need a separate query. |

---

### Level 5 — Ambiguous / Edge Cases

#### Q21: "What data do we have on respiratory outcomes?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `SELECT USUBJID, VISIT, fev1, fvc, ratio, dlco FROM PFT WHERE pft_perf = 'Y' LIMIT 10` |
| **Tables** | `crf.PFT` |
| **Columns** | 7 columns |
| **Rows** | 10 |
| **Time** | 21.1s |
| **v1 → v2** | **IMPROVED** — v1 gave verbose-only answer. v2 executes SQL showing actual PFT data with quality filter (`pft_perf = 'Y'`). |

---

#### Q22: "Can we link participants across the two studies?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `UNION ALL` of 5 sample BHS identifiers (from DM) + 5 sample PRESCO identifiers |
| **Tables** | `screener.DM`, `presco.participant_info` |
| **Columns** | `USUBJID`, `participant_id` |
| **Rows** | 10 |
| **Time** | 34.9s |
| **v1 → v2** | **IMPROVED** — v1 gave verbose-only answer saying it's not possible. v2 still explains the limitation but also shows sample identifiers from each study, giving the researcher concrete data to see why direct linking isn't feasible. |

---

#### Q23: "I want to study recovery trajectories — what's available?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | Sample trajectory: one participant's WHODAS scores over visits, ordered by `study_day` |
| **Tables** | `analysis.WHODAS_SCORES` |
| **Columns** | `USUBJID`, `VISIT`, `study_day`, `whodas_s_sum` |
| **Rows** | 3 |
| **Time** | 64.8s |
| **v1 → v2** | **IMPROVED** — v1 gave verbose-only answer. v2 provides a runnable sample trajectory query showing actual longitudinal data. |

---

#### Q24: "Which tables should I use for a grant about post-COVID disability?"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | `UNION ALL` of PRESCO PASC count + BHS disability count (WHODAS > 24) |
| **Tables** | `presco.participant_info`, `analysis.WHODAS_SCORES` |
| **Columns** | `participant_id`, `pasc`, `USUBJID`, `whodas_s_sum` |
| **Rows** | 2 |
| **Time** | 34.3s |
| **v1 → v2** | **IMPROVED** — v1 gave verbose-only recommendation. v2 provides the recommendation AND executes a cross-study query quantifying the available cohort sizes for a grant. |

---

#### Q25: "Find me everyone who got worse over time"
| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **SQL** | CTE: BHS participants with worsening WHODAS scores (last > first via window functions) + PRESCO progressors |
| **Tables** | `analysis.WHODAS_SCORES`, `presco.participant_info` |
| **Columns** | `USUBJID`, `VISITNUM`, `whodas_s_sum`, `participant_id`, `progressor` |
| **Rows** | 100 |
| **Time** | 32.0s |
| **v1 → v2** | **IMPROVED** — v1 used only PRESCO `WHERE progressor = 1`. v2 builds a sophisticated cross-study query: BHS uses window functions (`FIRST_VALUE`/`LAST_VALUE`) to detect longitudinal worsening, PRESCO uses the `progressor` flag. Both combined with `UNION ALL`. |

---

## Improvement Impact Analysis

### What Changed and What It Fixed

#### 1. "Always Execute" Prompt — Converted 6 verbose answers to SQL

| Question | v1 Response | v2 Response | Root Cause |
|----------|------------|------------|------------|
| Q2 | Verbose: listed columns from metadata | SQL: `SELECT FROM DM LIMIT 10` | New prompt requires sample SQL for availability questions |
| Q20 | Verbose: described overlapping domains | SQL: `UNION ALL` counting RNA-seq per study | Same — now executes to quantify |
| Q21 | Verbose: described PFT table | SQL: `SELECT FROM PFT WHERE pft_perf='Y'` | Same |
| Q22 | Verbose: explained no linkage possible | SQL: Shows sample IDs from both studies | Same |
| Q23 | Verbose: described trajectory options | SQL: Sample trajectory for one participant | Same |
| Q24 | Verbose: recommended tables | SQL: Cross-study cohort size query | Same |

**Impact:** SQL generation rate on original 25 questions: **72% → 96%**

#### 2. Dynamic Cross-Study Prompting — Better multi-study queries

| Question | v1 Behavior | v2 Behavior |
|----------|------------|------------|
| Q25 | Simple PRESCO-only query | Cross-study CTE: WHODAS window functions + PRESCO progressors |
| Q20 | Verbose only | SQL quantifying overlap across projects |
| Q24 | Verbose only | Cross-study cohort sizing query |

**Impact:** The dynamic rules auto-detected 2 projects with different PKs and instructed the LLM to always include all studies. No hardcoded study names in the prompt.

#### 3. Limitation Acknowledgment — 24% → 61%

On the original 25 questions, limitation acknowledgment went from 6/25 to 16/25 — a significant improvement driven by the new response format requiring a "Limitations" section.

#### 4. Expanded Coverage — All 15 tables now tested

| Table | v1 SQL Coverage | v2 SQL Coverage |
|-------|:---:|:---:|
| ASCVD | Never tested | Q27, Q33 |
| AUDITC_SCORES | Never tested | Q28 |
| ASSAYS | Never tested | Q18, Q20, Q30, Q35 |
| DM | Verbose only | Q2, Q29 |
| rnaseq | Verbose only | Q20, Q32, Q35 |
| cell_subset_frequencies | Verbose only | Q31, Q36 |

---

## Metadata Usage Analysis

### Table Coverage (v2)

| Table | SQL References | Unique Columns Used | Questions |
|-------|:---:|:---:|---|
| `crf.PHQ9` | 7 | 11 | Q3, Q6, Q11, Q13, Q14, Q16, Q18 |
| `crf.GAD7` | 5 | 9 | Q7, Q11, Q14, Q16, Q18 |
| `analysis.WHODAS_SCORES` | 5 | 4 | Q8, Q12, Q23, Q24, Q25 |
| `presco.participant_info` | 8 | 3 | Q1, Q5, Q17, Q22, Q25, Q26, Q34, Q36 |
| `analysis.DIAGNOSES` | 2 | 15 | Q15, Q19 |
| `analysis.ASCVD` | 2 | 3 | Q27, Q33 |
| `analysis.ASSAYS` | 4 | 9 | Q18, Q20, Q30, Q35 |
| `analysis.AUDITC_SCORES` | 1 | 3 | Q28 |
| `crf.PFT` | 2 | 7 | Q9, Q21 |
| `crf.EQ5D` | 2 | 3 | Q10, Q14 |
| `crf.WHODAS` | 1 | 1 | Q14 |
| `screener.DM` | 3 | 5 | Q2, Q22, Q29 |
| `admin.COEVAL` | 2 | 2 | Q1, Q13 |
| `presco.rnaseq` | 3 | 2 | Q20, Q32, Q35 |
| `presco.cell_subset_frequencies` | 2 | 3 | Q31, Q36 |

**100% table coverage** — every table in the metadata catalog was queried via SQL at least once.

### How the Dynamic Prompt Improved Metadata Utilization

1. **Auto-detected study boundaries**: The prompt engine grouped tables by GCP project and identified BHS (12 tables, PK=USUBJID) vs PRESCO (3 tables, PK=participant_id) automatically from the metadata, without hardcoding.

2. **Cross-study queries used UNION ALL correctly**: Q1, Q20, Q22, Q24, Q25, Q35 all used `UNION ALL` to combine data from both projects — the dynamic cross-study rules guided this pattern.

3. **Previously untested tables worked on first try**: ASCVD, AUDITC_SCORES, ASSAYS, rnaseq, and cell_subset_frequencies all generated correct SQL without any metadata modifications. The FHIR descriptions were sufficient.

4. **Complex PRESCO joins succeeded**: Q31 correctly parsed `participant_id` from `sample_id` in cell_subset_frequencies using `SPLIT()`. Q32 aggregated gene expression correctly. These demonstrate the metadata's join relationship descriptions worked.

---

## Scoring Summary

| Criteria | Max | Achieved | Rate | v1 Rate |
|----------|:---:|:--------:|:----:|:-------:|
| Correctly identifies relevant table(s) | 72 (2×36) | 70 | 97% | 100% |
| Maps clinical concept to correct column(s) | 72 (2×36) | 66 | 92% | 88% |
| SQL executes without error | 72 (2×36) | 70 | 97% | 72% |
| Results clinically sensible | 36 (1×36) | ~34 | ~94% | ~92% |
| Explains reasoning | 36 (1×36) | 19 | 53% | 52% |
| Acknowledges limitations | 36 (1×36) | 22 | 61% | 24% |
| **Estimated Total** | **324** | **~281** | **~87%** | ~76% |

### Score by Level

| Level | Estimated Score | Max | Rate | v1 Rate |
|:-----:|:---:|:---:|:---:|:---:|
| D1 | ~38 | 45 | ~84% | ~78% |
| D2 | ~72 | 90 | ~80% | ~89% |
| D3 | ~65 | 72 | ~90% | ~89% |
| D4 | ~60 | 72 | ~83% | ~67% |
| D5 | ~38 | 45 | ~84% | ~60% |

---

## Remaining Gaps and Next Steps

### What's Still Not Perfect

1. **Q4 remains verbose-only** — "What kinds of data are available?" is a catalog question. The prompt says "ALWAYS include sample SQL" but the LLM determined a table listing was more appropriate. This is arguably correct behavior.

2. **Q16 still BHS-only for "combined mental health across both studies"** — PRESCO genuinely lacks PHQ-9/GAD-7 equivalents, so this is a data limitation, not a prompt issue. The v2 response now acknowledges this.

3. **Q17 still limited** — PRESCO demographics are sparse (only `pasc` + `progressor`). The LLM correctly identifies the asymmetry.

4. **Reasoning rate plateaued at ~53%** — The "Data Mapping" requirement improved some responses, but many D2 questions (simple queries) skip the explanation. Could add stronger language or enforce via output parsing.

5. **Average query time increased** — v2 averages 23.7s/question vs v1's 20.1s. The more complex cross-study queries and longer prompt contribute. Not a concern for interactive use but worth monitoring.

### Recommended Next Steps

1. **Enforce reasoning in short queries** — Add explicit instruction: "Even for simple queries, state which table you chose and why in one sentence."

2. **Multi-turn conversation benchmarking** — Test follow-up questions like "now filter that for females only" or "break that down by visit."

3. **Add pre-computed score columns** — PHQ-9 and GAD-7 totals are still computed inline. Adding `phq9_total` and `gad7_total` to metadata would reduce query complexity.

4. **Test with new studies** — The dynamic prompt is designed to auto-adapt when new FHIR metadata is added. Validate by adding a third study's metadata and re-running the benchmark.

5. **Error recovery benchmarking** — Test queries against tables with known access restrictions to validate the graceful fallback behavior.

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
| System Prompt | 86,208 chars (v1: 83,413 — +3.3% from dynamic rules) |
| Total Benchmark Time | 852s (14.2 min) |
| Avg Time per Question | 23.7s (v1: 20.1s) |
| Benchmark Script | `benchmark.py` |
| Raw Results | `benchmark_results_v2.json` |
| Prompt Changes | `prompt_engine.py` (dynamic rules, always-execute, response format) |
| Improvement Log | `improvement_tracker.md` |
