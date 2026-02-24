# WB Data Explorer — Improvement Tracker

This document captures the iterative improvements made to the WB Data Explorer's NLQ backend, driven by benchmarking results. Each iteration runs the full benchmark, analyzes gaps, implements targeted fixes, and re-benchmarks to measure impact.

---

## Iteration 1: Baseline Benchmark (v1)

**Date:** February 23, 2026  
**Benchmark:** 25 questions across 5 difficulty levels  
**Report:** `benchmark_report.md`

### Key Findings

| Metric | v1 Result |
|--------|-----------|
| Questions answered | 25/25 (100%) |
| SQL generated + executed | 18/25 (72%) |
| Verbose-only answers | 7/25 (28%) |
| SQL execution success rate | 18/18 (100%) |
| Explained reasoning | 13/25 (52%) |
| Acknowledged limitations | 6/25 (24%) |

### Gaps Identified

1. **Verbose answers lack runnable examples** — Q21, Q23, Q24 gave text-only answers with no SQL for the researcher to start with.
2. **Hardcoded study names in prompt** — Rules 3 and 10 referenced "BHS", "PRESCO", "DIAGNOSES" by name. In a real deployment, researchers don't know what studies exist, and adding new studies would require manual prompt edits.
3. **Low reasoning explanation rate (52%)** — Only half of responses explained why specific tables/columns were chosen.
4. **Low limitation acknowledgment (24%)** — Most responses didn't flag assumptions or missing data.
5. **Incomplete table coverage** — 5 tables (ASCVD, ASSAYS, AUDITC_SCORES, rnaseq, cell_subset_frequencies) were never queried via SQL. PRESCO was barely exercised.
6. **No graceful error handling** — If a query failed due to access/permissions, the system treated it as a hard failure instead of still providing the analysis.

---

## Iteration 2: Targeted Improvements (v2)

**Date:** February 23, 2026  
**Benchmark:** 36 questions (11 new) across 5 difficulty levels  
**Report:** `benchmark_report_v2.md`

### Changes Made

#### 1. Dynamic Cross-Study Prompting (`prompt_engine.py`)

**Problem:** The system prompt hardcoded study names ("BHS", "PRESCO") and specific join keys. A researcher wouldn't know these, and adding new studies required manual prompt edits.

**Solution:** Replaced hardcoded rules with three new functions that analyze loaded metadata at runtime:

- `_analyze_data_landscape(schemas)` — Groups tables by GCP project, extracts study names from table titles, identifies large tables and omics data
- `_build_join_rules(schemas, landscape)` — Generates Rule 3 (join keys) dynamically from primary keys and cross-project structure
- `_build_large_table_rules(landscape)` — Generates Rule 10 (large table awareness) from column counts

**Before (hardcoded):**
```
3. Join tables using the Primary Key:
   - For BHS tables: join on `USUBJID` (and `VISIT` when both tables have it)
   - For PRESCO tables: join on `participant_id`
   - For Billing tables: join on `athena_id`
```

**After (dynamic, auto-generated from metadata):**
```
3. Join tables using the Primary Key:
   - Tables sharing key `USUBJID` (12 tables): COEVAL, DM, PHQ9, GAD7, ...
     → Join on `USUBJID` (and `VISIT` when both tables have it)
   - Tables sharing key `participant_id` (3 tables): participant_info, cell_subset_frequencies, rnaseq
     → Join on `participant_id`

   CROSS-STUDY DATA — IMPORTANT:
   The data spans multiple independent studies:
   - BHS (project: wb-beamish-acorn-6393, 12 tables)
   - PRESCO (project: wb-glittery-carrot-8816, 3 tables)
   These use different participant identifiers...
```

**Impact:** The prompt now auto-adapts to any set of FHIR metadata. Adding a new study requires only adding its FHIR JSONs — no prompt edits needed.

#### 2. Always-Execute with Graceful Fallback (`prompt_engine.py`)

**Problem:** Verbose answers about data availability didn't include runnable SQL. Access errors were treated as failures.

**Solution:** Updated the system prompt with two new instructions:
- "When answering questions about data availability, ALWAYS include a sample SQL query and execute it"
- "If a query fails due to access/permissions, still present the SQL and explain the error — do NOT treat infrastructure errors as analysis failures"

**Impact:** Expected to convert 3–5 verbose-only answers to SQL + execution, and gracefully handle permission errors.

#### 3. Data Mapping Explanation in Response Format (`prompt_engine.py`)

**Problem:** Only 52% of responses explained their reasoning (which tables/columns were chosen and why).

**Solution:** Added a required "Data Mapping" step to the response format:
```
2. **Data Mapping**: Explain which tables and columns you chose and why
   (e.g., "I mapped 'depression' → PHQ9 table because it contains the PHQ-9 scores")
```

**Impact:** Expected to increase reasoning explanation from 52% to ~90%.

#### 4. Limitation Acknowledgment in Response Format (`prompt_engine.py`)

**Problem:** Only 24% of responses acknowledged limitations, assumptions, or data gaps.

**Solution:** Added a required "Limitations" step to the response format:
```
5. **Limitations**: Note any limitations — missing data, assumptions about clinical thresholds,
   tables not available, or potential biases in the data
```

Also added to IMPORTANT section: "Always note any limitations, assumptions, or caveats."

**Impact:** Expected to increase limitation acknowledgment from 24% to ~80%.

#### 5. Expanded Benchmark Questions (`benchmark.py`, `benchmarking_questions.md`)

**Problem:** 5 tables were never queried via SQL. PRESCO had minimal coverage. No questions targeted ASCVD, AUDITC, ASSAYS, DM, rnaseq, or cell_subset_frequencies.

**Solution:** Added 11 new questions (Q26–Q36):

| New Q# | Level | Target Table(s) | Question |
|--------|-------|-----------------|----------|
| Q26 | D2 | participant_info | How many PRESCO participants are progressors vs non-progressors? |
| Q27 | D2 | ASCVD | What is the average cardiovascular risk score? |
| Q28 | D2 | AUDITC_SCORES | Show me alcohol screening results |
| Q29 | D2 | DM | What are the demographics broken down by sex and race? |
| Q30 | D2 | ASSAYS | Which participants have lab assay data and what was measured? |
| Q31 | D3 | cell_subset + participant_info | Show immune cell subset frequencies for PASC vs non-PASC |
| Q32 | D3 | rnaseq | What are the top expressed genes in the PRESCO data? |
| Q33 | D3 | ASCVD + WHODAS_SCORES | Is cardiovascular risk related to disability severity? |
| Q34 | D4 | WHODAS cross-study | Compare disability scores between the two study populations |
| Q35 | D4 | Cross-study metadata | What biological and clinical data exists for COVID recovery? |
| Q36 | D4 | cell_subset + rnaseq + PHQ9 + GAD7 | Build combined dataset with immune + mental health data |

**Impact:** Full table coverage. Every table in the metadata catalog is now tested by at least one SQL question.

---

## Results Comparison

| Metric | v1 (25 Qs) | v2 (36 Qs) | v2 on same 25 | Change (same 25) |
|--------|:---:|:---:|:---:|--------|
| Questions answered | 25/25 (100%) | 36/36 (100%) | 25/25 | — |
| SQL generated + executed | 18/25 (72%) | 35/36 (97%) | 24/25 (96%) | **+24pp** |
| Verbose-only answers | 7/25 (28%) | 1/36 (3%) | 1/25 (4%) | **-24pp** |
| SQL execution success | 18/18 (100%) | 35/35 (100%) | 24/24 (100%) | — |
| Explained reasoning | 13/25 (52%) | 19/36 (53%) | 15/25 (60%) | +8pp |
| Acknowledged limitations | 6/25 (24%) | 22/36 (61%) | 16/25 (64%) | **+40pp** |
| Tables with SQL coverage | 10/15 (67%) | 15/15 (100%) | — | **+33pp** |
| Total benchmark time | 503s | 852s | — | +349s (more Qs) |
| Avg time per question | 20.1s | 23.7s | — | +3.6s |
| Estimated score | ~172/225 (76%) | ~281/324 (87%) | — | **+11pp** |

### Key Wins
1. **SQL generation rate**: 72% → 96% on same questions. The "always-execute" prompt converted 6 verbose-only answers to SQL.
2. **Limitation acknowledgment**: 24% → 64% on same questions. The required "Limitations" response section drove this.
3. **Full table coverage**: Every table in metadata is now tested via SQL (previously 5 tables were never queried).
4. **Cross-study queries improved**: Q25 went from simple PRESCO-only to sophisticated cross-study CTE with window functions.

---

## Future Iterations

### Iteration 3 (Planned)
- **Multi-turn follow-up questions** — Test conversational context (e.g., "now filter for females")
- **Pre-computed score columns** — Add PHQ-9/GAD-7 totals to metadata to simplify SQL
- **Error recovery benchmarking** — Deliberately test queries against tables with known access issues

### Iteration 4 (Planned)
- **Researcher persona testing** — Test with domain-specific jargon vs. layperson language
- **Performance optimization** — Reduce average query time through prompt compression
- **Confidence scoring** — Add self-assessment of answer quality to responses
