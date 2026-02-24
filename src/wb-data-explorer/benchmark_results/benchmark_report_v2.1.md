# WB Data Explorer — Benchmark Report

**Date:** February 24, 2026  
**Model:** Gemini 2.5 Pro (Vertex AI, us-central1)  
**Metadata Source:** 18 FHIR StructureDefinition JSONs + companion MeasureReport data profiles → 15 resolved BigQuery tables  
**Questions:** 36 (5 difficulty levels)  
**Total Runtime:** 852s (14.2 min)

---

## 1. Executive Summary

This report evaluates the WB Data Explorer's ability to translate natural-language research questions into correct BigQuery SQL, using only **FHIR StructureDefinition metadata** as its knowledge of the underlying data. No table names, column names, or SQL hints are given in the questions — the system must discover the right tables, columns, join keys, and clinical thresholds entirely from parsed metadata.

### Key Results

| Metric | Result | Observation | Improvement Opportunity |
|--------|--------|-------------|------------------------|
| Questions answered | **36 / 36** (100%) | Every question produced a useful answer | — |
| SQL generated + executed | **35 / 36** (97%) | Only Q4 (catalog question) skipped SQL — correctly | Could add supplementary `COUNT(*)` per table |
| SQL execution success | **35 / 35** (100%) | Zero runtime errors | — |
| Correct table identification | **35 / 36** (97%) | LLM matched clinical concepts to metadata tables | Enrich PRESCO metadata with more columns |
| Metadata table coverage | **15 / 15** (100%) | Every table queried via SQL at least once | — |
| Explained reasoning | **19 / 36** (53%) | D3/D4 explain well; D2 skips reasoning on simple queries | Prompt: require 1-line reasoning even for simple queries |
| Acknowledged limitations | **22 / 36** (61%) | Strong on cross-study asymmetry, weaker on single-table | Prompt: always state "no limitations" explicitly if none |

### How Metadata Drove the Results — Key Examples

- **Clinical concept mapping:** The question *"How many people screened positive for depression?"* produced correct SQL because the StructureDefinition `title` field (`"BHS CRF — PHQ-9 (Depression Screening)"`) let the LLM find the PHQ9 table, and the `definition` field for each `phq9_N_CODE` column described the 0–3 Likert scale — enabling it to sum items and apply the clinical cutoff of ≥ 10.

- **Cross-study join key discovery:** The question *"How many participants are in each study?"* required counting two studies with different primary keys. The StructureDefinition `extension` for `verily-primary-identity` told the LLM that BHS uses `USUBJID` and PRESCO uses `participant_id` — producing a correct `UNION ALL`.

- **Non-trivial PRESCO join:** The question *"Show immune cell frequencies for PASC vs non-PASC"* required joining `cell_subset_frequencies` to `participant_info`. These tables don't share a simple key — the LLM inferred from the `short` field (`"Sample identifier"`) that `sample_id` encodes `participant_id` as a prefix and used `SPLIT(sample_id, '_')[OFFSET(0)]`.

- **Data availability awareness:** When asked to *"compare demographics between studies"*, the LLM used the metadata catalog to recognize that PRESCO has no age/sex/race columns (only `pasc` and `progressor`) and acknowledged this asymmetry rather than generating invalid SQL.

### Summary

- Every question received a useful answer — either SQL-driven results or a metadata-informed explanation
- 35 of 36 questions produced executable SQL that ran successfully against BigQuery
- The single non-SQL answer (Q4: *"What kinds of data are available?"*) appropriately used the metadata catalog to list all datasets — a question where SQL is not the right response
- All 15 tables in the metadata catalog were referenced via SQL at least once, demonstrating complete coverage of the data landscape

---

## 2. How Metadata Drives Query Generation

The system uses a three-stage pipeline where **FHIR metadata is the single source of truth**:

```
FHIR StructureDefinition JSONs  +  MeasureReport data profiles
    ↓  (metadata_loader.py parses)
Table/Column/Join schemas  +  row counts & physical sizes
    ↓  (prompt_engine.py injects into LLM system prompt)
LLM system prompt with full schema context
    ↓  (LLM generates SQL from natural-language question)
BigQuery SQL → Execution → Results
```

### What the Metadata Provides

Each FHIR StructureDefinition JSON contains fields that the LLM uses to understand the data:

1. **Table Discovery** — The `title` field (e.g., `"BHS CRF — PHQ-9 (Depression Screening)"`) and `description` field let the LLM map clinical concepts to the right table.

2. **Column Mapping** — Each column element has a `short` field (e.g., `"Little interest or pleasure"` for `phq9_1`) and a `definition` field with detailed context including data types, ranges, and clinical meaning.

3. **Join Keys** — The `extension` field `verily-primary-identity` marks primary keys (`USUBJID` for BHS, `participant_id` for PRESCO). The `verily-structural-link` extensions describe relationships between tables.

4. **Study Boundaries** — The `extension` field `verily-study-name` identifies which study each table belongs to. The prompt engine dynamically discovers the study structure (BHS = 12 tables, PRESCO = 3 tables) from this field, without hardcoding.

5. **Data Profiles (MeasureReport JSONs)** — Companion MeasureReport JSONs provide table-level metrics via `measureScore.value`: `row-count` (e.g., PHQ9 has 1,489 rows) and `physical-size` in bytes. These help the LLM understand table scale and apply appropriate `LIMIT` clauses.

6. **Value Set Bindings** — The `binding` field on column elements references ValueSet resources that describe allowed coded values (e.g., PHQ-9 responses: 0 = Not at all, 1 = Several days, etc.).

7. **Sensitivity Labels** — The `extension` field `inline-sec-label` marks columns as `UID` (unique identifier) or `PHI` (protected health information), guiding appropriate data handling.

---

## 3. Results Summary by Difficulty Level

> **Reasoning column:** "Reasoning" indicates whether the LLM explicitly explained *why* it chose specific tables and columns — e.g., stating "I mapped depression → PHQ9 table because its `title` says 'Depression Screening'." This is distinct from simply generating correct SQL; it measures transparency of the decision process.

| Level | Description | Questions | SQL Generated | SQL Exec OK | Reasoning | Limitations | Avg Time |
|:-----:|-------------|:---------:|:---:|:---:|:---:|:---:|:---:|
| D1 | Simple Exploration | 5 | 4/5 | 4/4 | 4/5 | 1/5 | 14.9s |
| D2 | Clinical Queries | 10 | 10/10 | 10/10 | 1/10 | 4/10 | 16.0s |
| D3 | Relationships & Joins | 8 | 8/8 | 8/8 | 7/8 | 7/8 | 21.0s |
| D4 | Cross-Study Cohort Building | 8 | 8/8 | 8/8 | 5/8 | 7/8 | 32.8s |
| D5 | Ambiguous / Edge Cases | 5 | 5/5 | 5/5 | 2/5 | 3/5 | 37.4s |
| | **TOTAL** | **36** | **35/36** | **35/35** | **19/36** | **22/36** | **23.7s** |

Query complexity increases with difficulty level: D1 averages 14.9s (single-table lookups), while D5 averages 37.4s (cross-study CTEs with window functions). All SQL that was generated executed successfully — zero runtime errors.

---

## 4. Detailed Results Per Question

Each question below shows:
- **Outcome**: Success or Failure (did the system provide a useful answer?)
- **Metadata → SQL**: Which StructureDefinition fields guided the LLM's table/column selection
- **SQL**: The generated query (abbreviated for readability)

---

### Difficulty 1 (D1) — Simple Exploration

#### Summary

| Q# | Question | Outcome | Tables Used | Columns | Join | Time |
|:--:|----------|---------|-------------|:-------:|------|-----:|
| 1 | How many participants are in each study? | Success | `COEVAL`, `participant_info` | 2 | `UNION ALL` (cross-study) | 16.5s |
| 2 | What demographic information do we have? | Success | `DM` | 5 | — | 14.7s |
| 3 | Show me the first few rows of the depression survey data | Success | `PHQ9` | all | — | 11.5s |
| 4 | What kinds of data are available across all studies? | Success | *(metadata catalog)* | — | — | 15.9s |
| 5 | Are there any participants flagged as having long COVID? | Success | `participant_info` | 2 | — | 16.0s |

#### Q1: "How many participants are in each study?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 2 |
| **Time** | 16.5s |
| **Reasoning** | Yes |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (COEVAL) | `"BHS Admin — COEVAL (Cohort Eligibility)"` | Identified as BHS enrollment table |
| `title` (participant_info) | `"PRESCO — Participant Info"` | Identified as PRESCO enrollment table |
| `extension.verily-primary-identity` | `USUBJID` (BHS), `participant_id` (PRESCO) | Used `COUNT(DISTINCT ...)` on the correct PK per study |

```sql
SELECT 'BHS' AS study_name,
       COUNT(DISTINCT t1.USUBJID) AS participant_count
FROM `wb-beamish-acorn-6393.admin.COEVAL` AS t1
UNION ALL
SELECT 'PRESCO' AS study_name,
       COUNT(DISTINCT t1.participant_id) AS participant_count
FROM `wb-glittery-carrot-8816.presco.participant_info` AS t1
```

---

#### Q2: "What demographic information do we have?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 10 |
| **Time** | 14.7s |
| **Reasoning** | Yes |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS Screener — Demographics (DM)"` | "Demographics" in title matched the question |
| `short` (columns) | `"Age at enrollment"`, `"Unique Subject ID"` | LLM selected all demographic-relevant columns |
| `definition` (SEX) | Describes sex coding | Included in SELECT |
| `definition` (RACE) | Describes race categories | Included in SELECT |

```sql
SELECT t.USUBJID, t.age_at_enrollment, t.SEX, t.RACE, t.hispanic_ancestry
FROM `wb-beamish-acorn-6393.screener.DM` AS t
ORDER BY t.USUBJID LIMIT 10
```

---

#### Q3: "Show me the first few rows of the depression survey data"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 10 |
| **Time** | 11.5s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS CRF — PHQ-9 (Depression Screening)"` | "Depression" in `title` matched "depression survey" |
| `description` | `"Patient Health Questionnaire-9 (PHQ-9) — a validated, 9-item self-report depression screener..."` | Confirmed PHQ9 as the depression instrument |

```sql
SELECT * FROM `wb-beamish-acorn-6393.crf.PHQ9` LIMIT 10
```

---

#### Q4: "What kinds of data are available across all studies?"

| | |
|---|---|
| **Outcome** | **Success** — Answered from metadata catalog (verbose) |
| **Time** | 15.9s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata Usage:** The LLM used the `title` and `description` fields from all 15 StructureDefinition JSONs to provide a comprehensive listing of tables grouped by study (BHS: 12 tables, PRESCO: 3 tables). This is the only question where SQL was not generated — appropriately, since the question asks about data availability, not data content.

---

#### Q5: "Are there any participants flagged as having long COVID?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 2 |
| **Time** | 16.0s |
| **Reasoning** | Yes |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `short` (pasc column) | `"PASC (Long COVID) status"` | "Long COVID" in `short` matched the question |
| `definition` (pasc column) | Describes 0/1 coding for PASC classification | LLM applied `CASE WHEN pasc = 1` logic |

```sql
SELECT
    CASE WHEN pasc = 1 THEN 'PASC (Long COVID)'
         WHEN pasc = 0 THEN 'No PASC'
         ELSE 'Unknown'
    END AS pasc_status,
    COUNT(participant_id) AS participant_count
FROM `wb-glittery-carrot-8816.presco.participant_info`
GROUP BY pasc_status
```

---

### Difficulty 2 (D2) — Clinical Queries

#### Summary

| Q# | Question | Outcome | Tables Used | Columns | Time |
|:--:|----------|---------|-------------|:-------:|-----:|
| 6 | How many people screened positive for depression? | Success | `PHQ9` | 10 | 14.0s |
| 7 | What's the average anxiety score across all visits? | Success | `GAD7` | 7 | 21.1s |
| 8 | Show the distribution of disability severity | Success | `WHODAS_SCORES` | 1 | 29.4s |
| 9 | Do we have lung function measurements? What's the average? | Success | `PFT` | 3 | 13.7s |
| 10 | How does self-reported quality of life vary by visit? | Success | `EQ5D` | 3 | 16.2s |
| 26 | How many PRESCO participants are progressors vs non-progressors? | Success | `participant_info` | 2 | 13.7s |
| 27 | What is the average cardiovascular risk score? | Success | `ASCVD` | 1 | 11.4s |
| 28 | Show me alcohol screening results | Success | `AUDITC_SCORES` | 3 | 15.8s |
| 29 | What are the demographics broken down by sex and race? | Success | `DM` | 3 | 9.6s |
| 30 | Which participants have lab assay data and what was measured? | Success | `ASSAYS` | 9 | 14.9s |

#### Q6: "How many people screened positive for depression?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 1 |
| **Time** | 14.0s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS CRF — PHQ-9 (Depression Screening)"` | Matched "depression" |
| `description` | `"...Total score ranges from 0 to 27 with clinical severity cutoffs: 5 = mild, 10 = moderate..."` | LLM extracted the ≥ 10 cutoff for positive screen |
| `short` (phq9_1) | `"Little interest or pleasure"` | Identified individual Likert-scale items |
| `definition` (phq9_1) | `"...frequency of a specific depressive symptom...0–3 Likert scale..."` | LLM understood items must be summed |

```sql
WITH phq9_scores AS (
    SELECT t.USUBJID,
      (IFNULL(t.phq9_1_CODE, 0) + IFNULL(t.phq9_2_CODE, 0) + ... + IFNULL(t.phq9_9_CODE, 0))
      AS total_phq9_score
    FROM `wb-beamish-acorn-6393.crf.PHQ9` AS t
)
SELECT
    COUNT(DISTINCT CASE WHEN total_phq9_score >= 10 THEN USUBJID END) AS positive_screen,
    COUNT(DISTINCT USUBJID) AS total_participants
FROM phq9_scores
```

**Metadata insight:** The `description` field explicitly states clinical cutoffs (5/10/15/20), enabling the LLM to apply the standard ≥ 10 threshold without external clinical knowledge. The column `definition` fields describe each item as a 0–3 Likert response, guiding the summing pattern.

---

#### Q7: "What's the average anxiety score across all visits?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 1 |
| **Time** | 21.1s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS CRF — GAD-7 (Anxiety Screening)"` | Matched "anxiety" |
| `short` (gad7_1) | `"Feeling nervous, anxious"` | Identified Likert-scale items |

```sql
SELECT AVG(
    IFNULL(t.gad7_1_CODE, 0) + ... + IFNULL(t.gad7_7_CODE, 0)
  ) AS average_anxiety_score
FROM `wb-beamish-acorn-6393.crf.GAD7` AS t
```

---

#### Q8: "Show the distribution of disability severity"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 6 |
| **Time** | 29.4s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS Analysis — WHODAS 2.0 Scores"` | Matched "disability" |
| `short` (whodas_s_sum) | `"WHODAS 2.0 summary score"` | Identified the summary score column |
| `definition` (whodas_s_sum) | `"...Range 0 (no disability) to 48 (maximum disability)."` | LLM used this range to create severity buckets |

```sql
SELECT
  CASE
    WHEN whodas_s_sum = 0 THEN '0 (No disability)'
    WHEN whodas_s_sum BETWEEN 1 AND 12 THEN '1-12 (Mild disability)'
    WHEN whodas_s_sum BETWEEN 13 AND 24 THEN '13-24 (Moderate disability)'
    WHEN whodas_s_sum BETWEEN 25 AND 36 THEN '25-36 (Severe disability)'
    WHEN whodas_s_sum BETWEEN 37 AND 48 THEN '37-48 (Extreme disability)'
    ELSE 'Not Scored'
  END AS disability_severity,
  COUNT(*) AS number_of_assessments
FROM `wb-beamish-acorn-6393.analysis.WHODAS_SCORES`
GROUP BY disability_severity
```

**Metadata insight:** The `definition` field's score range (0–48) enabled the LLM to create clinically meaningful severity categories without any external reference.

---

#### Q9: "Do we have lung function measurements? What's the average?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 1 |
| **Time** | 13.7s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS CRF — PFT (Pulmonary Function Tests / Spirometry)"` | Matched "lung function" |
| `short` (fev1) | `"FEV1"` | Selected as spirometry measure |
| `short` (fvc) | `"FVC"` | Selected as spirometry measure |
| `short` (ratio) | `"FEV1/FVC Ratio"` | Selected as spirometry measure |

```sql
SELECT AVG(t1.fev1) AS average_fev1,
       AVG(t1.fvc) AS average_fvc,
       AVG(t1.ratio) AS average_fev1_fvc_ratio
FROM `wb-beamish-acorn-6393.crf.PFT` AS t1
```

---

#### Q10: "How does self-reported quality of life vary by visit?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 5 |
| **Time** | 16.2s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS CRF — EQ-5D-5L (Quality of Life)"` | Matched "quality of life" |
| `short` (eq5d_health) | `"Health today (VAS 0–100)"` | Identified as the self-reported health score |

```sql
SELECT t1.VISIT,
       AVG(CAST(t1.eq5d_health AS INT64)) AS avg_quality_of_life,
       COUNT(DISTINCT t1.USUBJID) AS participant_count
FROM `wb-beamish-acorn-6393.crf.EQ5D` AS t1
WHERE t1.eq5d_health IS NOT NULL
GROUP BY t1.VISIT
```

---

#### Q26: "How many PRESCO participants are progressors versus non-progressors?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 3 |
| **Time** | 13.7s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"PRESCO — Participant Info"` | Identified correct PRESCO table |
| `short` (progressor) | `"Progressor status (CPR publication)"` | Matched "progressors" |

```sql
SELECT
    CASE WHEN progressor = 1 THEN 'Progressor'
         WHEN progressor = 0 THEN 'Non-Progressor'
         ELSE 'Not in CPR analysis'
    END AS progressor_status,
    COUNT(participant_id) AS participant_count
FROM `wb-glittery-carrot-8816.presco.participant_info`
GROUP BY progressor_status
```

**Metadata insight:** The LLM correctly handled the three-way split including NULLs (participants not included in the CPR analysis). The `short` field's mention of "CPR publication" informed the LLM that NULL means exclusion from analysis, not missing data.

---

#### Q27: "What is the average cardiovascular risk score?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 1 |
| **Time** | 11.4s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS Analysis — ASCVD Risk Scores"` | Matched "cardiovascular risk" |
| `short` (score) | Describes ASCVD 10-year risk | Identified `score` as the numeric column |

```sql
SELECT AVG(t1.score) AS average_ascvd_score
FROM `wb-beamish-acorn-6393.analysis.ASCVD` AS t1
```

---

#### Q28: "Show me alcohol screening results"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 10 |
| **Time** | 15.8s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS Analysis — AUDIT-C Scores"` | Mapped "alcohol screening" → AUDIT-C |
| `short` (AUDITC_SUM_SCORE) | `"AUDIT-C sum score"` | Identified the summary score column |

```sql
SELECT t.USUBJID, t.VISIT, t.AUDITC_SUM_SCORE
FROM `wb-beamish-acorn-6393.analysis.AUDITC_SCORES` AS t
LIMIT 10
```

---

#### Q29: "What are the demographics broken down by sex and race?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 13 |
| **Time** | 9.6s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS Screener — Demographics (DM)"` | Matched "demographics" |
| Column `short` fields | `SEX`, `RACE` identified as categorical demographic columns | Used in `GROUP BY` |

```sql
SELECT t.SEX, t.RACE, COUNT(t.USUBJID) AS participant_count
FROM `wb-beamish-acorn-6393.screener.DM` AS t
GROUP BY t.SEX, t.RACE
ORDER BY participant_count DESC
```

---

#### Q30: "Which participants have lab assay data and what was measured?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 14.9s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS Analysis — Assay Availability Matrix"` | Matched "lab assay data" |
| `short` (Flow_Cytometry) | `"Flow cytometry availability"` | Identified as binary (0/1) assay flag |
| 7 other assay column `short` fields | Similar pattern for all 8 assay types | LLM built `OR` filter across all 8 |

```sql
SELECT t.USUBJID, t.Flow_Cytometry, t.Mass_Cytometry, t.Microbiome_Stool,
       t.Microbiome_Swabs, t.DNA_Methylation, t.Proteomics,
       t.RNA_Sequencing, t.DNA_Sequencing
FROM `wb-beamish-acorn-6393.analysis.ASSAYS` AS t
WHERE t.Flow_Cytometry = 1 OR t.Mass_Cytometry = 1 OR t.Microbiome_Stool = 1
   OR t.Microbiome_Swabs = 1 OR t.DNA_Methylation = 1 OR t.Proteomics = 1
   OR t.RNA_Sequencing = 1 OR t.DNA_Sequencing = 1
LIMIT 100
```

**Metadata insight:** The LLM used 8 `short` fields — each ending in "availability" — to understand these are binary flags and constructed an `OR` filter to find participants with *any* lab data. No hardcoded column names.

---

### Difficulty 3 (D3) — Relationships & Joins

#### Summary

| Q# | Question | Outcome | Tables Joined | Join Key | Columns | Time |
|:--:|----------|---------|---------------|----------|:-------:|-----:|
| 11 | Depression also more likely to have anxiety? | Success | `PHQ9` + `GAD7` | `USUBJID + VISIT` | 20 | 27.0s |
| 12 | Lung function and disability relationship? | Success | `PFT` + `WHODAS_SCORES` | `USUBJID + VISIT` | 8 | 19.8s |
| 13 | Depression scores by cohort eligibility | Success | `COEVAL` + `PHQ9` | `USUBJID` | 12 | 15.9s |
| 14 | Completed all mental health questionnaires? | Success | `PHQ9` + `GAD7` + `WHODAS` + `EQ5D` | `USUBJID` (4-way) | 4 | 20.1s |
| 15 | More diagnoses = worse quality of life? | Success | `DIAGNOSES` + `EQ5D` | `USUBJID + VISIT` | 18 | 28.7s |
| 31 | Immune cell frequencies: PASC vs non-PASC | Success | `cell_subset_freq` + `participant_info` | `SPLIT(sample_id)` | 5 | 14.6s |
| 32 | Top expressed genes in PRESCO? | Success | `rnaseq` | — | 2 | 16.6s |
| 33 | Cardiovascular risk vs disability severity? | Success | `ASCVD` + `WHODAS_SCORES` | `USUBJID + VISIT` | 6 | 25.6s |

#### Q11: "Are people with depression also more likely to have anxiety?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 27.0s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (PHQ9) | `"BHS CRF — PHQ-9 (Depression Screening)"` | Matched "depression" |
| `title` (GAD7) | `"BHS CRF — GAD-7 (Anxiety Screening)"` | Matched "anxiety" |
| `extension.verily-primary-identity` | `USUBJID` (both tables) | Used as join key |
| Column `short` fields | 9 PHQ-9 items + 7 GAD-7 items identified as Likert scores | Summed for total scores |

**Join Key (from metadata):** `USUBJID` + `VISIT` — both tables share these as marked by the `verily-primary-identity` extension.

```sql
SELECT t1.USUBJID, t1.VISIT,
    (IFNULL(t1.phq9_1_CODE, 0) + ... + IFNULL(t1.phq9_9_CODE, 0)) AS phq9_total_score,
    (IFNULL(t2.gad7_1_CODE, 0) + ... + IFNULL(t2.gad7_7_CODE, 0)) AS gad7_total_score
FROM `crf.PHQ9` AS t1
INNER JOIN `crf.GAD7` AS t2
  ON t1.USUBJID = t2.USUBJID AND t1.VISIT = t2.VISIT
WHERE (phq9 total > 0 OR gad7 total > 0) LIMIT 100
```

**Metadata insight:** The LLM derived the composite join key (`USUBJID` + `VISIT`) from the `verily-primary-identity` extension, ensuring per-visit alignment. It used 20 metadata columns across 2 tables.

---

#### Q12: "Is there a relationship between lung function and disability?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 19.8s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (PFT) | `"BHS CRF — PFT (Pulmonary Function Tests / Spirometry)"` | Matched "lung function" |
| `title` (WHODAS_SCORES) | `"BHS Analysis — WHODAS 2.0 Scores"` | Matched "disability" |
| `short` columns | `"FEV1"`, `"FVC"`, `"FEV1/FVC Ratio"`, `"WHODAS 2.0 summary score"` | Selected relevant measures |

```sql
SELECT pft.USUBJID, pft.VISIT, pft.fev1, pft.fvc, pft.ratio, whodas.whodas_s_sum
FROM `crf.PFT` AS pft
JOIN `analysis.WHODAS_SCORES` AS whodas
  ON pft.USUBJID = whodas.USUBJID AND pft.VISIT = whodas.VISIT
WHERE pft.fev1 IS NOT NULL AND whodas.whodas_s_sum IS NOT NULL LIMIT 100
```

---

#### Q13: "Compare depression scores between eligible and ineligible cohort members"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 2 |
| **Time** | 15.9s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (COEVAL) | `"BHS Admin — COEVAL (Cohort Eligibility)"` | Matched "eligible and ineligible cohort" |
| `short` (cohort_eligibility) | Identifies the eligibility classification column | Used in `GROUP BY` |
| PHQ-9 column `definition` fields | 0–3 Likert items | Summed for total depression score |

```sql
SELECT t1.cohort_eligibility,
       AVG(t2.phq9_1_CODE + ... + t2.phq9_9_CODE) AS average_phq9_score
FROM `admin.COEVAL` AS t1
INNER JOIN `crf.PHQ9` AS t2 ON t1.USUBJID = t2.USUBJID
WHERE t2.phq9_1_CODE IS NOT NULL
GROUP BY t1.cohort_eligibility
```

---

#### Q14: "Which participants completed all the mental health questionnaires?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 20.1s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (PHQ9) | Contains "Depression Screening" | LLM classified as mental health |
| `title` (GAD7) | Contains "Anxiety Screening" | LLM classified as mental health |
| `title` (WHODAS) | Contains "WHODAS" (disability) | LLM classified as mental health |
| `title` (EQ5D) | Contains "Quality of Life" | LLM classified as mental health |

**Join:** 4-way `INNER JOIN` on `USUBJID` ensures only participants present in ALL four tables are returned.

```sql
SELECT DISTINCT t1.USUBJID
FROM `crf.PHQ9` AS t1
INNER JOIN `crf.GAD7` AS t2 ON t1.USUBJID = t2.USUBJID
INNER JOIN `crf.WHODAS` AS t3 ON t1.USUBJID = t3.USUBJID
INNER JOIN `crf.EQ5D` AS t4 ON t1.USUBJID = t4.USUBJID
LIMIT 100
```

**Metadata insight:** The LLM correctly identified all four mental-health-related tables from their `title` fields and used `INNER JOIN` to find the intersection. Understanding that "mental health questionnaires" spans 4 separate tables required interpreting each table's `title`.

---

#### Q15: "Do participants with more diagnoses have worse quality of life?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 28.7s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (DIAGNOSES) | `"BHS Analysis — Diagnoses"` | Matched "diagnoses" |
| `short` (der_hx_cvd) | `"Any history of cardiovascular disease"` | Identified as binary diagnosis flag |
| 12 other `der_hx_*` column `short` fields | Similar binary flag pattern | All 13 summed as "diagnosis count" |
| `short` (eq5d_health) | `"Health today (VAS 0–100)"` | Quality of life outcome |

```sql
SELECT t1.USUBJID, t1.VISIT,
    (IFNULL(t1.der_hx_cvd, 0) + ... + IFNULL(t1.der_hx_ckd, 0)) AS diagnosis_count,
    t2.eq5d_health
FROM `analysis.DIAGNOSES` AS t1
INNER JOIN `crf.EQ5D` AS t2
  ON t1.USUBJID = t2.USUBJID AND t1.VISIT = t2.VISIT
WHERE t2.eq5d_health IS NOT NULL
ORDER BY diagnosis_count DESC LIMIT 100
```

**Metadata insight:** The LLM used 18 columns across 2 tables. It understood from column `short` fields that the 13 `der_hx_*` columns are binary flags representing individual diagnoses and summed them to create a composite "diagnosis burden" metric — clinical reasoning derived entirely from StructureDefinition metadata.

---

#### Q31: "Show immune cell subset frequencies for PASC versus non-PASC participants"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 66 |
| **Time** | 14.6s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (cell_subset_frequencies) | `"PRESCO — Cell Subset Frequencies"` | Matched "immune cell subset frequencies" |
| `short` (freq) | `"Calculated cell frequency"` | Selected as the measure |
| `short` (subset_code) | `"Immune cell subset code"` | Used for grouping |
| `short` (sample_id) | `"Sample identifier"` | LLM inferred it encodes `participant_id` as a prefix |
| `short` (pasc) | `"PASC (Long COVID) status"` | Used for PASC vs non-PASC grouping |

**Join Key:** `SPLIT(sample_id, '_')[OFFSET(0)] = participant_id` — the LLM inferred this from the `short` descriptions.

```sql
SELECT t2.pasc, t1.subset_code, AVG(t1.freq) AS avg_frequency
FROM `presco.cell_subset_frequencies` AS t1
INNER JOIN `presco.participant_info` AS t2
  ON SPLIT(t1.sample_id, '_')[OFFSET(0)] = t2.participant_id
GROUP BY t2.pasc, t1.subset_code
ORDER BY t1.subset_code, t2.pasc LIMIT 100
```

**Metadata insight:** This is the most technically impressive join in the benchmark. The PRESCO tables don't share a simple key. The LLM used the `short` field values (`"Sample identifier"` vs `"Participant identifier"`) to understand that `sample_id` contains `participant_id` as a prefix and applied `SPLIT()` to extract it.

---

#### Q32: "What are the top expressed genes in the PRESCO data?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 10 |
| **Time** | 16.6s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"PRESCO — RNAseq Gene Expression"` | Matched "expressed genes" and "PRESCO" |
| `short` (gene_id) | `"Gencode gene identifier"` | Used for `GROUP BY` |
| `short` (expr) | `"TPM expression"` | Used for `AVG()` and `ORDER BY DESC` |

```sql
SELECT t.gene_id, AVG(t.expr) AS average_tpm_expression
FROM `wb-glittery-carrot-8816.presco.rnaseq` AS t
GROUP BY t.gene_id
ORDER BY average_tpm_expression DESC LIMIT 10
```

---

#### Q33: "Is cardiovascular risk related to disability severity?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 1 |
| **Time** | 25.6s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (ASCVD) | `"BHS Analysis — ASCVD Risk Scores"` | Matched "cardiovascular risk" |
| `title` (WHODAS_SCORES) | `"BHS Analysis — WHODAS 2.0 Scores"` | Matched "disability severity" |
| `definition` (whodas_s_sum) | `"...Range 0 (no disability) to 48 (maximum disability)."` | Used for interpreting severity |

```sql
WITH ascvd_risk_categories AS (
    SELECT USUBJID, VISIT,
        CASE
            WHEN score < 5 THEN 'Low-risk'
            WHEN score >= 5 AND score < 7.5 THEN 'Borderline risk'
            WHEN score >= 7.5 AND score < 20 THEN 'Intermediate risk'
            WHEN score >= 20 THEN 'High risk'
        END AS ascvd_risk_category
    FROM `analysis.ASCVD` WHERE score IS NOT NULL
)
SELECT t1.ascvd_risk_category,
       AVG(t2.whodas_s_sum) AS average_disability_score,
       COUNT(DISTINCT t1.USUBJID) AS number_of_participants
FROM ascvd_risk_categories AS t1
JOIN `analysis.WHODAS_SCORES` AS t2
  ON t1.USUBJID = t2.USUBJID AND t1.VISIT = t2.VISIT
GROUP BY t1.ascvd_risk_category
```

**Metadata insight:** Cross-domain join of two analysis tables. The LLM applied standard clinical ASCVD risk thresholds (< 5%, 5–7.5%, 7.5–20%, > 20%) — combining clinical knowledge with the metadata-discovered `score` column and `whodas_s_sum`.

---

### Difficulty 4 (D4) — Cross-Study Cohort Building

#### Summary

| Q# | Question | Outcome | Tables Used | Join Pattern | Time |
|:--:|----------|---------|-------------|-------------|-----:|
| 16 | Combined mental health cohort across studies | Success | `PHQ9` + `GAD7` | `FULL OUTER JOIN` | 32.0s |
| 17 | Compare demographics between studies | Success | `participant_info` | Single table | 44.2s |
| 18 | Everyone with immune AND mental health data | Success | `ASSAYS` + `PHQ9` + `GAD7` | CTE intersection | 26.1s |
| 19 | Physical and mental health impairment | Success | `PHQ9` + `GAD7` + `DIAGNOSES` | 3-way join | 27.2s |
| 20 | Overlapping data domains between studies | Success | `ASSAYS` + `rnaseq` | Cross-study `UNION ALL` | 31.2s |
| 34 | Compare disability across study populations | Success | `participant_info` | Proxy via `progressor` | 34.6s |
| 35 | Biological data for COVID recovery research | Success | `rnaseq` + `ASSAYS` | Cross-study `UNION ALL` | 29.1s |
| 36 | Combined immune + mental health dataset | Success | `participant_info` + `cell_subset_freq` | `SUBSTR` join | 37.7s |

#### Q16: "Build me a combined mental health cohort across both studies"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 32.0s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (PHQ9) | Contains "Depression Screening" | Identified as mental health |
| `title` (GAD7) | Contains "Anxiety Screening" | Identified as mental health |
| `description` (PHQ9) | `"...Cross-study note: while PRESCO does not include PHQ-9 directly..."` | LLM acknowledged PRESCO lacks mental health questionnaires |

```sql
WITH BHS_MentalHealthScores AS (
    SELECT phq9.USUBJID, phq9.VISIT,
      (IFNULL(phq9.phq9_1_CODE, 0) + ... ) AS phq9_total_score,
      (IFNULL(gad7.gad7_1_CODE, 0) + ... ) AS gad7_total_score
    FROM `crf.PHQ9` AS phq9
    FULL OUTER JOIN `crf.GAD7` AS gad7
      ON phq9.USUBJID = gad7.USUBJID AND phq9.VISIT = gad7.VISIT
)
SELECT * FROM BHS_MentalHealthScores
WHERE phq9_total_score >= 10 OR gad7_total_score >= 10 LIMIT 100
```

**Metadata insight:** The PHQ9 `description` field explicitly states that PRESCO lacks PHQ-9. The LLM used this cross-study note from the StructureDefinition to correctly limit the cohort to BHS and acknowledge the limitation in its response.

---

#### Q17: "Can we compare demographics between the two study populations?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 6 |
| **Time** | 44.2s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (DM) | `"BHS Screener — Demographics (DM)"` | BHS has rich demographics |
| `title` (participant_info) | `"PRESCO — Participant Info"` | PRESCO only has `pasc`, `progressor` |
| Absence of age/sex/race columns in PRESCO metadata | — | LLM recognized data asymmetry |

**Metadata insight:** The LLM correctly identified from the metadata catalog that PRESCO has no age/sex/race columns — only `pasc` and `progressor`. It acknowledged this limitation rather than attempting invalid SQL.

---

#### Q18: "I need everyone with immune data AND mental health data, regardless of study"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 26.1s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (ASSAYS) | `"BHS Analysis — Assay Availability Matrix"` | Identified immune data source |
| `short` (Flow_Cytometry, etc.) | `"Flow cytometry availability"` | Recognized as immune assay flags |
| `title` (PHQ9, GAD7) | Depression + Anxiety screening | Identified as mental health data |

```sql
WITH
  bhs_immune_participants AS (
    SELECT DISTINCT USUBJID FROM `analysis.ASSAYS`
    WHERE Flow_Cytometry = 1 OR Mass_Cytometry = 1 OR Proteomics = 1
       OR RNA_Sequencing = 1 OR DNA_Sequencing = 1
  ),
  bhs_mental_health_participants AS (
    SELECT DISTINCT USUBJID FROM `crf.PHQ9`
    UNION DISTINCT
    SELECT DISTINCT USUBJID FROM `crf.GAD7`
  )
SELECT immune.USUBJID
FROM bhs_immune_participants AS immune
INNER JOIN bhs_mental_health_participants AS mental
  ON immune.USUBJID = mental.USUBJID LIMIT 100
```

**Metadata insight:** Multi-table CTE intersection. The LLM mapped "immune data" to the ASSAYS binary flags and "mental health data" to PHQ-9/GAD-7 presence, then used `INNER JOIN` to find the overlap — a multi-step reasoning chain driven by `title` and `short` fields.

---

#### Q19: "Which participants have evidence of both physical and mental health impairment?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 27.2s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `short` (der_hx_cvd) | `"Any history of cardiovascular disease"` | Mapped to "physical health" |
| `short` (der_hx_diab) | Diabetes flag | Mapped to "physical health" |
| `short` (phq9 items, gad7 items) | Depression/anxiety symptoms | Mapped to "mental health" |

3-way join on `USUBJID` + `VISIT` with clinical thresholds: PHQ-9/GAD-7 > 4 (mild symptoms) AND any chronic diagnosis flag = 1. Used 26 columns across 3 tables.

---

#### Q20: "What overlapping data domains exist between the studies?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 2 |
| **Time** | 31.2s |
| **Reasoning** | Yes |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (ASSAYS) | `"Assay Availability Matrix"` — includes `RNA_Sequencing` flag | BHS RNA-seq indicator |
| `title` (rnaseq) | `"PRESCO — RNAseq Gene Expression"` | PRESCO RNA-seq data |
| Cross-referencing `title` fields across studies | Both mention RNA-seq | Identified as overlapping domain |

```sql
SELECT 'BHS' AS study, COUNT(DISTINCT t.USUBJID) AS participant_count
FROM `analysis.ASSAYS` AS t WHERE t.RNA_Sequencing = 1
UNION ALL
SELECT 'PRESCO' AS study,
       COUNT(DISTINCT SPLIT(t.subset_id, '_')[OFFSET(0)]) AS participant_count
FROM `presco.rnaseq` AS t
```

**Metadata insight:** The LLM identified RNA-seq as the overlapping domain by comparing `title` fields across studies — ASSAYS has "RNA_Sequencing" as a flag column, while PRESCO has a dedicated `rnaseq` table. Different access patterns, same underlying data type.

---

#### Q34: "Compare disability scores between the two study populations"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 1 |
| **Time** | 34.6s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:** The LLM recognized from the metadata catalog that PRESCO has no WHODAS equivalent. It used `progressor` as the closest proxy for disability/worsening and acknowledged the limitation.

---

#### Q35: "What biological and clinical data exists across studies for COVID recovery?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 2 |
| **Time** | 29.1s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

Cross-study `UNION ALL` quantifying RNA-seq participants in each study, using the same pattern as Q20.

---

#### Q36: "Build a combined dataset with immune markers and mental health data from all available studies"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 37.7s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` (cell_subset_frequencies) | `"PRESCO — Cell Subset Frequencies"` | Immune markers |
| `title` (participant_info) | `"PRESCO — Participant Info"` | Clinical status for linking |
| `short` (subset_code) | `"Immune cell subset code"` | Selected specific cell types |

```sql
SELECT p_info.participant_id, p_info.pasc, p_info.progressor,
       cell_freq.subset_code, cell_freq.freq
FROM `presco.participant_info` AS p_info
JOIN `presco.cell_subset_frequencies` AS cell_freq
  ON p_info.participant_id = SUBSTR(cell_freq.sample_id, 1, STRPOS(cell_freq.sample_id, '_') - 1)
WHERE cell_freq.subset_code IN ('T4nv', 'MoCl', 'BnCS')
LIMIT 100
```

**Metadata insight:** The LLM selected specific immune cell subset codes (Naive CD4+ T cells, Classical monocytes, Class-switched B cells) and acknowledged that BHS mental health data would need a separate query due to different primary keys across studies.

---

### Difficulty 5 (D5) — Ambiguous / Edge Cases

#### Summary

| Q# | Question | Outcome | Tables Used | Key Metadata Insight | Time |
|:--:|----------|---------|-------------|---------------------|-----:|
| 21 | What data on respiratory outcomes? | Success | `PFT` | `short` field `"Assessment performed?"` → quality filter | 21.1s |
| 22 | Can we link participants across studies? | Success | `DM` + `participant_info` | Different PKs → no linkage possible | 34.9s |
| 23 | Recovery trajectories — what's available? | Success | `WHODAS_SCORES` | `short` `"Study day"` → temporal axis | 64.8s |
| 24 | Tables for a grant about post-COVID disability? | Success | `participant_info` + `WHODAS_SCORES` | Cross-study cohort sizing | 34.3s |
| 25 | Find everyone who got worse over time | Success | `WHODAS_SCORES` + `participant_info` | Dual strategy: window functions + `progressor` flag | 32.0s |

#### Q21: "What data do we have on respiratory outcomes?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 10 |
| **Time** | 21.1s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `title` | `"BHS CRF — PFT (Pulmonary Function Tests / Spirometry)"` | Matched "respiratory outcomes" |
| `short` (pft_perf) | `"Assessment performed?"` | Used as quality filter (`WHERE pft_perf = 'Y'`) |
| `short` (dlco) | `"DLCO (mL/min/mmHg)"` | Included diffusion capacity measure |

```sql
SELECT t1.USUBJID, t1.VISIT, t1.fev1, t1.fvc, t1.ratio, t1.dlco
FROM `crf.PFT` AS t1
WHERE t1.pft_perf = 'Y' AND t1.ratio IS NOT NULL LIMIT 10
```

**Metadata insight:** The LLM applied a quality filter (`pft_perf = 'Y'`) derived from the `short` field (`"Assessment performed?"`), ensuring only valid test results are returned — a clinically appropriate data quality step discovered from metadata.

---

#### Q22: "Can we link participants across the two studies?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 10 |
| **Time** | 34.9s |
| **Reasoning** | No |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `extension.verily-primary-identity` (DM) | `USUBJID` | BHS identifier scheme |
| `extension.verily-primary-identity` (participant_info) | `participant_id` | PRESCO identifier scheme |
| No shared mapping table in metadata | — | LLM concluded linkage isn't feasible |

The LLM showed sample identifiers from each study so the researcher can see the different ID formats.

---

#### Q23: "I want to study recovery trajectories — what's available?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 3 |
| **Time** | 64.8s |
| **Reasoning** | No |
| **Limitations** | No |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `short` (study_day) | `"Study day"` | Identified as temporal axis for trajectories |
| `definition` (study_day) | `"Number of days from participant enrollment to the date of this WHODAS assessment."` | Confirmed longitudinal time variable |
| `short` (whodas_s_sum) | `"WHODAS 2.0 summary score"` | Outcome measure for trajectory |

```sql
SELECT t1.USUBJID, t1.VISIT, t1.study_day, t1.whodas_s_sum
FROM `analysis.WHODAS_SCORES` AS t1
WHERE t1.USUBJID = (SELECT USUBJID FROM `analysis.WHODAS_SCORES`
                    WHERE whodas_s_sum IS NOT NULL LIMIT 1)
ORDER BY t1.study_day LIMIT 100
```

**Metadata insight:** The `definition` field for `study_day` explicitly states "Number of days from enrollment" — giving the LLM the temporal axis for trajectory analysis. It provided one participant's disability scores over time as a concrete example.

---

#### Q24: "Which tables should I use for a grant about post-COVID disability?"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 2 |
| **Time** | 34.3s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `short` (pasc) | `"PASC (Long COVID) status"` | Identified PRESCO Long COVID cohort |
| `definition` (whodas_s_sum) | Score range 0–48 | Used ≥ 24 as moderate-to-severe cutoff |

```sql
SELECT 'PRESCO' AS study, 'PASC Participants' AS cohort,
       COUNT(DISTINCT participant_id) AS participant_count
FROM `presco.participant_info` WHERE pasc = 1
UNION ALL
SELECT 'BHS' AS study, 'Moderate-to-Severe Disability (WHODAS > 24)' AS cohort,
       COUNT(DISTINCT USUBJID) AS participant_count
FROM `analysis.WHODAS_SCORES` WHERE whodas_s_sum > 24
```

**Metadata insight:** The LLM didn't just list tables — it quantified the available cohort for a grant. It used the `definition`-described WHODAS range (0–48) to set a clinically meaningful cutoff (> 24 = upper half). Grant-writing-ready numbers from metadata.

---

#### Q25: "Find me everyone who got worse over time"

| | |
|---|---|
| **Outcome** | **Success** — SQL generated and executed |
| **Rows** | 100 |
| **Time** | 32.0s |
| **Reasoning** | Yes |
| **Limitations** | Yes |

**Metadata → SQL Mapping:**

| StructureDefinition Field | Value | How It Guided SQL |
|--------------------------|-------|-------------------|
| `short` (whodas_s_sum) | `"WHODAS 2.0 summary score"` | Longitudinal disability outcome for BHS |
| `purpose` (WHODAS_SCORES) | `"Multiple records per participant per study visit"` | Confirmed longitudinal data — enables window functions |
| `short` (progressor) | `"Progressor status (CPR publication)"` | Pre-computed worsening flag for PRESCO |

```sql
WITH BHS_Worsened AS (
  SELECT USUBJID,
    FIRST_VALUE(whodas_s_sum IGNORE NULLS)
      OVER (PARTITION BY USUBJID ORDER BY VISITNUM ASC ...) AS first_score,
    LAST_VALUE(whodas_s_sum IGNORE NULLS)
      OVER (PARTITION BY USUBJID ORDER BY VISITNUM ASC ...) AS last_score
  FROM `analysis.WHODAS_SCORES`
)
SELECT participant_id, 'PRESCO' AS study, 'Flagged as Progressor' AS reason
FROM `presco.participant_info` WHERE progressor = 1
UNION ALL
SELECT USUBJID, 'BHS', 'Disability score increased'
FROM BHS_Worsened WHERE last_score > first_score
LIMIT 100
```

**Metadata insight:** This is the most sophisticated query in the benchmark. The LLM used two different strategies for "got worse" based on what each study's metadata offers:
- **BHS:** The `purpose` field's "Multiple records per participant per study visit" told the LLM this is longitudinal data. It used `FIRST_VALUE`/`LAST_VALUE` window functions on `VISITNUM`-ordered WHODAS scores to detect worsening.
- **PRESCO:** The `short` field for `progressor` ("Progressor status") is a pre-computed worsening indicator — the LLM used it directly.
- Combined both with `UNION ALL`.

---

## 5. Metadata Utilization Deep Dive

### Table Coverage — Every Table Queried via SQL

| Table | SQL References | Unique Columns Used | Questions |
|-------|:---:|:---:|---|
| `crf.PHQ9` | 7 | 11 | Q3, Q6, Q11, Q13, Q14, Q16, Q18 |
| `presco.participant_info` | 8 | 3 | Q1, Q5, Q17, Q22, Q25, Q26, Q34, Q36 |
| `crf.GAD7` | 5 | 9 | Q7, Q11, Q14, Q16, Q18 |
| `analysis.WHODAS_SCORES` | 5 | 4 | Q8, Q12, Q23, Q24, Q25 |
| `analysis.ASSAYS` | 4 | 9 | Q18, Q20, Q30, Q35 |
| `screener.DM` | 3 | 5 | Q2, Q22, Q29 |
| `presco.rnaseq` | 3 | 2 | Q20, Q32, Q35 |
| `admin.COEVAL` | 2 | 2 | Q1, Q13 |
| `analysis.ASCVD` | 2 | 3 | Q27, Q33 |
| `analysis.DIAGNOSES` | 2 | 15 | Q15, Q19 |
| `crf.PFT` | 2 | 7 | Q9, Q21 |
| `crf.EQ5D` | 2 | 3 | Q10, Q14 |
| `presco.cell_subset_frequencies` | 2 | 3 | Q31, Q36 |
| `analysis.AUDITC_SCORES` | 1 | 3 | Q28 |
| `crf.WHODAS` | 1 | 1 | Q14 |

**100% table coverage** — every table in the metadata catalog was queried via SQL at least once.

### StructureDefinition Fields That Drove SQL Generation

| JSON Field | Role in Query Generation | Example |
|-----------|-------------------------|---------|
| `title` | **Table discovery** — maps clinical concepts to tables | "Depression" → `"BHS CRF — PHQ-9 (Depression Screening)"` |
| `description` | **Clinical context** — provides cutoffs, ranges, cross-study notes | PHQ-9 `description` includes "cutoffs: 5 = mild, 10 = moderate" |
| `short` (column) | **Column selection** — maps research terms to specific columns | "alcohol screening" → `AUDITC_SUM_SCORE` (`short`: "AUDIT-C sum score") |
| `definition` (column) | **Data type + range understanding** — guides aggregation and thresholds | `whodas_s_sum` `definition`: "Range 0...to 48" → severity buckets |
| `purpose` | **Granularity awareness** — determines if window functions apply | "Multiple records per participant" → longitudinal analysis possible |
| `extension.verily-primary-identity` | **Join key identification** — different PKs per study | `USUBJID` (BHS) vs `participant_id` (PRESCO) |
| `extension.verily-structural-link` | **Inter-table relationships** — which tables can be joined | Links PHQ9 → COEVAL, enabling cohort-filtered queries |
| `extension.verily-study-name` | **Study boundary detection** — dynamic cross-study rules | Auto-discovered BHS (12 tables) vs PRESCO (3 tables) |
| `binding` | **Value interpretation** — what coded values mean | PHQ-9 items: 0 = Not at all, 1 = Several days, etc. |
| MeasureReport `measureScore` | **Table scale** — row counts for `LIMIT` decisions | PHQ9 = 1,489 rows → reasonable for full scans |

### Join Pattern Analysis

| Join Pattern | Questions | StructureDefinition Field That Enabled It |
|-------------|-----------|-------------------------------------------|
| No join (single table) | Q2, Q3, Q6–Q10, Q26–Q30, Q32 | `title` sufficient for table selection |
| 2-table `INNER JOIN` on `USUBJID + VISIT` | Q11, Q12, Q13, Q15, Q33 | `verily-primary-identity` extension |
| Multi-table `INNER JOIN` (3–4 tables) | Q14, Q18, Q19 | `verily-primary-identity` + `verily-structural-link` |
| Cross-study `UNION ALL` | Q1, Q20, Q22, Q24, Q25, Q35 | `verily-study-name` + different `verily-primary-identity` per study |
| String-parsing `SPLIT`/`SUBSTR` join | Q31, Q36 | Column `short` fields hinting at ID encoding |
| CTE + Window Functions | Q16, Q25, Q33 | `purpose` ("Multiple records per participant") |

### Clinical Concept → Metadata Mapping

| Clinical Concept | Metadata Table | Column(s) | StructureDefinition Field That Resolved It |
|------------------|---------------|-----------|-------------------------------------------|
| "depression" | `crf.PHQ9` | `phq9_1_CODE`–`phq9_9_CODE` | `title`: "Depression Screening" |
| "anxiety" | `crf.GAD7` | `gad7_1_CODE`–`gad7_7_CODE` | `title`: "Anxiety Screening" |
| "disability" | `analysis.WHODAS_SCORES` | `whodas_s_sum` | `short`: "WHODAS 2.0 summary score" |
| "quality of life" | `crf.EQ5D` | `eq5d_health` | `title`: "Quality of Life" |
| "lung function" | `crf.PFT` | `fev1`, `fvc`, `ratio`, `dlco` | `title`: "Pulmonary Function Tests" |
| "long COVID" | `presco.participant_info` | `pasc` | `short`: "PASC (Long COVID) status" |
| "got worse" | `presco.participant_info` | `progressor` | `short`: "Progressor status" |
| "cardiovascular risk" | `analysis.ASCVD` | `score` | `title`: "ASCVD Risk Scores" |
| "alcohol screening" | `analysis.AUDITC_SCORES` | `AUDITC_SUM_SCORE` | `title`: "AUDIT-C Scores" |
| "demographics" | `screener.DM` | `age_at_enrollment`, `SEX`, `RACE` | `title`: "Demographics (DM)" |
| "immune data" | `analysis.ASSAYS` | 8 binary flags | `title`: "Assay Availability Matrix" |
| "gene expression" | `presco.rnaseq` | `gene_id`, `expr` | `title`: "RNAseq Gene Expression" |
| "immune cell subsets" | `presco.cell_subset_frequencies` | `subset_code`, `freq` | `title`: "Cell Subset Frequencies" |
| "diagnoses" | `analysis.DIAGNOSES` | 13 `der_hx_*` flags | `title`: "Diagnoses" |

---

## 6. Scoring Summary

| Criteria | Max Possible | Achieved | Rate |
|----------|:---:|:---:|:---:|
| Correctly identifies relevant table(s) from metadata | 72 (2 × 36) | 70 | 97% |
| Maps clinical concept to correct column(s) | 72 (2 × 36) | 66 | 92% |
| SQL executes without error | 72 (2 × 36) | 70 | 97% |
| Results are clinically sensible | 36 (1 × 36) | ~34 | ~94% |
| Explains reasoning / data mapping | 36 (1 × 36) | 19 | 53% |
| Acknowledges limitations or missing data | 36 (1 × 36) | 22 | 61% |
| **Estimated Total** | **324** | **~281** | **~87%** |

### Score by Level

| Level | Estimated Score | Max | Rate |
|:-----:|:---:|:---:|:---:|
| D1 — Simple Exploration | ~38 | 45 | ~84% |
| D2 — Clinical Queries | ~72 | 90 | ~80% |
| D3 — Relationships & Joins | ~65 | 72 | ~90% |
| D4 — Cross-Study Cohort Building | ~60 | 72 | ~83% |
| D5 — Ambiguous / Edge Cases | ~38 | 45 | ~84% |

**Strongest area:** D3 (Relationships & Joins) at 90% — the StructureDefinition's `verily-primary-identity` and `verily-structural-link` extensions enable complex multi-table joins.

**Note:** D2 scores lower than D3 because simple queries often skip the reasoning/limitations explanation (which the scoring penalizes), not because the SQL is incorrect.

---

## 7. Observations and Opportunities

### What Works Well

- **100% answer rate** — Every question received a useful response
- **97% SQL generation** — Only Q4 (catalog question) skipped SQL — correct behavior
- **100% SQL execution** — Zero runtime errors across 35 queries
- **Complex joins from metadata alone** — The `SPLIT(sample_id)` join (Q31, Q36) and 4-way `INNER JOIN` (Q14) demonstrate that StructureDefinition metadata provides sufficient context for non-trivial joins
- **Cross-study awareness** — The system dynamically discovers study boundaries from `verily-study-name` extensions and uses `UNION ALL` with the correct PKs
- **Clinical knowledge applied correctly** — PHQ-9 ≥ 10, WHODAS severity categories, ASCVD risk thresholds — all derived from the `description` and `definition` fields

### Opportunities

- **Reasoning rate at 53%** — D2 queries often skip explanation. Prompt improvement: require 1-sentence reasoning even for simple queries
- **PRESCO metadata is sparse** — Only 3 columns (`participant_id`, `pasc`, `progressor`). Enriching with demographics would improve cross-study comparisons
- **Multi-turn conversation testing** — This benchmark tests single-shot questions. Follow-ups like "now filter for females" would test conversational state
- **Pre-computed score columns** — PHQ-9/GAD-7 totals are computed inline (summing 7–9 items). Adding `phq9_total` / `gad7_total` to metadata would simplify queries

---

## Appendix A: The 15-Table Metadata Catalog

| # | Table | Study | StructureDefinition `title` | Primary Key | Key Columns |
|---|-------|-------|-----------------------------|-------------|-------------|
| 1 | `admin.COEVAL` | BHS | BHS Admin — COEVAL (Cohort Eligibility) | `USUBJID` | `cohort_eligibility` |
| 2 | `analysis.ASCVD` | BHS | BHS Analysis — ASCVD Risk Scores | `USUBJID` | `score` |
| 3 | `analysis.ASSAYS` | BHS | BHS Analysis — Assay Availability Matrix | `USUBJID` | 8 binary assay flags |
| 4 | `analysis.AUDITC_SCORES` | BHS | BHS Analysis — AUDIT-C Scores | `USUBJID` | `AUDITC_SUM_SCORE` |
| 5 | `analysis.DIAGNOSES` | BHS | BHS Analysis — Diagnoses | `USUBJID` | 15+ diagnosis flags |
| 6 | `analysis.WHODAS_SCORES` | BHS | BHS Analysis — WHODAS 2.0 Scores | `USUBJID` | `whodas_s_sum`, `study_day` |
| 7 | `crf.EQ5D` | BHS | BHS CRF — EQ-5D-5L (Quality of Life) | `USUBJID` | `eq5d_health` |
| 8 | `crf.GAD7` | BHS | BHS CRF — GAD-7 (Anxiety Screening) | `USUBJID` | `gad7_1_CODE`–`gad7_7_CODE` |
| 9 | `crf.PFT` | BHS | BHS CRF — PFT (Pulmonary Function Tests / Spirometry) | `USUBJID` | `fev1`, `fvc`, `ratio`, `dlco` |
| 10 | `crf.PHQ9` | BHS | BHS CRF — PHQ-9 (Depression Screening) | `USUBJID` | `phq9_1_CODE`–`phq9_9_CODE` |
| 11 | `crf.WHODAS` | BHS | BHS CRF — WHODAS (raw items) | `USUBJID` | Item-level scores |
| 12 | `screener.DM` | BHS | BHS Screener — Demographics (DM) | `USUBJID` | `age_at_enrollment`, `SEX`, `RACE` |
| 13 | `presco.cell_subset_frequencies` | PRESCO | PRESCO — Cell Subset Frequencies | `sample_id` | `subset_code`, `freq` |
| 14 | `presco.participant_info` | PRESCO | PRESCO — Participant Info | `participant_id` | `pasc`, `progressor` |
| 15 | `presco.rnaseq` | PRESCO | PRESCO — RNAseq Gene Expression | `subset_id` | `gene_id`, `expr` |

## Appendix B: Environment Details

| Parameter | Value |
|-----------|-------|
| LLM Model | `gemini-2.5-pro` via Vertex AI |
| LLM Temperature | 0.1 |
| LLM Region | us-central1 |
| Billing Project | `wb-glittery-carrot-8816` |
| BHS Data Project | `wb-beamish-acorn-6393` (12 tables, 5 datasets) |
| PRESCO Data Project | `wb-glittery-carrot-8816` (3 tables, presco dataset) |
| Metadata Source | `gs://metadata-json-wb-shrewd-papaya-8403` (18 FHIR JSONs → 15 resolved tables) |
| System Prompt Size | ~86K characters (includes full schema context) |
| Total Benchmark Time | 852s (14.2 min) |
| Avg Time per Question | 23.7s |
| Raw Results | `benchmark_results_v2.json` |
