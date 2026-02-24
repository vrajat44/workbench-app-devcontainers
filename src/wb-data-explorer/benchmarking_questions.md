# WB Data Explorer — Benchmarking Questions (v2)

Questions a real researcher would ask — **no column names, no table hints, no technical context.**  
The LLM must rely entirely on FHIR metadata to understand and answer.

**v2 Changes:** Added 11 new questions (Q26–Q36) targeting untested tables (ASCVD, AUDITC, ASSAYS, DM, rnaseq, cell_subset_frequencies), PRESCO-focused queries, and cross-study cohort building.

---

## Level 1 — Simple Exploration (5 questions)

| # | Question |
|---|----------|
| 1 | How many participants are in each study? |
| 2 | What demographic information do we have? |
| 3 | Show me the first few rows of the depression survey data |
| 4 | What kinds of data are available across all studies? |
| 5 | Are there any participants flagged as having long COVID? |

---

## Level 2 — Clinical Queries (10 questions)

| # | Question | Target Tables |
|---|----------|---------------|
| 6 | How many people screened positive for depression? | PHQ9 |
| 7 | What's the average anxiety score across all visits? | GAD7 |
| 8 | Show the distribution of disability severity | WHODAS_SCORES |
| 9 | Do we have lung function measurements? What's the average? | PFT |
| 10 | How does self-reported quality of life vary by visit? | EQ5D |
| **26** | **How many PRESCO participants are progressors versus non-progressors?** | **participant_info** |
| **27** | **What is the average cardiovascular risk score?** | **ASCVD** |
| **28** | **Show me alcohol screening results** | **AUDITC_SCORES** |
| **29** | **What are the demographics broken down by sex and race?** | **DM** |
| **30** | **Which participants have lab assay data and what was measured?** | **ASSAYS** |

---

## Level 3 — Relationships & Joins (8 questions)

| # | Question | Target Tables |
|---|----------|---------------|
| 11 | Are people with depression also more likely to have anxiety? | PHQ9 + GAD7 |
| 12 | Is there a relationship between lung function and disability? | PFT + WHODAS_SCORES |
| 13 | Compare depression scores between eligible and ineligible cohort members | COEVAL + PHQ9 |
| 14 | Which participants completed all the mental health questionnaires? | PHQ9 + GAD7 + EQ5D + WHODAS |
| 15 | Do participants with more diagnoses have worse quality of life? | DIAGNOSES + EQ5D |
| **31** | **Show immune cell subset frequencies for PASC versus non-PASC participants** | **cell_subset_frequencies + participant_info** |
| **32** | **What are the top expressed genes in the PRESCO data?** | **rnaseq** |
| **33** | **Is cardiovascular risk related to disability severity?** | **ASCVD + WHODAS_SCORES** |

---

## Level 4 — Cross-Study Cohort Building (8 questions)

| # | Question | Target Tables |
|---|----------|---------------|
| 16 | Build me a combined mental health cohort across both studies | PHQ9 + GAD7 + participant_info |
| 17 | Can we compare demographics between the two study populations? | DM + participant_info |
| 18 | I need everyone with immune data AND mental health data, regardless of study | cell_subset + rnaseq + PHQ9 + GAD7 |
| 19 | Which participants have evidence of both physical and mental health impairment across any study? | WHODAS + PHQ9 + GAD7 |
| 20 | What overlapping data domains exist between the studies? Could we do a combined analysis? | All tables (metadata) |
| **34** | **Compare disability scores between the two study populations** | **WHODAS_SCORES + cross-study** |
| **35** | **What biological and clinical data exists across studies for COVID recovery research?** | **Cross-study metadata** |
| **36** | **Build a combined dataset with immune markers and mental health data from all available studies** | **cell_subset + rnaseq + PHQ9 + GAD7** |

---

## Level 5 — Ambiguous / Edge Cases (5 questions)

| # | Question |
|---|----------|
| 21 | What data do we have on respiratory outcomes? |
| 22 | Can we link participants across the two studies? |
| 23 | I want to study recovery trajectories — what's available? |
| 24 | Which tables should I use if I'm writing a grant about post-COVID disability? |
| 25 | Find me everyone who got worse over time |

---

## Scoring Guide

| Criteria | Score |
|----------|-------|
| Correctly identifies relevant table(s) from metadata | +2 |
| Maps clinical concept to correct column(s) | +2 |
| SQL executes without error | +2 |
| Results are clinically sensible | +1 |
| Explains reasoning / data mapping | +1 |
| Acknowledges limitations or missing data | +1 |
| **Max per question** | **9** |

**Total possible: 324 points (36 × 9)**

---

## Top 5 Demo Questions

If you only have time for 5:

1. **D1:** `What kinds of data are available across all studies?`
2. **D2:** `How many people screened positive for depression?`
3. **D3:** `Are people with depression also more likely to have anxiety?`
4. **D4:** `Build me a combined mental health cohort across both studies`
5. **D5:** `I want to study recovery trajectories — what's available?`

---

## Question Coverage by Table

| Table | Questions Testing It |
|-------|---------------------|
| PHQ9 | Q3, Q6, Q11, Q13, Q14, Q16, Q18, Q19, Q36 |
| GAD7 | Q7, Q11, Q14, Q16, Q18, Q19, Q36 |
| WHODAS_SCORES | Q8, Q12, Q14, Q19, Q33, Q34 |
| EQ5D | Q10, Q14, Q15 |
| PFT | Q9, Q12 |
| DIAGNOSES | Q15 |
| COEVAL | Q1, Q13 |
| DM | Q2, Q17, Q29 |
| participant_info | Q5, Q17, Q26, Q31 |
| ASCVD | Q27, Q33 |
| AUDITC_SCORES | Q28 |
| ASSAYS | Q30 |
| cell_subset_frequencies | Q18, Q31, Q36 |
| rnaseq | Q18, Q32, Q36 |
