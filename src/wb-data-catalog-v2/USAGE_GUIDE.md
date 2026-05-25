# Data Catalog v2 — Usage Guide

## Setup

1. Open the app. If no project is configured, you'll see the Settings page.
2. Set your **Billing Project** (GCP project for compute costs) and **Data Project** (project with your BigQuery tables). These can be different.
3. Pick a **Gemini Model** from the dropdown or leave on Auto-detect.
4. Click **Save & reload**.

## Profiling Tables

The catalog shows your BQ datasets. Click a dataset name to expand it and see its tables.

**To profile tables:**
- Click checkboxes to select tables, then use the floating action bar to choose **Technical**, **Semantic**, or **Both**.
- Or click **Profile all** on a dataset header.
- Or click **Profile entire project** at the top.

**Technical profiling** (no AI): column types, null rates, distinct counts, top values, patterns. ~10-30 seconds per table.

**Semantic profiling** (AI): business names, definitions, sensitivity labels, terminology bindings, join paths, cohort dimensions. ~1-2 minutes per table. Requires technical profiling first.

A toast notification appears when profiling finishes.

## Viewing a Table Profile

Click any table name to open its detail page. Five tabs:

| Tab | What it shows |
|-----|--------------|
| **Preview** | First 50 rows of raw data |
| **Technical** | Column stats, null rates, value distributions, anomalies |
| **Semantic** | AI-generated definitions, sensitivity codes, terminology bindings. Editable — click Edit to change definitions or sensitivity labels. |
| **Key Insights** | AI-suggested chart visualizations |
| **Interactive Explorer** | Drag-and-drop visual analytics (Graphic Walker) |

## Sensitivity Labels

Columns are tagged with HIPAA Safe Harbor codes:

| Code | Meaning |
|------|---------|
| P_DOB | Date of birth |
| P_SSN | Social Security Number |
| P_MRN | Medical record number |
| P_PNAME | Patient name |
| P_EMAIL | Email address |
| P_PHONE | Phone number |
| P_RACEETHNICITY | Race/ethnicity |
| UID | Unique identifier |
| FREETEXT | Free text needing DLP scan |

Red = HIPAA direct identifier. Amber = quasi-identifier. Blue = UID/free text.

## Data AMA Agent

Ask questions about your data in natural language.

**Q&A Mode**: metadata questions. "What tables have diagnosis data?", "Explain the SUBJID column", "What joins exist?"

**Agent Mode**: generates and executes SQL. "Count patients with diabetes", "Show top 10 diagnosis codes", "Find subjects with PHQ-9 > 10"

Click **Load full details** to give the agent access to complete column-level statistics for more precise answers.

## Terminology

Browse standardized codes (LOINC, SNOMED, ICD-10, RxNorm, CPT) discovered across all profiled tables. Shows which columns in which tables use each code.

## Cohort Builder

Build patient cohorts using three approaches:
- **Table Filters**: pick dimensions from semantic profiles, set filter conditions
- **Terminology**: filter by standardized codes across tables
- **Natural Language**: describe your cohort in plain English, AI generates SQL

## Settings

- Change billing/data project at any time
- Switch Gemini model
- **Reset All Profiles**: deletes all profiling artifacts to start fresh (requires confirmation)

## Help

Click the **?** button in the sidebar to open contextual help tips for the current page.

## Feedback

Use the banner at the top of the page to report bugs or share feedback.
