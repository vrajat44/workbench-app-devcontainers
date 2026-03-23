# WB Metadata Creator

## What Is This?

Healthcare data stored in BigQuery warehouses is just tables and columns — it has no inherent meaning a machine or a new team member can understand. Before this data can be used in clinical research, analytics, or interoperability, someone has to document what every column means, how sensitive it is, and how it maps to healthcare standards like FHIR.

**WB Metadata Creator automates that entire process.** Point it at any BigQuery dataset and it will:

1. **Profile the data** — scan every table to understand column types, null rates, distinct values, and statistical distributions
2. **Generate rich descriptions** — use Google's Gemini LLM to write human-readable definitions, classify sensitivity (PHI, PII, etc.), and map columns to FHIR resources
3. **Produce standards-compliant FHIR metadata** — output StructureDefinitions, ValueSets, ConceptMaps, and data profiles that conform to the Verily FHIR Implementation Guide (VFIG)
4. **Validate everything** — run an 8-check validation pipeline (using LLM-as-a-Judge) to catch errors before the metadata ships

The result: what used to take a data steward days or weeks of manual spreadsheet work is done in minutes, with higher consistency and full FHIR compliance.

---

## Who Is This For?

- **Data stewards and curators** who need to catalog new datasets entering Verily's platform
- **Research teams** onboarding new studies and needing FHIR-compliant metadata for Cortex
- **Platform engineers** automating metadata pipelines for data ingestion

No FHIR expertise is required — the app handles all the structural complexity. Users review and edit the results through a simple web UI.

---

## How It Works

The app is a browser-based tool with a guided multi-step workflow:

### Step 1 — Setup
Pick your study, configure compliance settings (HIPAA zone, retention, confidentiality), and select which BigQuery datasets and tables to process.

### Step 2 — Profile & Define
The app runs BigQuery profiling queries (null counts, distinct values, top-N distributions, numeric stats) on each table, then sends the schema + profile data to Gemini to generate:
- Column descriptions and short labels
- Sensitivity classifications (using a 29-code DS4P vocabulary — PHI, PII, UID, etc.)
- Measurement method tags (self-reported, lab-measured, device-collected, etc.)
- FHIR type mappings and FHIR resource mappings

Everything appears in an editable table so you can review and override any AI-generated value.

### Step 3 — Validate Inputs
Before generating any output, an LLM-as-a-Judge validation checks your metadata table for:
- Missing or low-quality descriptions
- Incorrect FHIR type mappings
- Sensitivity label accuracy
- Cross-table consistency issues

Issues are flagged with severity levels and fix suggestions you can apply with one click.

### Step 4 — Generate FHIR Metadata
The app assembles complete FHIR R4 resources:
- **StructureDefinitions** (logical type) — one per table, with full element definitions, sensitivity labels, and FHIR mappings
- **ValueSets** — for coded columns with enumerated values
- **ConceptMaps** — mapping source codes to standard terminologies
- **Terminology Bundles** — packaging CodeSystems + ValueSets + ConceptMaps
- **Data Profiles** (MeasureReport) — statistical summaries of column distributions
- **Measure Definitions** — shared metric templates

### Step 5 — Deliver to Workbench
Save generated metadata to a GCS bucket or download as a ZIP archive for integration into your data catalog.

### Step 6 — Validate Output JSONs
A post-generation validation pipeline runs 8 independent checks:

| Check | What It Verifies |
|:-----:|------------------|
| 1 | Column coverage — all BQ columns present in the StructureDefinition |
| 2 | Data type accuracy — FHIR types semantically match the source data |
| 3 | VFIG mapping accuracy — FHIR resource mappings are plausible |
| 4 | Security label accuracy — sensitivity classifications are correct |
| 5 | Measurement method accuracy — method tags are appropriate |
| 6 | Cross-file consistency — shared columns are consistent across tables |
| 7 | L3 metadata completeness — all table-level fields are populated |
| 8 | ValueSet binding completeness — coded columns are bound to ValueSets |

### Chat-Based Refinement
After generation, a built-in LangGraph agent lets you refine metadata through natural-language conversation (e.g., "Change phq9_1 to PHI sensitivity" or "Add a mapping comment for the enrollment date column").

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Web UI | Gradio (≥5.0) |
| LLM | Google Gemini 2.5 (via Vertex AI) |
| Data profiling | BigQuery (INFORMATION_SCHEMA + profiling queries) |
| Chat agent | LangGraph + LangChain |
| Output format | FHIR R4 (StructureDefinition, ValueSet, ConceptMap, MeasureReport) |
| Storage | Google Cloud Storage (optional) |
| Deployment | Verily Workbench (custom cloud app) or local |

---

## Running Locally

### Prerequisites

- Python 3.10+
- GCP credentials with access to the data projects:
  ```bash
  gcloud auth application-default login
  ```
- Install dependencies:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

### Quick Start

```bash
source .venv/bin/activate
python app.py \
  --project=<BILLING_PROJECT_ID> \
  --data-project <DATA_PROJECT_ID> \
  --port=7870
```

Open `http://localhost:7870` in your browser.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--project` | None | GCP project for billing (BQ jobs + Vertex AI) |
| `--data-project` | Same as `--project` | BQ project(s) where source data tables live (space-separated) |
| `--json-dir` | Local metadata dir | Path or `gs://` URI to existing FHIR metadata (for cross-reference) |
| `--output-bucket` | None | GCS bucket to upload generated JSONs |
| `--port` | `7860` | Gradio server port |
| `--share` | `false` | Create a public Gradio share link |

---

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Gradio multi-tab UI — application entry point |
| `models.py` | Data models (BQColumnInfo, StudyConfig, ColumnMetadata, TableMetadata) |
| `bq_profiler.py` | BigQuery schema discovery and 4-phase data profiling |
| `bq_executor.py` | BigQuery execution utilities |
| `fhir_llm.py` | LLM-based generation (StructureDefinitions, ValueSets, ConceptMaps, column definitions) |
| `fhir_builder.py` | Deterministic FHIR JSON assembly (no LLM calls) |
| `fhir_validator.py` | Input + output validation pipeline (LLM-as-Judge, 8 checks) |
| `fhir_generator.py` | Re-export shim for backwards-compatible imports |
| `sensitivity.py` | Sensitivity classification using a 29-code DS4P vocabulary |
| `prompt_engine.py` | LLM prompt construction and response parsing |
| `metadata_loader.py` | Parses existing FHIR JSONs for cross-reference during generation |
| `agent.py` | LangGraph chat agent for natural-language metadata refinement |
| `gcs_utils.py` | GCS bucket and folder discovery |
| `start.sh` | Container startup script (for Workbench deployment) |
| `Dockerfile` | Container image definition |

---

## Deploying to Workbench

The app is deployed as a Verily Workbench custom cloud app.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Auto-detected from GCE metadata on Workbench |
| `DATA_PROJECT_IDS` | Space-separated BQ data project IDs |
| `METADATA_SOURCE` | GCS URI or local path for existing FHIR metadata |
| `OUTPUT_GCS_BUCKET` | GCS bucket for saving generated metadata |

---

## Required GCP Permissions

| IAM Role | Why |
|----------|-----|
| `roles/bigquery.dataViewer` | Read BQ table schemas and data |
| `roles/bigquery.jobUser` | Run BQ profiling queries |
| `roles/aiplatform.user` | Call Vertex AI / Gemini LLM |
