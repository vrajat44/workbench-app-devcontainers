# WB Metadata Creator

**FHIR metadata generation for Verily Workbench** — a multi-step Gradio UI that profiles BigQuery tables and generates FHIR StructureDefinition JSONs enriched with LLM-generated semantic definitions.

Built with **Gradio** (multi-tab UI) + **Gemini** (LLM) + **BigQuery** (data profiling) + **FHIR R4** (output format).

---

## How It Works

The app implements a 5-step workflow:

| Tab | Step | What It Does |
|:---:|------|--------------|
| 1 | **Setup** | Configure study name, domain, and select BQ source tables |
| 2 | **Profile & Define** | Profile BQ tables (nulls, cardinality, types) + LLM generates column definitions → editable table |
| 3 | **Validate Inputs** | LLM-as-a-Judge validates the metadata table before JSON generation |
| 4 | **Generate & Export** | Build FHIR StructureDefinition JSONs + optional ValueSets + data profiles → download |
| 5 | **Validate JSONs** | Post-generation 8-check validation of the output JSONs |

---

## Running Locally

### Prerequisites

- Python 3.10+
- GCP credentials with access to the data projects:
  ```bash
  gcloud auth application-default login
  ```
- Install dependencies (using the repo-level venv):
  ```bash
  source <REPO_ROOT>/.venv/bin/activate
  pip install -r requirements.txt
  ```

### Quick Start (Local Dev)

The fastest way to run locally using the repo-level venv:

```bash
source <REPO_ROOT>/.venv/bin/activate
cd <REPO_ROOT>/WB_exp/WB_Metadata_Creator
python app.py --port=7870
```

This uses the default local metadata path (`../../product_mgmnt/Metadata/Metadata JSON for Demo/JSON Metadata`) for cross-referencing existing metadata. Open `http://localhost:7870`.

> **Note:** `start.sh` is for Docker/Workbench deployment — for local dev, run `app.py` directly.

### Full Mode (with BigQuery + GCS Output)

```bash
python app.py \
  --project=<BILLING_PROJECT_ID> \
  --data-project <DATA_PROJECT_1> <DATA_PROJECT_2> \
  --json-dir gs://<EXISTING_METADATA_BUCKET> \
  --output-bucket=<OUTPUT_GCS_BUCKET> \
  --port=7870
```

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

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio multi-tab UI — entry point |
| `models.py` | Shared data classes (BQColumnInfo, BQTableInfo, StudyConfig, etc.) |
| `bq_profiler.py` | BQ schema discovery + 4-phase data profiling |
| `fhir_builder.py` | Deterministic FHIR JSON assembly (SD, VS, CS, TermBundle, DataProfile, Measure) |
| `fhir_llm.py` | LLM-based generation (StructureDef, ValueSet, ConceptMap, column defs) |
| `gcs_utils.py` | GCS bucket/folder discovery |
| `fhir_generator.py` | Re-export shim (backwards-compatible imports) |
| `fhir_validator.py` | Input + JSON validation pipeline (LLM-as-Judge) |
| `sensitivity.py` | Sensitivity classification (29-code DS4P vocabulary) |
| `prompt_engine.py` | LLM prompt helpers |
| `metadata_loader.py` | Parses existing FHIR JSONs for cross-reference |
| `bq_executor.py` | BigQuery execution utilities |
| `agent.py` | LangGraph agent for LLM interactions |
| `start.sh` | Container startup script (for Workbench deployment) |
| `Dockerfile` | Container image definition |

---

## Deploying to Workbench

The app is deployed as a **Workbench custom cloud app**.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Auto-detected from GCE metadata on Workbench |
| `DATA_PROJECT_IDS` | Space-separated BQ data project IDs |
| `METADATA_SOURCE` | GCS URI or local path for existing FHIR metadata (cross-reference) |
| `OUTPUT_GCS_BUCKET` | GCS bucket for saving generated metadata |
