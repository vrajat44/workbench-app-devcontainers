# WB Data Explorer

**Natural language data exploration for Verily Workbench** — researchers ask questions in plain English, the app converts them to BigQuery SQL, executes the query, and returns results. No SQL knowledge required.

Built with **Gradio** (chat UI) + **LangGraph** (agentic loop) + **Gemini 2.5 Pro** (LLM) + **FHIR metadata** (semantic context).

---

## Why This Exists

Without rich metadata, every layer of data exploration breaks down:

1. **Opaque schemas → metadata provides meaning** — Column names like `PHQTOT` or `AUDITC_SCORES` are meaningless on their own. Metadata supplies the semantic context: *"PHQ-9 Total Score, 0–27 scale, self-reported, LOINC 44261-6."* Without it, both humans and LLMs are guessing.

2. **LLM hallucination → metadata provides grounding** — An LLM without metadata invents table names, guesses column meanings, and produces plausible-looking SQL that silently returns wrong results. Rich metadata constrains the LLM to only reference tables, columns, join keys, and relationships that actually exist.

3. **Cross-study silos → metadata provides connectivity** — Data spanning multiple GCP projects and datasets has no inherent discoverability. Metadata captures which studies exist, what instruments they used, how tables relate, and which identifiers link across boundaries — enabling cross-study queries that would otherwise require institutional knowledge.

4. **The SQL barrier → metadata enables natural language** — Writing correct BigQuery SQL with fully-qualified table names, proper joins, and cross-project references is a non-trivial skill. Metadata gives the LLM everything it needs — primary keys, structural links, granularity, data volume — to generate correct SQL from a plain English question.

**WB Data Explorer is the proof point**: when metadata is rich enough, an LLM can reason about data the way a domain expert would.

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Gradio Chat UI  │────▶│  LangGraph Agent │────▶│  BigQuery Executor   │
│   (app.py)       │◀────│  (agent.py)      │◀────│  (bq_executor.py)    │
└──────────────────┘     └────────┬─────────┘     └──────────────────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
              ┌──────▼──────┐          ┌───────▼───────┐
              │ Prompt       │          │ Metadata      │
              │ Engine       │          │ Loader        │
              └──────────────┘          └───────┬───────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │                       │
                              GCS Bucket              Local JSON dir
                          (production)              (dev / testing)
```

**Flow:**

1. **Metadata loads at startup** — FHIR StructureDefinition JSONs are parsed from GCS (or a local directory). Each JSON describes one table: what it contains, what each column means, how it relates to other tables, and what sensitivity labels apply.

2. **MeasureReport data profiles attach** — Companion JSON files provide actual row counts and physical sizes from BigQuery. This lets the LLM answer dataset fitness questions (*"is this table large enough for my study?"*) and apply appropriate query guardrails for high-volume tables (e.g., 413M-row RNAseq data).

3. **FHIR → BigQuery resolution** — Logical table names from metadata (e.g., `bhs.crf.PHQ9`) are matched against actual BigQuery tables at runtime, producing fully-qualified names (e.g., `project.crf.PHQ9`).

4. **Prompt engine builds context** — All metadata is injected into the LLM system prompt: table descriptions, column definitions, join rules, cross-study boundaries, and large-table warnings. Everything the LLM needs to write correct SQL is derived from metadata — nothing is hardcoded.

5. **User asks a question** → LangGraph agent calls Gemini 2.5 Pro → LLM generates SQL → BigQuery executes it → results are returned with explanation.

6. **Error recovery** — If a query fails, the executor sends the error back to the LLM for a fix-and-retry cycle (up to 3 attempts).

---

## Metadata-Driven Intelligence

### The 4-Level Metadata Framework

The app's intelligence comes from a structured metadata hierarchy. Each level adds context that helps both humans and AI agents understand data:

| Level | Scope | What It Captures | FHIR Resource | Status |
|:-----:|-------|-------------------|---------------|--------|
| **L1** | Organization / Program | Domain archetype, hierarchy, environment | `Organization` | Planned |
| **L2** | Study / Dataset | Business intent, study description, ownership, component inventory | Study-level StructureDefinition | Planned |
| **L3** | Table / Asset | Entity definition, granularity, primary keys, structural links, compliance | `StructureDefinition` | ✅ Implemented |
| **L4** | Column / Field | Semantic definition, concept bindings (LOINC, SNOMED, ICD-10), security labels, measurement method, value sets | `ElementDefinition` | ✅ Implemented |

**The app currently operates on L3 + L4 metadata**, with `MeasureReport` resources providing quantitative data characteristics (row counts, physical sizes). This gives the LLM enough context to:

- Know that `PHQTOT` is the "PHQ-9 Total Score (0–27 scale, sum of 9 items)"
- Know that `USUBJID` is the join key across all BHS tables
- Know that `presco.rnaseq` has 413M rows and must always be filtered
- Know that BHS and PRESCO use different participant identifiers and can't be directly linked
- Know that `putative_cohort` is PHI and `STUDYID` is non-sensitive

### Why FHIR StructureDefinitions (Not Flattened CSVs)

Early experimentation with flattened SQL-on-FHIR CSV representations of metadata showed that approach to be error-prone for LLM consumption:

- **Deeply nested columns** — Flattened CSVs produce column names like `extension.0.valueCodeableConcept.coding.0.code` that require `UNNEST` operations in BigQuery SQL. LLM-generated unnesting queries have a high failure rate due to incorrect nesting depth, wrong alias scoping, and ambiguous repeated fields.
- **Lossy representation** — Flattening discards structural relationships (e.g., which extension belongs to which element), making it ambiguous whether a security label applies to a column or to the table.
- **Brittle parsing** — Minor schema changes (a new extension, a reordered field) break CSV-based parsing logic.

Using FHIR JSONs directly preserves the full semantic structure. The `metadata_loader.py` parser extracts exactly what the LLM needs — definitions, types, bindings, mappings, sensitivity labels — without loss.

### Why Cortex → Workbench (Not Cortex → Analysis → Workbench)

The metadata delivery path matters. Going from **Cortex (FHIR registry) → Analysis systems (flattened SQL-on-FHIR) → Workbench** introduces an extra hop that:

- Adds a system to coordinate and maintain
- Requires flattening and re-inflation of FHIR structures, which is lossy
- Creates a sync point that can drift

The leaner path is **Cortex → Workbench directly**: FHIR StructureDefinitions are exported from Cortex, stored in GCS, and consumed natively by the app. Fewer systems, less coordination, and the metadata arrives intact.

### What a StructureDefinition Gives the LLM

Each table's metadata provides (example from BHS PHQ-9):

| Metadata | Value | How the LLM Uses It |
|----------|-------|---------------------|
| **Table description** | *"PHQ-9 depression screening scores per participant per visit"* | Understands what the table is about |
| **Granularity** | *"One record per participant per study visit"* | Knows the grain for aggregation |
| **Primary key** | `USUBJID + VISIT` | Knows how to join and deduplicate |
| **Structural links** | `→ bhs.admin.COEVAL`, `→ bhs.crf.GAD7` | Knows which tables can be joined |
| **Column: PHQTOT** | *"PHQ-9 Total Score — sum of all 9 items (0–27)"* | Can write `WHERE PHQTOT >= 10` for "moderate depression" |
| **Concept binding** | LOINC `44261-6` | Knows this is a standard depression instrument |
| **Security label** | `Non-sensitive` | Knows this column is safe to display |
| **Measurement method** | `Self-reported` | Can explain data provenance to the researcher |
| **VFIG mapping** | `QuestionnaireResponse.item.answer` | Links to standard clinical FHIR concepts |
| **Row count** | `1,489` (from MeasureReport) | Knows this is a small table, no special handling needed |

---

## Benchmark Results

Tested against **36 context-free questions** across 5 difficulty levels (simple exploration → ambiguous edge cases):

| Metric | Result | Notable Gap |
|--------|--------|-------------|
| Questions answered | **36 / 36** (100%) | — |
| SQL generated + executed | **35 / 36** (97.2%) | Q4 ("what data is available?") answered from metadata without SQL — correct behavior for a catalog question |
| SQL execution success | **35 / 35** (100%) | — |
| Correct tables identified | **35 / 36** (97.2%) | Q16 cross-study mental health query was BHS-only — PRESCO lacks PHQ-9/GAD-7 equivalents (data gap, not LLM error) |
| All metadata tables tested | **15 / 15** (100%) | — |

| Difficulty | Description | Success | Avg Time |
|:-----:|-------------|:-------:|:--------:|
| D1 | Simple exploration | 5/5 | 14.9s |
| D2 | Clinical queries | 10/10 | 16.0s |
| D3 | Relationships & joins | 8/8 | 23.5s |
| D4 | Cross-study cohort | 8/8 | 32.8s |
| D5 | Ambiguous / edge cases | 5/5 | 37.4s |

See `benchmark_report_v2.1.md` for detailed per-question results.

---

## Current Scope

### Studies & Tables

| Study | Tables | Rows | Description |
|-------|:------:|-----:|-------------|
| **BHS** (Baseline Health Study) | 12 | ~98K | Clinical CRFs (PHQ-9, GAD-7, EQ-5D, WHODAS, PFT), analysis scores (ASCVD, AUDIT-C, diagnoses), demographics, assays |
| **PRESCO** | 4 | ~413M | Participant info, cell subset frequencies, RNAseq gene expression (413M rows), subset code mappings |
| **Billing** | 2 | ~75K | PMPM claims and billing review (not in BQ currently) |

### Metadata Coverage

- **18 FHIR StructureDefinition JSONs** — one per table
- **18 MeasureReport data profiles** — row counts and physical sizes from BigQuery
- **VFIG mappings** — 80.2% column coverage linking BQ columns to standard FHIR concepts
- **Terminology** — CodeSystems and ValueSets for coded columns

---

## Key Design Decisions

### D1: FHIR JSONs as the Metadata Format
**Decision:** Use FHIR R4 StructureDefinition JSONs directly, not flattened CSVs or custom schemas.
**Rationale:** Preserves full semantic structure, avoids lossy flattening, and aligns with Cortex as the metadata registry. See *"Why FHIR StructureDefinitions"* above.

### D2: Two-Project Pattern for BigQuery
**Decision:** BQ jobs run in the workspace/billing project while querying data in separate data projects.
**Rationale:** Users typically don't have `bigquery.jobs.create` permission in data projects. The executor creates the BQ client with the billing project and queries cross-project.

### D3: GCS-Based Metadata Loading
**Decision:** Metadata JSONs live in a GCS bucket, not bundled in the Docker image.
**Rationale:** This pattern establishes a reusable approach for serving FHIR JSON metadata for any data asset cataloged by Exchange and other data producers. New tables can be added by uploading a JSON — no image rebuild needed. Multiple environments share the same metadata source, and the app auto-detects `gs://` URIs vs local paths.

### D4: Runtime FHIR → BigQuery Resolution
**Decision:** Logical table names from FHIR metadata are matched against actual BigQuery tables at startup.
**Rationale:** FHIR metadata uses study-prefixed names (e.g., `bhs.crf.PHQ9`) while BQ uses project-prefixed names (e.g., `project.crf.PHQ9`). Runtime resolution means metadata doesn't need to know which GCP project it's deployed in.

### D5: Study Name from Metadata Extension (L3 Stopgap for L2)
**Decision:** Added a `verily-study-name` extension to each table's StructureDefinition JSON to capture the study name (e.g., `"BHS"`, `"PRESCO"`).
**Tradeoff:** This is a per-table (L3) workaround — the study name is repeated across every table JSON. It solves the immediate need (the prompt engine was previously parsing study names from title strings using inferred domain qualifiers). **Long-term, L2 (study / data product) StructureDefinitions should provide a single source of truth** for study name, description, PI, and governance — L3 table definitions would reference the L2 study via canonical URL instead of carrying the name themselves.

| Concern | Current (L3 per-table extension) | With L2 Metadata |
|---------|-------------------------------|-------------------|
| Study name | ✅ Per-table extension | ✅ Single source of truth |
| Study description / purpose | ❌ Not captured | ✅ Study-level narrative |
| PI / data producer contact | ❌ Not captured | ✅ Study-level ownership |
| Cross-study governance rules | ❌ Not captured | ✅ Study-level policies |
| Rename / reorganize a study | Edit N files | Edit 1 file |

### D6: Row Counts from MeasureReport Data Profiles
**Decision:** Each table has a companion `MeasureReport` JSON containing actual row counts and physical sizes from BigQuery.
**Rationale:** The prompt engine previously relied on keyword matching (e.g., `"rnaseq"`, `"proteomics"`) to infer which tables were high-volume. This was imprecise — it flagged 5 tables as potentially large when only 1 (`presco.rnaseq`, 413M rows) actually needed the "always filter" warning. MeasureReport data profiles provide the real numbers, and the prompt engine now uses a threshold (`>100K rows`) instead of keyword inference.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio chat UI — entry point |
| `agent.py` | LangGraph agent with BigQuery tools |
| `prompt_engine.py` | Builds LLM system prompt from metadata |
| `metadata_loader.py` | Parses FHIR JSONs + MeasureReport profiles, resolves against BQ |
| `bq_executor.py` | BigQuery execution with LLM-assisted error retry |
| `benchmark.py` | Automated benchmarking against question sets |
| `start.sh` | Container startup script |
| `Dockerfile` | Container image definition |

---

## Limitations

**Workbench Platform:**
- **No deployed app logging** — Workbench custom apps lack SSH, serial port, and Cloud Logging access on VMs, making deployed app debugging trial-and-error.
- **Sharing requires workspace access** — Collaborators need workspace-level role grants to use the app; there's no simple URL-based sharing.
- **No incremental app updates** — Updating the deployed app requires deleting and recreating it; Workbench doesn't support in-place updates for custom apps.

**Memory & Infrastructure:**
- **No conversation persistence** — Chat history is held in memory; restarting the app loses all prior context.
- **Single-user sessions** — Conversation state is global, not per-user; concurrent users would overwrite each other's context.
- **System prompt scales with table count** — Currently ~86K chars for 18 tables. Adding significantly more tables could approach Gemini's context window limit.
- **LLM response capped at 4,096 tokens** — Complex multi-table queries with large result interpretations may be truncated.
- **No query cost guardrails** — No pre-execution cap on bytes scanned; a user could accidentally trigger expensive full-table scans on high-volume tables.

---

## Future Improvements and Impact

| Priority | Improvement | Impact | Addresses Limitation |
|:--------:|-------------|--------|----------------------|
| P0 | **Multi-turn refinement with triage** | A triage step where the agent interacts with the user to clarify vague questions before generating SQL — removes ambiguity upfront, reduces hallucination, and enables iterative refinement (e.g., *"now break that down by visit"*) | — |
| P0 | **Per-user session state** | Isolate conversation history per user so concurrent users don't overwrite each other's context | Single-user sessions |
| P1 | **Conversation persistence** | Store chat history in Firestore or Cloud SQL so sessions survive app restarts and users can resume prior conversations | No conversation persistence |
| P1 | **L2 metadata (study-level StructureDefinitions)** | Single source of truth for study name, description, PI, and governance — eliminates per-table repetition and gives the LLM richer study context for better cross-study reasoning | — |
| P1 | **Model evaluation (Flash Pro vs coding models)** | Compare Gemini 2.5 Flash Pro, coding-optimized models, and other LLMs on SQL accuracy, latency, and cost — select the best quality/speed tradeoff for production use | — |
| P1 | **Query cost guardrails** | Estimate bytes scanned via dry-run before execution; warn or block queries exceeding a configurable threshold | No query cost guardrails |
| P2 | **Streaming LLM responses** | Researchers see the LLM's reasoning in real time instead of waiting 15–35s for a complete response | — |
| P2 | **Increase max output tokens** | Raise the 4,096-token cap (or switch to a model with a larger output window) so complex multi-table answers aren't truncated | LLM response capped at 4,096 tokens |
| P2 | **Dynamic context windowing** | Summarize or selectively load metadata so the system prompt stays within context limits as the table catalog grows | System prompt scales with table count |
| P2 | **CSV download for query results** | One-click export of results for downstream analysis in R, Python, or Excel | — |
| P3 | **Automated MeasureReport generation** | Periodic BQ profiling keeps row counts and data characteristics fresh without manual updates | — |
| P3 | **Workbench platform improvements** | App-level log access and simpler sharing (URL-based, not workspace-role-based) would accelerate debugging and adoption | — |

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
  pip install -r requirements.txt
  ```

### Quick Start

```bash
python3 app.py \
  --project=<BILLING_PROJECT_ID> \
  --data-project <DATA_PROJECT_1> <DATA_PROJECT_2> \
  --json-dir gs://<METADATA_BUCKET> \
  --port=7860
```

Open `http://localhost:7860` and ask a question.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--project` | None | GCP project for billing (BQ jobs + Vertex AI) |
| `--data-project` | Same as `--project` | BQ project(s) where data tables live (space-separated) |
| `--json-dir` | `./metadata/` | Local path or `gs://` URI to FHIR metadata JSONs |
| `--llm-model` | `gemini-2.5-pro` | Vertex AI model name |
| `--port` | `7860` | Gradio server port |
| `--share` | `false` | Create a public Gradio share link |

### Modes

```bash
# Full mode — BQ + LLM (recommended)
python3 app.py --project=... --data-project ... --json-dir gs://... --port=7860

# Metadata-only — no BQ queries, for UI testing
python3 app.py --json-dir gs://... --port=7860

# Local metadata — for offline development
python3 app.py --project=... --data-project ... --json-dir /path/to/local/jsons/ --port=7860
```

---

## Deploying to Workbench

The app is deployed as a **Workbench custom cloud app** via a GitHub fork.

**Deployment repo**: [workbench-app-devcontainers](https://github.com/vrajat44/workbench-app-devcontainers) (`src/wb-data-explorer/`)

| Setting | Value |
|---------|-------|
| Repository | `https://github.com/vrajat44/workbench-app-devcontainers.git` |
| Branch | `master` |
| Folder path | `src/wb-data-explorer` |
| Port | `8080` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `METADATA_SOURCE` | GCS URI or local path for FHIR JSONs |
| `DATA_PROJECT_IDS` | Space-separated BQ data project IDs |
| `LLM_MODEL` | Vertex AI model (default: `gemini-2.5-pro`) |
| `GCP_PROJECT_ID` | Auto-detected from GCE metadata on Workbench |

### How Deployment Works

1. Workbench clones the GitHub repo and builds the Docker image
2. `start.sh` auto-detects the GCP project from the GCE metadata server
3. App loads metadata from GCS, resolves against BigQuery, starts serving on port 8080

### Adding New Tables

Upload a FHIR StructureDefinition JSON (and optionally a MeasureReport data profile) to the GCS metadata bucket → restart the app. No code changes needed.
