# WB Lineage Explorer

**Visual FHIR data lineage for Verily Workbench** — explore how FHIR resources flow through ingestion, enrichment, and workflow processes by querying Provenance tables in BigQuery. Supports both aggregate (high-level) and instance-level lineage tracing with interactive graph visualizations.

Built with **Gradio** (UI) + **Pyvis** (interactive network graphs) + **BigQuery** (Provenance queries) + **FHIR Provenance** (data source).

---

## Why This Exists

Understanding data lineage in a FHIR-based clinical data platform is critical for:

1. **Transparency** — Researchers and data stewards need to see how a resource (e.g., a Patient, Observation, or CarePlan) was created, what enrichment processes touched it, and what source data it originated from.

2. **Debugging data pipelines** — When a resource has unexpected values, lineage tracing reveals which activities (ingestion, enrichment, workflow) contributed to its current state.

3. **Compliance & audit** — Regulatory and governance requirements demand traceability from raw ingested data through to curated clinical data records (CDR).

4. **Cross-mirror comparison** — The same data flows through three mirrors (Landing → Operational → CDR), each with different levels of enrichment. Lineage visualization makes it easy to compare what happened at each stage.

**WB Lineage Explorer** makes this lineage visible and interactive — no SQL required.

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Gradio UI       │────▶│  BQ Lineage      │────▶│  BigQuery            │
│  (app.py)        │◀────│  (bq_lineage.py) │◀────│  Provenance Tables   │
└──────────────────┘     └────────┬─────────┘     └──────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Graph Builder  │
                         │ (lineage_graph  │
                         │     .py)        │
                         └────────┬────────┘
                                  │
                         Interactive Pyvis
                         Network Graph
```

**Flow:**

1. **User selects a dataset** — one of three FHIR mirrors in `prj-d-1v-ucd`:
   - `landing_fhir_mirror` — raw ingested FHIR resources
   - `operational_fhir_mirror` — enriched FHIR resources
   - `cdr_fhir_mirror` — curated clinical data records

2. **High-Level Lineage** — Aggregate query over the Provenance table groups all resource flows by `entity_type → activity → target_type`, producing a directed graph showing how resource types flow through activities at scale.

3. **Instance-Level Lineage** — Given a single FHIR Resource ID, the app auto-detects its resource type, then iteratively walks backward through the Provenance chain (multi-hop), building a complete lineage graph from the earliest source to the selected resource.

4. **Interactive visualization** — Pyvis renders the graph with left-to-right hierarchical layout (source → activity → target), color-coded nodes, hover tooltips, zoom, and drag.

---

## Two Views

### Tab 1: High-Level Lineage

Aggregate view showing how **resource types** flow through **activities** across the entire dataset.

```
Source Resources ──▶ Activities / Enrichments ──▶ Target Resources
   (circles)            (diamonds)                   (circles)
```

| Feature | Description |
|---------|-------------|
| **Full universe** | Click "Load High-Level Lineage" to see all resource flows |
| **Focus by resource type** | Select a target resource type (e.g., Patient, Observation) to zoom into its lineage |
| **Filter by FHIR profile** | After selecting a resource type, filter by its `meta.profile` (e.g., US Core Patient) to see only lineage for that specific profile |
| **Back to full universe** | One-click reset to the unfiltered view |
| **Summary panel** | Top activities and target resource types with counts |
| **Raw data table** | Expandable accordion with the underlying `entity_type → activity → target_type` rows |

**Profile filtering** works by joining the Provenance table against the actual resource table (e.g., `Patient`) on the typed ID field, then filtering by `meta.profile`. This means you can trace lineage for specific FHIR profiles — not just resource types.

### Tab 2: Instance-Level Lineage

Trace the **complete lineage chain** for a specific FHIR resource instance.

| Feature | Description |
|---------|-------------|
| **Single input** | Paste a FHIR Resource ID — the resource type is **auto-detected** |
| **Multi-hop tracing** | Iteratively walks backward through provenance chains (configurable depth, default 5) |
| **Full graph** | Shows every source resource, activity, and intermediate resource from origin to the selected target |
| **Color-coded nodes** | Resource types have distinct colors; activities are diamonds; the selected target has a gold border |
| **Legend** | Auto-generated legend showing all resource types and activities in the graph |
| **Summary panel** | Provenance record count, depth levels, distinct activities, and source entity types |

**How multi-hop works:** Starting from the target resource, the app queries Provenance for all records where that resource is a `target`. It then collects the `entity` references from those records and uses them as the next frontier — querying Provenance for records where those entities are targets. This continues iteratively up to the configured max depth, building the full backward lineage chain.

---

## Graph Layout

All graphs use **left-to-right hierarchical layout**, matching the natural data flow direction:

```
← Earlier in pipeline                    Later in pipeline →

  Source          Activity /           Target
  Resources       Enrichment           Resources
  (Level 0)       (Level 1)            (Level 2)

  [Patient] ──▶ [INGEST] ──────────▶ [Patient]
  [Binary]  ──▶ [enrichment:dq] ──▶ [Observation]
                [Action Engine] ───▶ [CarePlan]
```

**Node types:**
- **Circles** = FHIR resource instances or types
- **Diamonds** = Activities (ingestion, enrichment, workflow)
- **Gold-bordered circle** = The selected target resource (instance view)

**Edge width** scales with volume (high-level view) to show which flows carry the most data.

---

## Datasets

The app queries Provenance tables in **BigQuery dev-stable** (`prj-d-1v-ucd`):

| Dataset | Description | Use Case |
|---------|-------------|----------|
| `landing_fhir_mirror` | Raw FHIR resources as ingested | See what came in before any processing |
| `operational_fhir_mirror` | Enriched FHIR resources (data quality, patient priority, eligibility, etc.) | See the full enrichment chain |
| `cdr_fhir_mirror` | Curated clinical data records | See the final state after curation |

Provenance resources themselves are **hidden** from all graphs — only clinical/administrative resources and activities are shown.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio UI — two-tab interface, event wiring, handlers |
| `bq_lineage.py` | BigQuery queries against Provenance tables (high-level, instance, profile filtering) |
| `lineage_graph.py` | Pyvis graph construction (high-level and instance graphs, color palettes, layout) |
| `requirements.txt` | Python dependencies |
| `start.sh` | Container startup script (auto-detects GCP project) |
| `Dockerfile` | Container image definition |
| `docker-compose.yaml` | Workbench container orchestration |
| `devcontainer-template.json` | Workbench app registration metadata |

---

## Running Locally

### Prerequisites

- Python 3.10+
- GCP credentials with BigQuery access to `prj-d-1v-ucd`:
  ```bash
  gcloud auth application-default login
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Quick Start

```bash
python app.py --port=8080
```

Open `http://localhost:8080`.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `8080` | Gradio server port |
| `--share` | `false` | Create a public Gradio share link |

---

## Deploying to Workbench

The app is deployed as a **Workbench custom cloud app**.

**Deployment repo**: [workbench-app-devcontainers](https://github.com/vrajat44/workbench-app-devcontainers) (`src/wb-lineage-explorer/`)

| Setting | Value |
|---------|-------|
| Repository | `https://github.com/vrajat44/workbench-app-devcontainers.git` |
| Branch | `master` |
| Folder path | `src/wb-lineage-explorer` |
| Port | `8080` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Auto-detected from GCE metadata on Workbench |
| `GOOGLE_APPLICATION_CREDENTIALS` | Auto-set by Workbench (ADC) |

### How Deployment Works

1. Workbench clones the GitHub repo and builds the Docker image
2. `start.sh` auto-detects the GCP project from the GCE metadata server
3. App starts serving on port 8080, querying `prj-d-1v-ucd` Provenance tables

---

## Example Resource IDs for Testing

These can be pasted directly into the Instance-Level Lineage tab (dataset: `operational_fhir_mirror`):

| Resource Type | Resource ID | Provenance Records |
|---------------|-------------|--------------------|
| Patient | `074ce5e1-19cc-4ebf-baf4-b16ba889e51e` | 15 |
| Observation | `55A9004A-9DF4-48D9-99E6-24AC15D0DF14` | 15 |
| Encounter | `35aa4c40-f118-4f30-9fdc-78b0b4c34d48` | 15 |
| Condition | `3d3ebc2b-d197-4fce-ab79-06063fadb957` | 15 |
| Communication | `f964ac65-5521-4c0c-b2a1-fd51f8f3a259` | 15 |
| CarePlan | `747d94a5-5e90-4a35-b8a2-964b4fc4fbf6` | 20 |

The resource type is auto-detected — just paste the ID.

---

## Key Design Decisions

### D1: Iterative Multi-Hop (Not Recursive SQL)
**Decision:** Instance-level lineage uses iterative Python queries instead of BigQuery recursive CTEs.
**Rationale:** BigQuery's recursive CTE support has limitations with complex FHIR structures (nested `UNNEST` joins, typed ID `COALESCE` expressions). Iterative Python queries are more reliable, debuggable, and allow visited-node tracking to avoid cycles.

### D2: Provenance Resources Excluded from Graphs
**Decision:** `Provenance` resources are filtered out of both `target` and `entity` references in all queries and graphs.
**Rationale:** Provenance records are metadata about other resources — showing them as nodes creates visual noise and circular references. The graph should only show clinical/administrative resources connected by activities.

### D3: Profile Filtering via Resource Table Join
**Decision:** Profile filtering joins Provenance against the actual resource table (e.g., `Patient`) to check `meta.profile`.
**Rationale:** The Provenance table itself doesn't carry profile information — it only references targets by type and ID. To filter by profile, we must join to the resource table and check its `meta.profile` array. This is more expensive but gives accurate results.

### D4: Left-to-Right Hierarchical Layout
**Decision:** All graphs use Pyvis hierarchical layout with `direction: "LR"`.
**Rationale:** Data flows from left (sources/earlier pipeline stages) to right (targets/later stages), matching how engineers and data stewards naturally think about data pipelines. Top-to-bottom layout was confusing for lineage visualization.

### D5: Auto-Detection of Resource Type
**Decision:** Instance-level lineage only requires a Resource ID — the type is auto-detected.
**Rationale:** Users copying IDs from logs, dashboards, or other systems often don't know (or shouldn't need to know) the resource type. The app queries Provenance targets to resolve the type automatically, reducing friction.
