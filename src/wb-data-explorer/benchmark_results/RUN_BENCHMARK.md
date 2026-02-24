# How to Run the Benchmark

## Prerequisites

1. **Python virtual environment** with dependencies installed:
   ```bash
   cd WB_exp/WB_Data_Explorer
   source .venv/bin/activate        # or however your venv is set up
   pip install -r requirements.txt
   ```

2. **GCP authentication** — you need credentials that can access Vertex AI (Gemini) and BigQuery:
   ```bash
   gcloud auth application-default login
   ```

3. **Access to the metadata GCS bucket** and both BigQuery data projects.

---

## Run the Full Benchmark (36 questions)

```bash
cd WB_exp/WB_Data_Explorer

python benchmark.py \
  --project wb-glittery-carrot-8816 \
  --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
  --json-dir gs://metadata-json-wb-shrewd-papaya-8403
```

This will:
- Load 18 FHIR StructureDefinition + MeasureReport JSONs from GCS
- Resolve them against live BigQuery tables (15 tables across 2 projects)
- Run all 36 questions through the LangGraph agent (Gemini 2.5 Pro)
- Save structured results to `benchmark_results/benchmark_results_<timestamp>.json`
- Print a human-readable summary to stdout

**Expected runtime:** ~15 minutes (average ~24s per question).

---

## Useful Options

### Run a subset of questions

```bash
python benchmark.py \
  --project wb-glittery-carrot-8816 \
  --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
  --json-dir gs://metadata-json-wb-shrewd-papaya-8403 \
  --questions 1 6 11 16 21
```

### Dry run (metadata analysis only — no LLM or BigQuery calls)

```bash
python benchmark.py \
  --json-dir gs://metadata-json-wb-shrewd-papaya-8403 \
  --dry-run
```

### Resume a previous interrupted run

```bash
python benchmark.py \
  --project wb-glittery-carrot-8816 \
  --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
  --json-dir gs://metadata-json-wb-shrewd-papaya-8403 \
  --resume
```

### Custom output file

```bash
python benchmark.py \
  --project wb-glittery-carrot-8816 \
  --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
  --json-dir gs://metadata-json-wb-shrewd-papaya-8403 \
  --output benchmark_results/my_run.json
```

### Use local metadata JSONs (faster, no GCS access needed)

```bash
python benchmark.py \
  --project wb-glittery-carrot-8816 \
  --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
  --json-dir /path/to/local/JSON\ Metadata/
```

### Use a different LLM model

```bash
python benchmark.py \
  --project wb-glittery-carrot-8816 \
  --data-project wb-beamish-acorn-6393 wb-glittery-carrot-8816 \
  --json-dir gs://metadata-json-wb-shrewd-papaya-8403 \
  --llm-model gemini-2.0-flash
```

---

## Output

| Output | Location |
|--------|----------|
| Structured JSON results | `benchmark_results/benchmark_results_<timestamp>.json` |
| Human-readable summary | stdout (pipe to file with `> run.log 2>&1` if needed) |

The JSON file contains per-question details: generated SQL, execution status, row counts, tables/columns used from metadata, LLM response text, and timing.

---

## Generating the Report

The benchmark report (`benchmark_report_v2.1.md`) was written by analyzing the JSON results file. To regenerate or create a new version:

1. Run the benchmark (above) to produce a fresh `benchmark_results_<timestamp>.json`
2. Review the JSON for per-question SQL, metadata usage, and timing
3. The report structure follows this outline:
   - **Executive Summary** — key metrics, metadata-driven examples
   - **How Metadata Drives Query Generation** — pipeline diagram, StructureDefinition fields used
   - **Results Summary by Difficulty Level** — D1–D5 aggregate table
   - **Detailed Results Per Question** — per-difficulty sections, each with a summary table then per-question details including Metadata → SQL Mapping tables
   - **Metadata Utilization Deep Dive** — table coverage, field analysis, join patterns, clinical concept mapping
   - **Scoring Summary** — estimated scores by criteria and difficulty level
   - **Observations and Opportunities** — what works, what to improve
   - **Appendix A** — 15-table metadata catalog
   - **Appendix B** — environment details
