"""
FHIR Metadata Generator — Re-export Shim.

This module re-exports everything from the modular sub-packages so that
existing imports like ``from fhir_generator import X`` continue to work
without changes.

Actual implementations live in:
  - models.py          — Data classes (BQColumnInfo, BQTableInfo, StudyConfig, etc.)
  - bq_profiler.py     — BigQuery discovery & data profiling
  - fhir_builder.py    — Deterministic FHIR JSON builders (no LLM)
  - fhir_llm.py        — LLM-based generation (StructureDef, ValueSet, ConceptMap, etc.)
  - gcs_utils.py       — GCS bucket/folder discovery
"""

from __future__ import annotations

# ── models.py ─────────────────────────────────────────────────────────────────
from models import (  # noqa: F401
    BQColumnInfo,
    BQTableInfo,
    StudyConfig,
    GenerationResult,
    ColumnProfile,
    TableProfile,
    ColumnMetadata,
    TableMetadata,
    REVIEW_COLUMNS,
)

# ── bq_profiler.py ────────────────────────────────────────────────────────────
from bq_profiler import (  # noqa: F401
    discover_bq_datasets,
    discover_bq_tables,
    format_bq_schema_for_prompt,
    profile_bq_table,
    _NUMERIC_BQ_TYPES,
)

# ── fhir_builder.py ──────────────────────────────────────────────────────────
from fhir_builder import (  # noqa: F401
    map_bq_type_to_fhir,
    build_structure_definition,
    build_value_set,
    build_code_system,
    build_terminology_bundle,
    build_terminology_bundle_with_concept_maps,
    build_concept_map,
    generate_data_profile,
    build_measure_definition,
)

# ── fhir_llm.py ──────────────────────────────────────────────────────────────
from fhir_llm import (  # noqa: F401
    generate_structure_definition,
    generate_value_sets,
    generate_column_definitions,
    generate_concept_maps,
    _parse_concept_map_response,
    generate_all_metadata,
    summarize_generation_results,
)

# ── gcs_utils.py ──────────────────────────────────────────────────────────────
from gcs_utils import (  # noqa: F401
    discover_gcs_buckets,
    discover_gcs_folders,
)
