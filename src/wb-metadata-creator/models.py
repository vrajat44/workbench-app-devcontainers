"""
Shared data models for WB Metadata Creator.

Contains all dataclasses used across modules:
  - BQ schema models (BQColumnInfo, BQTableInfo)
  - Study configuration (StudyConfig)
  - Generation result (GenerationResult)
  - Profiling models (ColumnProfile, TableProfile)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── BigQuery Schema Models ────────────────────────────────────────────────────

@dataclass
class BQColumnInfo:
    """Column information from BigQuery INFORMATION_SCHEMA."""
    column_name: str
    data_type: str
    is_nullable: str  # "YES" or "NO"
    description: Optional[str] = None
    ordinal_position: int = 0


@dataclass
class BQTableInfo:
    """Table information from BigQuery INFORMATION_SCHEMA."""
    project_id: str
    dataset_id: str
    table_id: str
    columns: list[BQColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    table_type: str = "BASE TABLE"

    @property
    def fq_name(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"


# ── Study Configuration ──────────────────────────────────────────────────────

@dataclass
class StudyConfig:
    """Study-level configuration for metadata generation."""
    study_name: str  # e.g., "BHS", "PRESCO"
    compliance_zone: str = "HIPAA-covered"
    retention_years: int = 7
    schema_stability: str = "stable"
    domain_contact: str = ""
    confidentiality: str = "R"  # R=Restricted, N=Normal


# ── Generation Result ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Result from a FHIR metadata generation."""
    table_name: str
    structure_definition: Optional[dict] = None
    value_sets: list[dict] = field(default_factory=list)
    data_profile: Optional[dict] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.structure_definition is not None and self.error is None


# ── Profiling Models ──────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """Profiling statistics for a single column."""
    column_name: str
    null_count: int = 0
    null_percent: float = 0.0
    distinct_count: int = 0
    top_values: list[str] = field(default_factory=list)
    # Value counts for top values (value → count) — for accurate freq-distribution
    value_counts: Optional[dict[str, int]] = None
    # String length stats (for STRING/BYTES columns)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    # Numeric stats (for INT64/FLOAT64/NUMERIC columns)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median: Optional[float] = None
    stddev: Optional[float] = None


@dataclass
class TableProfile:
    """Profiling statistics for a table."""
    table_name: str
    total_rows: int = 0
    columns: dict[str, ColumnProfile] = field(default_factory=dict)


# ── Typed Column / Table Metadata ─────────────────────────────────────────────
# These replace the fragile DataFrame + dict approach.
# ColumnMetadata is the *single source of truth* for every column's metadata.
# The Gradio DataFrame is just a view — edits are written back here.

_VALID_MEASUREMENT_METHODS = frozenset({
    "self-reported", "calculated", "laboratory-measured",
    "clinician-observed", "device-collected", "extracted-from-ehr",
    "administrative",
})

# 8-column review table headers (must match Gradio table)
REVIEW_COLUMNS: list[str] = [
    "Column", "BQ Type", "Short Label", "Description",
    "Required", "Sensitivity", "Measurement", "Coded",
]


@dataclass
class ColumnMetadata:
    """
    All metadata for a single column — single source of truth.

    Replaces the old split between the 8-column review DataFrame,
    _column_enrichments dict, and ad-hoc dicts passed to the builder.
    """
    column_name: str
    bq_type: str
    fhir_type: str = "string"
    short_label: str = ""
    description: str = ""
    required: bool = False
    sensitivity: str = ""       # NONE / PHI / UID / P_BIRTHSEX / etc.
    measurement_method: str = ""
    coded: bool = False
    fhir_mapping: str = ""      # e.g. Patient.identifier

    # ── Rich semantic fields for gold-standard parity ──
    comment: str = ""               # Rich element comment (binding type, PK context, sensitivity justification)
    mapping_comment: str = ""       # FHIR mapping explanation (e.g., "Maps to FHIR Patient.identifier — study-specific subject ID")
    measurement_text: str = ""      # Rich measurement method text (e.g., "Self-reported by participant during enrollment")
    fixed_value: str = ""           # fixedString for constant columns (auto-populated when distinct_count == 1)

    # ── Conversion helpers ────────────────────────────────────────────

    def to_review_row(self) -> dict:
        """Convert to the 8-column review table row for Gradio."""
        return {
            "Column": self.column_name,
            "BQ Type": self.bq_type,
            "Short Label": self.short_label,
            "Description": self.description,
            "Required": "Yes" if self.required else "No",
            "Sensitivity": self.sensitivity if self.sensitivity not in ("NONE", "") else "",
            "Measurement": self.measurement_method,
            "Coded": "Yes" if self.coded else "No",
        }

    def update_from_review_row(self, row: dict) -> None:
        """Update from an edited review-table row (user edits via Gradio)."""
        self.short_label = str(row.get("Short Label", self.short_label) or "")
        self.description = str(row.get("Description", self.description) or "")
        self.required = str(row.get("Required", "No")).strip().lower() in ("yes", "true", "1")
        sens = str(row.get("Sensitivity", "")).strip().upper()
        self.sensitivity = sens if sens else ""
        self.measurement_method = str(row.get("Measurement", "")).strip()
        self.coded = str(row.get("Coded", "No")).strip().lower() in ("yes", "true", "1")

    def to_builder_dict(self) -> dict:
        """
        Convert to the dict format expected by fhir_builder._build_elements.

        Includes ALL fields (FHIR Type, FHIR Mapping) so the builder never
        needs to re-derive or look up enrichments from a side store.
        """
        return {
            "Column": self.column_name,
            "BQ Type": self.bq_type,
            "FHIR Type": "code" if self.coded else self.fhir_type,
            "Short Label": self.short_label,
            "Description": self.description,
            "Required": "Yes" if self.required else "No",
            "Sensitivity": self.sensitivity,
            "Measurement": self.measurement_method,
            "Coded": "Yes" if self.coded else "No",
            "FHIR Mapping": self.fhir_mapping,
            # Rich semantic fields
            "Comment": self.comment,
            "Mapping Comment": self.mapping_comment,
            "Measurement Text": self.measurement_text,
            "Fixed Value": self.fixed_value,
        }

    def apply_llm_output(self, llm: dict, bq_description: str = "") -> None:
        """
        Merge LLM-generated column metadata into this object.

        Applies validation rules:
          - Rejects garbage descriptions (< 10 chars or bare metadata values)
          - Validates measurement_method against allowed vocabulary
          - Preserves fhir_type and fhir_mapping from LLM
        """
        # Description — validate quality
        desc = str(llm.get("description", "")).strip()
        _garbage = {"yes", "no", "true", "false", "none", ""}
        if desc.lower() in _garbage or len(desc) < 10:
            # Keep existing (BQ description or empty) — don't overwrite with junk
            if bq_description and not self.description:
                self.description = bq_description
        else:
            self.description = desc

        # Short label
        label = str(llm.get("short_label", "")).strip()
        if label:
            self.short_label = label

        # Sensitivity
        sens = str(llm.get("sensitivity", "NONE")).strip().upper()
        self.sensitivity = sens if sens != "NONE" else ""

        # Measurement method — validate vocabulary
        meth = str(llm.get("measurement_method", "")).strip().lower()
        if meth and meth not in _VALID_MEASUREMENT_METHODS:
            meth = "administrative"
        self.measurement_method = meth

        # FHIR type — LLM semantic override
        ft = str(llm.get("fhir_type", "")).strip().lower()
        if ft:
            self.fhir_type = ft
            if ft == "code":
                self.coded = True

        # FHIR mapping
        fm = str(llm.get("fhir_mapping", "")).strip()
        if fm:
            self.fhir_mapping = fm

        # Comment — rich element comment
        comment = str(llm.get("comment", "")).strip()
        if comment:
            self.comment = comment

        # Mapping comment
        mapping_cmt = str(llm.get("mapping_comment", "")).strip()
        if mapping_cmt:
            self.mapping_comment = mapping_cmt

        # Measurement text
        meas_text = str(llm.get("measurement_text", "")).strip()
        if meas_text:
            self.measurement_text = meas_text

        # Fixed value (LLM can confirm/override auto-detected value)
        fixed = str(llm.get("fixed_value", "")).strip()
        if fixed:
            self.fixed_value = fixed

        # Coded — also set if LLM says code
        if llm.get("fhir_type") == "code" and not self.coded:
            self.coded = True


@dataclass
class TableMetadata:
    """
    Complete metadata for one table — single source of truth.

    Replaces the old triple: _review_data[name] (DataFrame),
    _table_meta[name] (dict), and _column_enrichments[name] (dict).
    """
    table_fq_name: str
    title: str = ""
    description: str = ""
    purpose: str = ""
    primary_key: str = ""
    columns: list[ColumnMetadata] = field(default_factory=list)

    # ── References to source data (consolidates separate globals) ──
    bq_table_info: Optional[BQTableInfo] = None       # BQ schema + row_count + size_bytes
    profile: Optional[TableProfile] = None             # Per-column profiling stats
    concept_maps: list[dict] = field(default_factory=list)  # LLM-generated ConceptMaps
    structural_links: list[str] = field(default_factory=list)  # Canonical URLs of related tables

    # ── Column lookup ─────────────────────────────────────────────────

    def get_column(self, name: str) -> Optional[ColumnMetadata]:
        """Find a column by name."""
        for c in self.columns:
            if c.column_name == name:
                return c
        return None

    # ── Auto-enrichment helpers ───────────────────────────────────────

    def auto_populate_fixed_values(self) -> None:
        """Set fixed_value on columns where profiling shows exactly 1 distinct value."""
        if not self.profile:
            return
        for col in self.columns:
            if col.fixed_value:
                continue  # Already set (by LLM or previously)
            cp = self.profile.columns.get(col.column_name)
            if cp and cp.distinct_count == 1 and cp.top_values:
                col.fixed_value = str(cp.top_values[0])

    def enrich_pk_comments(self) -> None:
        """If a column is part of primary_key, append PK context to its comment if missing."""
        if not self.primary_key:
            return
        pk_cols = [c.strip() for c in self.primary_key.replace("+", ",").split(",")]
        pk_cols = [c for c in pk_cols if c]
        for col in self.columns:
            if col.column_name in pk_cols:
                pk_mention = f"primary key ({self.primary_key})"
                if pk_mention.lower() not in col.comment.lower() and "primary key" not in col.comment.lower():
                    if col.comment:
                        col.comment = col.comment.rstrip(". ") + f". Part of the composite primary key ({self.primary_key})."
                    else:
                        col.comment = f"Part of the composite primary key ({self.primary_key})."

    # ── Conversion to review DataFrame (for Gradio) ──────────────────

    def to_review_dataframe(self):
        """Convert to a pandas DataFrame for the Gradio review table."""
        import pandas as pd
        rows = [col.to_review_row() for col in self.columns]
        return pd.DataFrame(rows, columns=REVIEW_COLUMNS)

    def update_from_review_dataframe(self, df) -> None:
        """Apply user edits from the Gradio DataFrame back to the model."""
        df_by_col = {}
        for _, row in df.iterrows():
            df_by_col[row["Column"]] = row
        for col_meta in self.columns:
            row = df_by_col.get(col_meta.column_name)
            if row is not None:
                col_meta.update_from_review_row(row)

    # ── Conversion for table-level meta dict ─────────────────────────

    def to_table_meta_dict(self) -> dict:
        """Dict for fhir_builder.build_structure_definition table_meta arg."""
        return {
            "title": self.title,
            "description": self.description,
            "purpose": self.purpose,
            "primary_key": self.primary_key,
            "structural_links": self.structural_links,
        }

    # ── Conversion for builder column list ───────────────────────────

    def to_builder_columns(self) -> list[dict]:
        """
        List of dicts for fhir_builder._build_elements.
        Includes FHIR Type and FHIR Mapping — no side-store needed.
        """
        return [col.to_builder_dict() for col in self.columns]

    # ── Apply LLM table-level output ─────────────────────────────────

    def apply_llm_table_meta(self, llm_table: dict) -> None:
        """Merge LLM-generated table metadata."""
        self.title = llm_table.get("title", self.title) or self.title
        self.description = llm_table.get("description", self.description) or self.description
        self.purpose = llm_table.get("purpose", self.purpose) or self.purpose
        self.primary_key = llm_table.get("primary_key", self.primary_key) or self.primary_key
