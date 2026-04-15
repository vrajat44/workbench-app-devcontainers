"""GCS object paths for profiling outputs."""

from __future__ import annotations


TECH_FILENAME = "tech_profile.json"
SEM_FILENAME = "semantic_profile.json"


def profile_prefix(project_id: str, dataset_id: str, table_id: str) -> str:
    """Prefix ending with / for one table's profiling folder."""
    return f"profiling/{project_id}/{dataset_id}/{table_id}/"


def tech_object_path(project_id: str, dataset_id: str, table_id: str) -> str:
    return f"{profile_prefix(project_id, dataset_id, table_id)}{TECH_FILENAME}"


def sem_object_path(project_id: str, dataset_id: str, table_id: str) -> str:
    return f"{profile_prefix(project_id, dataset_id, table_id)}{SEM_FILENAME}"


def parse_fq_table(fq: str) -> tuple[str, str, str]:
    parts = fq.split(".", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid fully-qualified table: {fq}")
    return parts[0], parts[1], parts[2]
