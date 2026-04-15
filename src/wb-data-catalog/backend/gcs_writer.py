"""Write profiling JSON to GCS."""

from __future__ import annotations

import json
from typing import Any, Optional

from google.cloud import storage

from gcs_paths import parse_fq_table, sem_object_path, tech_object_path


def upload_json(
    bucket_name: str,
    object_path: str,
    data: dict[str, Any],
    project_id: Optional[str] = None,
) -> str:
    client = storage.Client(project=project_id) if project_id else storage.Client()
    bucket = client.bucket(bucket_name.replace("gs://", ""))
    blob = bucket.blob(object_path)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type="application/json",
    )
    return f"gs://{bucket.name}/{object_path}"


def write_tech_profile(
    bucket_name: str,
    fq_table: str,
    profile_dict: dict[str, Any],
    project_id: Optional[str] = None,
) -> str:
    p, d, t = parse_fq_table(fq_table)
    path = tech_object_path(p, d, t)
    return upload_json(bucket_name, path, profile_dict, project_id)


def write_sem_profile(
    bucket_name: str,
    fq_table: str,
    profile_dict: dict[str, Any],
    project_id: Optional[str] = None,
) -> str:
    p, d, t = parse_fq_table(fq_table)
    path = sem_object_path(p, d, t)
    return upload_json(bucket_name, path, profile_dict, project_id)
