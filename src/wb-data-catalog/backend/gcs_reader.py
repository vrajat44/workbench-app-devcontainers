"""Read profiling JSON from GCS."""

from __future__ import annotations

import json
from typing import Any, Optional

from google.cloud import storage

from gcs_paths import SEM_FILENAME, TECH_FILENAME, parse_fq_table, profile_prefix  # noqa: F401


def _client(project_id: Optional[str] = None) -> storage.Client:
    return storage.Client(project=project_id) if project_id else storage.Client()


def blob_exists(bucket_name: str, object_path: str, project_id: Optional[str] = None) -> bool:
    client = _client(project_id)
    b = client.bucket(bucket_name.replace("gs://", ""))
    return b.blob(object_path).exists(client)


def read_json_if_exists(
    bucket_name: str,
    object_path: str,
    project_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    client = _client(project_id)
    b = client.bucket(bucket_name.replace("gs://", ""))
    blob = b.blob(object_path)
    if not blob.exists(client):
        return None
    return json.loads(blob.download_as_text())


def scan_profile_availability(
    bucket_name: str,
    data_project_id: str,
    billing_project_id: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """
    List all profiling/* under data_project and return
    fq_table -> {technical, semantic, business_name?, table_definition?}.
    """
    client = _client(billing_project_id)
    bucket = client.bucket(bucket_name.replace("gs://", ""))
    prefix = f"profiling/{data_project_id}/"
    out: dict[str, dict[str, Any]] = {}
    sem_blobs: list[tuple[str, Any]] = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix) :].strip("/")
        parts = rel.split("/")
        if len(parts) < 3:
            continue
        ds, tbl, fname = parts[0], parts[1], parts[2]
        fq = f"{data_project_id}.{ds}.{tbl}"
        entry = out.setdefault(fq, {"technical": False, "semantic": False})
        if fname == TECH_FILENAME:
            entry["technical"] = True
        elif fname == SEM_FILENAME:
            entry["semantic"] = True
            sem_blobs.append((fq, blob))

    for fq, blob in sem_blobs:
        try:
            data = json.loads(blob.download_as_text())
            out[fq]["business_name"] = data.get("business_name", "")
            out[fq]["table_definition"] = data.get("table_definition", "")
        except Exception:
            pass

    return out


def get_tech_path_for_fq(bucket_name: str, fq_table: str) -> str:
    p, d, t = parse_fq_table(fq_table)
    return f"gs://{bucket_name.replace('gs://', '')}/{profile_prefix(p, d, t)}{TECH_FILENAME}"


def get_sem_path_for_fq(bucket_name: str, fq_table: str) -> str:
    p, d, t = parse_fq_table(fq_table)
    return f"gs://{bucket_name.replace('gs://', '')}/{profile_prefix(p, d, t)}{SEM_FILENAME}"
