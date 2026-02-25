"""
Google Cloud Storage discovery utilities.

Provides bucket and folder listing for GCS-based metadata storage.
"""

from __future__ import annotations


def discover_gcs_buckets(project_id: str) -> list[str]:
    """List GCS buckets accessible in the given project."""
    from google.cloud import storage
    try:
        client = storage.Client(project=project_id)
        return sorted([b.name for b in client.list_buckets()])
    except Exception as e:
        print(f"⚠ Could not list buckets for {project_id}: {e}")
        return []


def discover_gcs_folders(bucket_name: str, prefix: str = "") -> list[str]:
    """
    List top-level 'folders' (common prefixes) in a GCS bucket.

    Args:
        bucket_name: Bucket name (no gs:// prefix).
        prefix: Optional prefix to list within (e.g. "metadata/").

    Returns:
        List of folder paths (e.g. ["BHS/", "PRESCO/", "CDC_Natality/"]).
    """
    from google.cloud import storage
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name.replace("gs://", ""))

        # Use delimiter to get "folder" prefixes only
        iterator = client.list_blobs(
            bucket,
            prefix=prefix if prefix else None,
            delimiter="/",
        )
        # Consume the iterator to populate prefixes
        _ = list(iterator)

        folders = sorted(iterator.prefixes)
        return folders
    except Exception as e:
        print(f"⚠ Could not list folders in {bucket_name}/{prefix}: {e}")
        return []
