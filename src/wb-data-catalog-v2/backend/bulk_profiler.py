"""
Bulk profiling manager.

Manages concurrent profiling of multiple tables with:
- Configurable concurrency (tech: 6 workers, semantic: 3 workers)
- Pipeline mode: tech -> semantic per table (no global barrier)
- Skip already-profiled tables
- Per-table status tracking with errors/warnings
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, Optional

from verily_profiler import (
    profile_technical,
    profile_semantic,
    write_tech_profile,
    write_sem_profile,
    read_registry,
    write_registry,
    scan_profile_availability,
    BQTableInfo,
)
from verily_profiler.storage import (
    parse_fq_table, tech_object_path, sem_object_path,
    read_json_if_exists, read_catalog_context, regenerate_catalog_context,
    read_sem_profile, read_tech_profile, upload_json,
)
from verily_profiler.llm import detect_available_model
from verily_profiler.models import TechTableProfile, TechColumnProfile, ValidationResult, TerminologyRegistry

TECH_CONCURRENCY = 6
SEM_CONCURRENCY = 3  # 3 concurrent semantic jobs works well with Gemini quota.


@dataclass
class TableJobStatus:
    fq_table: str
    tech_status: str = "pending"       # pending | skipped | running | done | failed
    sem_status: str = "pending"
    tech_error: str = ""
    sem_error: str = ""
    tech_duration: float = 0.0
    sem_duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fq_table": self.fq_table,
            "technical": self.tech_status,
            "semantic": self.sem_status,
            "tech_error": self.tech_error,
            "sem_error": self.sem_error,
            "tech_duration_s": round(self.tech_duration, 1),
            "sem_duration_s": round(self.sem_duration, 1),
        }


@dataclass
class BatchStatus:
    batch_id: str
    mode: str                          # technical | semantic | both
    total: int = 0
    tables: list[TableJobStatus] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"            # running | completed | failed
    logs: list[dict] = field(default_factory=list)

    def _log(self, level: str, msg: str):
        """Append a timestamped log entry. Thread-safe (list.append is atomic in CPython)."""
        self.logs.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "msg": msg,
        })

    def summary(self) -> dict:
        tech_done = sum(1 for t in self.tables if t.tech_status == "done")
        tech_failed = sum(1 for t in self.tables if t.tech_status == "failed")
        tech_skipped = sum(1 for t in self.tables if t.tech_status == "skipped")
        tech_running = sum(1 for t in self.tables if t.tech_status == "running")
        sem_done = sum(1 for t in self.tables if t.sem_status == "done")
        sem_failed = sum(1 for t in self.tables if t.sem_status == "failed")
        sem_skipped = sum(1 for t in self.tables if t.sem_status == "skipped")
        sem_running = sum(1 for t in self.tables if t.sem_status == "running")

        errors = [
            {"table": t.fq_table, "phase": "technical", "error": t.tech_error}
            for t in self.tables if t.tech_error
        ] + [
            {"table": t.fq_table, "phase": "semantic", "error": t.sem_error}
            for t in self.tables if t.sem_error
        ]

        warnings: list[dict] = []
        for t in self.tables:
            if t.tech_status == "skipped":
                warnings.append({"table": t.fq_table, "phase": "technical", "message": "Already profiled — skipped"})
            if t.sem_status == "skipped":
                warnings.append({"table": t.fq_table, "phase": "semantic", "message": "Already profiled — skipped"})

        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "mode": self.mode,
            "total": self.total,
            "technical": {"done": tech_done, "failed": tech_failed, "skipped": tech_skipped, "running": tech_running},
            "semantic": {"done": sem_done, "failed": sem_failed, "skipped": sem_skipped, "running": sem_running},
            "errors": errors,
            "warnings": warnings,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tables": [t.to_dict() for t in self.tables],
            "logs": self.logs,
        }


class BulkProfileManager:
    def __init__(self):
        self._lock = Lock()
        self._batches: dict[str, BatchStatus] = {}

    def get_batch(self, batch_id: str) -> Optional[BatchStatus]:
        with self._lock:
            return self._batches.get(batch_id)

    def start_batch(
        self,
        tables: list[str],
        mode: str,
        bucket: str,
        billing_project: str,
        data_project: str,
        model_name: Optional[str],
        force: bool = False,
        terminology_domains: Optional[list[str]] = None,
        doc_context: str = "",
        run_join_inference: bool = False,
    ) -> str:
        batch_id = str(uuid.uuid4())[:12]
        batch = BatchStatus(
            batch_id=batch_id,
            mode=mode,
            total=len(tables),
            tables=[TableJobStatus(fq_table=fq) for fq in tables],
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._batches[batch_id] = batch

        import threading
        t = threading.Thread(
            target=self._run_batch,
            args=(batch, bucket, billing_project, data_project, model_name, force,
                  terminology_domains, doc_context, run_join_inference),
            daemon=True,
        )
        t.start()
        return batch_id

    def _run_batch(
        self,
        batch: BatchStatus,
        bucket: str,
        billing_project: str,
        data_project: str,
        model_name: Optional[str],
        force: bool = False,
        terminology_domains: Optional[list[str]] = None,
        doc_context: str = "",
        run_join_inference: bool = False,
    ):
        try:
            batch._log("info", f"Starting {batch.mode} profiling for {batch.total} tables")

            profile_index = {}
            try:
                profile_index = scan_profile_availability(bucket, data_project, billing_project_id=billing_project)
            except Exception:
                pass

            gemini_model = None
            if batch.mode in ("semantic", "both"):
                gemini_model = model_name or detect_available_model(billing_project)

            registry = None
            if batch.mode in ("semantic", "both"):
                try:
                    registry = read_registry(bucket, data_project, billing_project_id=billing_project)
                except Exception:
                    registry = TerminologyRegistry()

            neighbor_ctx = None
            if batch.mode in ("semantic", "both"):
                try:
                    neighbor_ctx = read_catalog_context(bucket, data_project, billing_project_id=billing_project)
                    if neighbor_ctx:
                        logger.info(f"  Bulk: loaded neighbor context ({len(neighbor_ctx)} chars)")
                except Exception:
                    pass

            # Merge doc_context into neighbor context
            if doc_context:
                if neighbor_ctx:
                    neighbor_ctx = neighbor_ctx + "\n\n--- User-supplied documentation ---\n" + doc_context
                else:
                    neighbor_ctx = doc_context

            if batch.mode == "technical":
                self._run_technical_batch(batch, bucket, billing_project, data_project, profile_index, force)
            elif batch.mode == "semantic":
                self._run_semantic_batch(batch, bucket, billing_project, data_project, profile_index, gemini_model, registry, neighbor_ctx, force, terminology_domains)
            elif batch.mode == "both":
                self._run_pipeline_batch(batch, bucket, billing_project, data_project, profile_index, gemini_model, registry, neighbor_ctx, force, terminology_domains)

            if registry is not None and batch.mode in ("semantic", "both"):
                try:
                    write_registry(bucket, data_project, registry, billing_project_id=billing_project)
                except Exception as e:
                    logger.info(f"  Bulk: registry save failed: {e}")

            try:
                regenerate_catalog_context(bucket, data_project, billing_project_id=billing_project)
                try:
                    from chat_handler import invalidate_context_cache
                    invalidate_context_cache(data_project, bucket)
                except Exception:
                    pass
            except Exception as e:
                logger.info(f"  Bulk: catalog context regen failed: {e}")

            if batch.mode in ("semantic", "both"):
                self._run_second_pass(batch, bucket, billing_project, data_project, gemini_model, registry)

            # Join inference pass
            if run_join_inference and batch.mode in ("semantic", "both"):
                self._run_join_inference(batch, bucket, billing_project, data_project)

            # Final summary log
            done_count = sum(1 for t in batch.tables if t.tech_status == "done" or t.sem_status == "done")
            failed_count = sum(1 for t in batch.tables if t.tech_status == "failed" or t.sem_status == "failed")
            batch._log("info", f"All {batch.total} tables profiled. {done_count} done, {failed_count} failed.")

        except Exception as e:
            batch._log("error", f"Batch crashed: {e}")
            logger.info(f"  Bulk batch {batch.batch_id} crashed: {e}")
        finally:
            batch.finished_at = datetime.now(timezone.utc).isoformat()
            _invalidate_catalog_cache()
            any_failed = any(t.tech_status == "failed" or t.sem_status == "failed" for t in batch.tables)
            batch.status = "completed" if not any_failed else "completed_with_errors"

    def _run_technical_batch(self, batch, bucket, billing_project, data_project, profile_index, force=False):
        batch._log("info", f"Starting technical profiling for {batch.total} tables")

        def _do_tech(job: TableJobStatus, idx: int):
            existing = profile_index.get(job.fq_table, {})
            if existing.get("technical") and not force:
                job.tech_status = "skipped"
                return
            job.tech_status = "running"
            batch._log("info", f"Technical profiling: {job.fq_table} ({idx}/{batch.total})")
            t0 = time.time()
            try:
                info = _load_table_info(job.fq_table, billing_project, data_project)
                result = profile_technical(info, billing_project=billing_project)
                write_tech_profile(bucket, job.fq_table, result, project_id=billing_project)
                job.tech_status = "done"
                col_count = len(result.columns) if hasattr(result, "columns") else 0
                duration = round(time.time() - t0, 1)
                batch._log("info", f"{job.fq_table}: tech done ({duration}s) — {col_count} columns")
            except Exception as e:
                job.tech_status = "failed"
                job.tech_error = str(e)[:200]
                batch._log("error", f"{job.fq_table}: technical failed — {str(e)[:200]}")
            job.tech_duration = time.time() - t0

        with ThreadPoolExecutor(max_workers=TECH_CONCURRENCY) as ex:
            futures = {ex.submit(_do_tech, t, i + 1): t for i, t in enumerate(batch.tables)}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

    def _run_semantic_batch(self, batch, bucket, billing_project, data_project, profile_index, model, registry, neighbor_ctx=None, force=False, terminology_domains=None):
        batch._log("info", f"Starting semantic profiling for {batch.total} tables")

        def _do_sem(job: TableJobStatus, idx: int):
            existing = profile_index.get(job.fq_table, {})
            if existing.get("semantic") and not force:
                job.sem_status = "skipped"
                return
            if not existing.get("technical"):
                job.sem_status = "failed"
                job.sem_error = "Technical profile required first"
                batch._log("error", f"{job.fq_table}: semantic failed — Technical profile required first")
                return
            job.sem_status = "running"
            batch._log("info", f"Semantic profiling: {job.fq_table} ({idx}/{batch.total})")
            t0 = time.time()
            try:
                p, d, t = parse_fq_table(job.fq_table)
                tech_json = read_json_if_exists(bucket, tech_object_path(p, d, t), billing_project)
                tech_prof = _dict_to_tech_profile(tech_json)
                sem, new_entries = profile_semantic(
                    tech_prof, model=model, project_id=billing_project,
                    registry=registry, context_text=neighbor_ctx,
                    terminology_domains=terminology_domains,
                )
                write_sem_profile(bucket, job.fq_table, sem, project_id=billing_project)
                if new_entries and registry is not None:
                    with self._lock:
                        for entry in new_entries:
                            registry.upsert(entry)
                job.sem_status = "done"
                domain = getattr(sem, "semantic_domain", None)
                domain_str = getattr(domain, "primary", "unknown") if domain else "unknown"
                duration = round(time.time() - t0, 1)
                batch._log("info", f"{job.fq_table}: semantic done ({duration}s) — domain: {domain_str}")
            except Exception as e:
                job.sem_status = "failed"
                job.sem_error = str(e)[:200]
                batch._log("error", f"{job.fq_table}: semantic failed — {str(e)[:200]}")
            job.sem_duration = time.time() - t0

        with ThreadPoolExecutor(max_workers=SEM_CONCURRENCY) as ex:
            futures = {ex.submit(_do_sem, t, i + 1): t for i, t in enumerate(batch.tables)}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

    def _run_pipeline_batch(self, batch, bucket, billing_project, data_project, profile_index, model, registry, neighbor_ctx=None, force=False, terminology_domains=None):
        """Tech then semantic per table, pipelined: semantic starts as soon as its tech is done."""
        import queue
        import threading

        batch._log("info", f"Starting pipeline (tech+semantic) profiling for {batch.total} tables")

        sem_queue: queue.Queue[TableJobStatus] = queue.Queue()
        sem_done = threading.Event()
        sem_counter = [0]  # mutable counter for semantic index tracking

        def _sem_worker():
            while True:
                try:
                    job = sem_queue.get(timeout=1)
                except queue.Empty:
                    if sem_done.is_set() and sem_queue.empty():
                        break
                    continue
                if job is None:
                    break

                existing = profile_index.get(job.fq_table, {})
                if existing.get("semantic") and not force:
                    job.sem_status = "skipped"
                    continue

                sem_counter[0] += 1
                idx = sem_counter[0]
                job.sem_status = "running"
                batch._log("info", f"Semantic profiling: {job.fq_table} ({idx}/{batch.total})")
                t0 = time.time()
                try:
                    p, d, t = parse_fq_table(job.fq_table)
                    tech_json = read_json_if_exists(bucket, tech_object_path(p, d, t), billing_project)
                    if not tech_json:
                        job.sem_status = "failed"
                        job.sem_error = "Technical profile not found after profiling"
                        batch._log("error", f"{job.fq_table}: semantic failed — Technical profile not found after profiling")
                        continue
                    tech_prof = _dict_to_tech_profile(tech_json)
                    sem, new_entries = profile_semantic(
                        tech_prof, model=model, project_id=billing_project,
                        registry=registry, context_text=neighbor_ctx,
                        terminology_domains=terminology_domains,
                    )
                    write_sem_profile(bucket, job.fq_table, sem, project_id=billing_project)
                    if new_entries and registry is not None:
                        with self._lock:
                            for entry in new_entries:
                                registry.upsert(entry)
                    job.sem_status = "done"
                    domain = getattr(sem, "semantic_domain", None)
                    domain_str = getattr(domain, "primary", "unknown") if domain else "unknown"
                    duration = round(time.time() - t0, 1)
                    batch._log("info", f"{job.fq_table}: semantic done ({duration}s) — domain: {domain_str}")
                except Exception as e:
                    job.sem_status = "failed"
                    job.sem_error = str(e)[:200]
                    batch._log("error", f"{job.fq_table}: semantic failed — {str(e)[:200]}")
                job.sem_duration = time.time() - t0

        sem_threads = [threading.Thread(target=_sem_worker, daemon=True) for _ in range(SEM_CONCURRENCY)]
        for st in sem_threads:
            st.start()

        def _do_tech(job: TableJobStatus, idx: int):
            existing = profile_index.get(job.fq_table, {})
            if existing.get("technical") and not force:
                job.tech_status = "skipped"
                sem_queue.put(job)
                return
            job.tech_status = "running"
            batch._log("info", f"Technical profiling: {job.fq_table} ({idx}/{batch.total})")
            t0 = time.time()
            try:
                info = _load_table_info(job.fq_table, billing_project, data_project)
                result = profile_technical(info, billing_project=billing_project)
                write_tech_profile(bucket, job.fq_table, result, project_id=billing_project)
                job.tech_status = "done"
                col_count = len(result.columns) if hasattr(result, "columns") else 0
                duration = round(time.time() - t0, 1)
                batch._log("info", f"{job.fq_table}: tech done ({duration}s) — {col_count} columns")
                sem_queue.put(job)
            except Exception as e:
                job.tech_status = "failed"
                job.tech_error = str(e)[:200]
                job.sem_status = "failed"
                job.sem_error = "Skipped — technical profiling failed"
                batch._log("error", f"{job.fq_table}: technical failed — {str(e)[:200]}")
            job.tech_duration = time.time() - t0

        with ThreadPoolExecutor(max_workers=TECH_CONCURRENCY) as ex:
            futures = {ex.submit(_do_tech, t, i + 1): t for i, t in enumerate(batch.tables)}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

        sem_done.set()
        for st in sem_threads:
            st.join()


    def _run_join_inference(self, batch, bucket, billing_project, data_project):
        """Run cross-table join inference after semantic profiling completes."""
        from join_inference import infer_joins
        from verily_profiler.storage import read_sem_profile, read_tech_profile, upload_json

        batch._log("info", "Starting join inference across profiled tables")
        profiled_tables = [j.fq_table for j in batch.tables if j.sem_status == "done"]
        if len(profiled_tables) < 2:
            batch._log("info", "Join inference skipped — fewer than 2 profiled tables")
            return

        sem_profiles = []
        tech_profiles = []
        for fq in profiled_tables:
            try:
                sem = read_sem_profile(bucket, fq, project_id=billing_project)
                if sem:
                    sem_profiles.append(sem)
            except Exception:
                pass
            try:
                tech = read_tech_profile(bucket, fq, project_id=billing_project)
                if tech:
                    tech_profiles.append(tech)
            except Exception:
                pass

        try:
            join_map = infer_joins(sem_profiles, tech_profiles)
        except Exception as e:
            batch._log("error", f"Join inference failed: {e}")
            return

        total_joins = sum(len(paths) for paths in join_map.values())
        batch._log("info", f"Join inference found {total_joins} join paths across {len(join_map)} tables")

        for fq_table, join_paths in join_map.items():
            if not join_paths:
                continue
            try:
                sem = read_sem_profile(bucket, fq_table, project_id=billing_project)
                if not sem:
                    continue
                # Build a lookup of inferred joins by source column
                join_by_col: dict[str, list[str]] = {}
                for jp in join_paths:
                    src_col = jp.get("source_column", "")
                    target = jp.get("target", "")
                    if src_col and target:
                        join_by_col.setdefault(src_col, []).append(target)
                # Merge inferred join_paths into existing column metadata
                for col in sem.get("columns", []):
                    col_name = col.get("name", "")
                    inferred = join_by_col.get(col_name, [])
                    if inferred:
                        existing = col.get("join_paths") or []
                        merged = list(set(existing + inferred))
                        col["join_paths"] = merged
                p, d, t = parse_fq_table(fq_table)
                upload_json(bucket, sem_object_path(p, d, t), sem, billing_project)
            except Exception as e:
                batch._log("warn", f"Join inference update failed for {fq_table}: {e}")

    def _run_second_pass(self, batch, bucket, billing_project, data_project, model, registry):
        """Re-profile tables whose entity_anchor columns have empty join_paths."""
        try:
            updated_ctx = read_catalog_context(bucket, data_project, billing_project_id=billing_project)
            if not updated_ctx:
                logger.info("  Second pass: no catalog context available, skipping")
                return
        except Exception:
            return

        tables_needing_repass: list[TableJobStatus] = []
        for job in batch.tables:
            if job.sem_status != "done":
                continue
            try:
                sem_data = read_sem_profile(bucket, job.fq_table, project_id=billing_project)
                if not sem_data:
                    continue
                entity_anchor = sem_data.get("entity_anchor", "")
                if not entity_anchor:
                    continue
                columns = sem_data.get("columns", [])
                anchor_col = next((c for c in columns if c.get("name") == entity_anchor), None)
                if anchor_col and not anchor_col.get("join_paths"):
                    tables_needing_repass.append(job)
            except Exception:
                continue

        if not tables_needing_repass:
            logger.info("  Second pass: all entity_anchor columns already have join_paths, skipping")
            return

        logger.info(f"  Second pass: re-profiling {len(tables_needing_repass)} tables with updated context")
        for job in tables_needing_repass:
            try:
                p, d, t = parse_fq_table(job.fq_table)
                tech_json = read_json_if_exists(bucket, tech_object_path(p, d, t), billing_project)
                if not tech_json:
                    continue
                tech_prof = _dict_to_tech_profile(tech_json)
                sem, new_entries = profile_semantic(
                    tech_prof, model=model, project_id=billing_project,
                    registry=registry, context_text=updated_ctx,
                )
                write_sem_profile(bucket, job.fq_table, sem, project_id=billing_project)
                if new_entries and registry is not None:
                    with self._lock:
                        for entry in new_entries:
                            registry.upsert(entry)
                logger.info(f"    Re-profiled {job.fq_table}")
            except Exception as e:
                logger.info(f"    Second pass failed for {job.fq_table}: {e}")

        try:
            regenerate_catalog_context(bucket, data_project, billing_project_id=billing_project)
        except Exception:
            pass


bulk_manager = BulkProfileManager()


def _invalidate_catalog_cache():
    """Clear all profiling-related caches so next request gets fresh data."""
    try:
        from main import _invalidate_profiling_caches
        _invalidate_profiling_caches()
    except Exception:
        try:
            from main import _catalog_cache
            _catalog_cache.clear()
        except Exception:
            pass


def _load_table_info(fq_table: str, billing_project: str, data_project: str) -> BQTableInfo:
    from verily_profiler import discover_tables
    p, d, t = parse_fq_table(fq_table)
    tables = discover_tables(p, d, billing_project=billing_project)
    for tbl in tables:
        if tbl.table_id == t:
            return tbl
    raise RuntimeError(f"Table {fq_table} not found in BQ")


def _dict_to_tech_profile(data: dict) -> TechTableProfile:
    cols = []
    for c in data.get("columns", []):
        cols.append(TechColumnProfile(
            column_name=c.get("name", c.get("column_name", "")),
            data_type=c.get("data_type", ""),
            nullable=c.get("nullable", True),
            null_count=c.get("null_count", 0),
            null_percent=c.get("null_percent", 0.0),
            distinct_count=c.get("distinct_count", 0),
            top_values=c.get("top_values", []),
            value_counts=c.get("value_counts"),
            min_length=(c.get("string_stats") or {}).get("min_length"),
            max_length=(c.get("string_stats") or {}).get("max_length"),
            avg_length=(c.get("string_stats") or {}).get("avg_length"),
            min_value=(c.get("numeric_stats") or {}).get("min"),
            max_value=(c.get("numeric_stats") or {}).get("max"),
            median=(c.get("numeric_stats") or {}).get("median"),
            stddev=(c.get("numeric_stats") or {}).get("stddev"),
            detected_pattern=c.get("pattern"),
            anomalies=c.get("anomalies", []),
        ))
    v = data.get("validation", {})
    return TechTableProfile(
        table_name=data.get("table", ""),
        row_count=data.get("row_count", 0),
        size_bytes=data.get("size_bytes"),
        profiled_at=data.get("profiled_at", ""),
        columns=cols,
        validation=ValidationResult(
            status=v.get("status", "pass"),
            anomalies=v.get("anomalies", []),
            warnings=v.get("warnings", []),
        ),
    )
