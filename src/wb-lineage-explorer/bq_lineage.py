"""
BigQuery Lineage Queries
Queries the FHIR Provenance table for lineage data across
operational_fhir_mirror, landing_fhir_mirror, and cdr_fhir_mirror.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from google.cloud import bigquery

BQ_PROJECT = "prj-d-1v-ucd"
BQ_LOCATION = "us-west1"

DATASETS = {
    "operational_fhir_mirror": f"{BQ_PROJECT}.operational_fhir_mirror",
    "landing_fhir_mirror": f"{BQ_PROJECT}.landing_fhir_mirror",
    "cdr_fhir_mirror": f"{BQ_PROJECT}.cdr_fhir_mirror",
}

# Typed ID fields in target (from INFORMATION_SCHEMA for operational_fhir_mirror.Provenance)
_TARGET_ID_FIELDS = [
    "resourceId", "patientId", "observationId", "conditionId", "procedureId",
    "medicationRequestId", "encounterId", "taskId", "carePlanId",
    "requestGroupId", "basicId", "verificationResultId",
    "questionnaireResponseId", "documentReferenceId",
    "communicationRequestId", "communicationId", "deviceId",
    "personId", "organizationId", "groupId", "questionnaireId",
    "relatedPersonId", "immunizationId", "diagnosticReportId",
    "claimId", "explanationOfBenefitId", "careTeamId",
    "medicationId", "medicationAdministrationId", "consentId",
    "locationId", "medicationStatementId", "medicationDispenseId",
    "researchSubjectId", "coverageId", "insurancePlanId",
    "invoiceId", "supplyDeliveryId", "auditEventId",
    "coverageEligibilityResponseId", "coverageEligibilityRequestId",
    "practitionerId", "practitionerRoleId", "deviceDefinitionId",
    "conceptMapId", "imagingStudyId", "endpointId",
    "structureDefinitionId", "episodeOfCareId", "compositionId",
    "codeSystemId", "contractId", "allergyIntoleranceId",
    "appointmentId", "activityDefinitionId", "detectedIssueId",
    "deviceUseStatementId", "familyMemberHistoryId", "goalId",
    "measureId", "graphDefinitionId", "binaryId", "planDefinitionId",
    "listId", "bundleId", "healthcareServiceId",
    "chargeItemDefinitionId", "researchStudyId", "riskAssessmentId",
]

# Typed ID fields in entity.what (different set from target!)
_ENTITY_ID_FIELDS = [
    "resourceId", "patientId", "observationId", "conditionId", "procedureId",
    "taskId", "carePlanId", "requestGroupId", "basicId",
    "binaryId", "personId", "communicationId",
    "communicationRequestId", "documentReferenceId", "deviceId",
    "questionnaireResponseId", "questionnaireId", "encounterId",
    "organizationId", "groupId", "relatedPersonId",
    "immunizationId", "diagnosticReportId", "careTeamId",
    "medicationRequestId", "medicationAdministrationId",
    "consentId", "locationId", "medicationStatementId",
    "medicationDispenseId", "researchSubjectId", "coverageId",
    "insurancePlanId", "contractId", "compositionId",
    "episodeOfCareId", "practitionerId", "practitionerRoleId",
    "valueSetId", "measureId", "allergyIntoleranceId",
    "appointmentId", "goalId", "familyMemberHistoryId",
    "deviceUseStatementId", "supplyDeliveryId",
    "coverageEligibilityResponseId", "verificationResultId",
    "listId", "researchStudyId",
]


def _coalesce_fields(prefix: str, fields: list[str]) -> str:
    return "COALESCE(" + ", ".join(f"{prefix}.{f}" for f in fields) + ")"


def _get_client() -> bigquery.Client:
    return bigquery.Client(project=BQ_PROJECT)


def _run_query(sql: str) -> pd.DataFrame:
    client = _get_client()
    job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    job = client.query(sql, job_config=job_config, location=BQ_LOCATION)
    return job.result().to_dataframe()


# ── High-Level Lineage ───────────────────────────────────────────────────────


def get_high_level_lineage(dataset_key: str) -> pd.DataFrame:
    """
    Aggregate lineage: entity_type → activity → target_type with counts.
    Provenance-to-Provenance rows are excluded so the graph only shows
    clinical / administrative resources connected by activities.

    Returns a DataFrame with columns:
      entity_type, activity, target_type, cnt
    """
    ds = DATASETS[dataset_key]
    sql = f"""
    SELECT
      IFNULL(e.what.type, '(none)') AS entity_type,
      IFNULL(activity.text, '(unknown)') AS activity,
      IFNULL(t.type, '(none)') AS target_type,
      COUNT(*) AS cnt
    FROM `{ds}.Provenance` p,
      UNNEST(p.target) t
      LEFT JOIN UNNEST(p.entity) e
    WHERE IFNULL(t.type, '') != 'Provenance'
      AND IFNULL(e.what.type, '') != 'Provenance'
    GROUP BY 1, 2, 3
    ORDER BY cnt DESC
    LIMIT 300
    """
    return _run_query(sql)


def get_activity_summary(dataset_key: str) -> pd.DataFrame:
    """Top activities with counts."""
    ds = DATASETS[dataset_key]
    sql = f"""
    SELECT
      IFNULL(activity.text, '(unknown)') AS activity,
      COUNT(*) AS cnt
    FROM `{ds}.Provenance`
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 50
    """
    return _run_query(sql)


def get_target_type_summary(dataset_key: str) -> pd.DataFrame:
    """Top target resource types with counts (excludes Provenance itself)."""
    ds = DATASETS[dataset_key]
    sql = f"""
    SELECT
      IFNULL(t.type, '(unknown)') AS target_type,
      COUNT(*) AS cnt
    FROM `{ds}.Provenance` p,
      UNNEST(p.target) t
    WHERE IFNULL(t.type, '') != 'Provenance'
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 50
    """
    return _run_query(sql)


# ── Instance-Level Lineage ────────────────────────────────────────────────────


def resolve_resource_type(dataset_key: str, resource_id: str) -> str | None:
    """
    Given just a FHIR resource ID, find its resource type by looking it up
    in the Provenance target references.  Returns the type string
    (e.g. 'Patient') or None if not found.
    """
    ds = DATASETS[dataset_key]
    target_id_expr = _coalesce_fields("t", _TARGET_ID_FIELDS)
    sql = f"""
    SELECT DISTINCT t.type AS resource_type
    FROM `{ds}.Provenance` p,
      UNNEST(p.target) t
    WHERE {target_id_expr} = '{resource_id}'
      AND t.type IS NOT NULL
      AND t.type != 'Provenance'
    LIMIT 5
    """
    df = _run_query(sql)
    if df.empty:
        return None
    return df.iloc[0]["resource_type"]


def search_target_instances(
    dataset_key: str,
    resource_type: str,
    limit: int = 50,
) -> pd.DataFrame:
    """
    Find recent target resource instances of a given type.
    Returns provenance_id, target_id, activity, recorded.
    """
    ds = DATASETS[dataset_key]
    target_id_expr = _coalesce_fields("t", _TARGET_ID_FIELDS)
    sql = f"""
    SELECT DISTINCT
      {target_id_expr} AS target_id,
      IFNULL(activity.text, '(unknown)') AS activity,
      MIN(recorded) AS first_seen,
      MAX(recorded) AS last_seen,
      COUNT(*) AS provenance_count
    FROM `{ds}.Provenance` p,
      UNNEST(p.target) t
    WHERE t.type = '{resource_type}'
      AND {target_id_expr} IS NOT NULL
    GROUP BY 1, 2
    ORDER BY last_seen DESC
    LIMIT {limit}
    """
    return _run_query(sql)


def get_instance_lineage(
    dataset_key: str,
    resource_type: str,
    resource_id: str,
    max_depth: int = 5,
) -> pd.DataFrame:
    """
    Trace backward lineage from a final target node.
    Returns a flat table of all provenance records in the chain.

    Columns: provenance_id, activity, target_type, target_id,
             entity_role, entity_type, entity_id, agent_display, recorded
    """
    ds = DATASETS[dataset_key]
    target_id_expr = _coalesce_fields("t", _TARGET_ID_FIELDS)
    entity_id_expr = _coalesce_fields("e.what", _ENTITY_ID_FIELDS)

    # Recursive CTE to walk the provenance chain backward
    sql = f"""
    WITH RECURSIVE lineage AS (
      -- Seed: provenance records that target the selected resource
      SELECT
        p.id AS provenance_id,
        IFNULL(p.activity.text, '(unknown)') AS activity,
        t.type AS target_type,
        {target_id_expr} AS target_id,
        e.role AS entity_role,
        IFNULL(e.what.type, '(none)') AS entity_type,
        {entity_id_expr} AS entity_id,
        p.recorded,
        1 AS depth
      FROM `{ds}.Provenance` p,
        UNNEST(p.target) t
        LEFT JOIN UNNEST(p.entity) e
      WHERE t.type = '{resource_type}'
        AND {target_id_expr} = '{resource_id}'

      UNION ALL

      -- Walk backward: find provenance where entity from previous step is the target
      SELECT
        p2.id AS provenance_id,
        IFNULL(p2.activity.text, '(unknown)') AS activity,
        t2.type AS target_type,
        {_coalesce_fields("t2", _TARGET_ID_FIELDS)} AS target_id,
        e2.role AS entity_role,
        IFNULL(e2.what.type, '(none)') AS entity_type,
        {_coalesce_fields("e2.what", _ENTITY_ID_FIELDS)} AS entity_id,
        p2.recorded,
        l.depth + 1 AS depth
      FROM lineage l
      JOIN `{ds}.Provenance` p2 ON TRUE,
        UNNEST(p2.target) t2
        LEFT JOIN UNNEST(p2.entity) e2
      WHERE l.entity_id IS NOT NULL
        AND l.entity_type != '(none)'
        AND t2.type = l.entity_type
        AND {_coalesce_fields("t2", _TARGET_ID_FIELDS)} = l.entity_id
        AND l.depth < {max_depth}
        -- Avoid revisiting the same provenance
        AND p2.id != l.provenance_id
    )
    SELECT DISTINCT * FROM lineage
    ORDER BY depth, recorded
    LIMIT 500
    """
    return _run_query(sql)


def get_instance_lineage_flat(
    dataset_key: str,
    resource_type: str,
    resource_id: str,
) -> pd.DataFrame:
    """
    Single-hop lineage: get all provenance records directly targeting this resource,
    including their source entities. Cheaper than recursive.
    """
    ds = DATASETS[dataset_key]
    target_id_expr = _coalesce_fields("t", _TARGET_ID_FIELDS)
    entity_id_expr = _coalesce_fields("e.what", _ENTITY_ID_FIELDS)

    sql = f"""
    SELECT
      p.id AS provenance_id,
      IFNULL(p.activity.text, '(unknown)') AS activity,
      t.type AS target_type,
      {target_id_expr} AS target_id,
      e.role AS entity_role,
      IFNULL(e.what.type, '(none)') AS entity_type,
      {entity_id_expr} AS entity_id,
      p.recorded,
      -- agent info
      a.who.type AS agent_type,
      COALESCE(a.who.deviceId, a.who.organizationId, a.who.practitionerId,
               a.who.patientId) AS agent_id
    FROM `{ds}.Provenance` p,
      UNNEST(p.target) t
      LEFT JOIN UNNEST(p.entity) e
      LEFT JOIN UNNEST(p.agent) a
    WHERE t.type = '{resource_type}'
      AND {target_id_expr} = '{resource_id}'
      AND IFNULL(t.type, '') != 'Provenance'
      AND IFNULL(e.what.type, '') != 'Provenance'
    ORDER BY p.recorded
    LIMIT 200
    """
    return _run_query(sql)


def get_multi_hop_lineage(
    dataset_key: str,
    resource_type: str,
    resource_id: str,
    max_depth: int = 3,
) -> pd.DataFrame:
    """
    Multi-hop iterative lineage: walks backward through provenance chains.
    Uses iterative Python queries instead of recursive SQL (more reliable for BQ).
    """
    all_rows = []
    visited_ids = set()
    frontier = [(resource_type, resource_id)]

    ds = DATASETS[dataset_key]
    target_id_expr = _coalesce_fields("t", _TARGET_ID_FIELDS)
    entity_id_expr = _coalesce_fields("e.what", _ENTITY_ID_FIELDS)

    for depth in range(1, max_depth + 1):
        if not frontier:
            break

        # Build WHERE clause for all frontier items
        conditions = []
        for rtype, rid in frontier:
            conditions.append(
                f"(t.type = '{rtype}' AND {target_id_expr} = '{rid}')"
            )
        where_clause = " OR ".join(conditions)

        sql = f"""
        SELECT DISTINCT
          p.id AS provenance_id,
          IFNULL(p.activity.text, '(unknown)') AS activity,
          t.type AS target_type,
          {target_id_expr} AS target_id,
          e.role AS entity_role,
          IFNULL(e.what.type, '(none)') AS entity_type,
          {entity_id_expr} AS entity_id,
          p.recorded,
          {depth} AS depth
        FROM `{ds}.Provenance` p,
          UNNEST(p.target) t
          LEFT JOIN UNNEST(p.entity) e
        WHERE ({where_clause})
          AND IFNULL(t.type, '') != 'Provenance'
          AND IFNULL(e.what.type, '') != 'Provenance'
        ORDER BY p.recorded
        LIMIT 200
        """
        df = _run_query(sql)
        if df.empty:
            break

        all_rows.append(df)

        # Build next frontier from entities we haven't visited
        next_frontier = []
        for _, row in df.iterrows():
            etype = row.get("entity_type", "(none)")
            eid = row.get("entity_id")
            if eid and etype != "(none)":
                key = (etype, eid)
                if key not in visited_ids:
                    visited_ids.add(key)
                    next_frontier.append(key)

        frontier = next_frontier

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


# ── Profile Filtering ─────────────────────────────────────────────────────────

# Resource types excluded from profile queries (not useful or too large/meta)
_PROFILE_EXCLUDED_TYPES = {
    "(unknown)", "AuditEvent", "Bundle", "CodeSystem",
    "GraphDefinition", "StructureDefinition",
}


def _resource_type_to_id_field(resource_type: str) -> str:
    """Convert a FHIR resource type to its typed ID field name in Provenance.target.
    E.g. 'Patient' → 'patientId', 'MedicationRequest' → 'medicationRequestId'."""
    return resource_type[0].lower() + resource_type[1:] + "Id"


def get_filterable_target_types(dataset_key: str) -> list[str]:
    """Return target resource types eligible for profile filtering."""
    df = get_target_type_summary(dataset_key)
    return [
        t for t in df["target_type"].tolist()
        if t not in _PROFILE_EXCLUDED_TYPES
    ]


def get_resource_profiles(dataset_key: str, resource_type: str) -> pd.DataFrame:
    """Query distinct meta.profile values from a resource type table.
    Returns DataFrame with columns: profile, cnt."""
    ds = DATASETS[dataset_key]
    sql = f"""
    SELECT
      prof AS profile,
      COUNT(*) AS cnt
    FROM `{ds}.{resource_type}` r,
      UNNEST(r.meta.profile) prof
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 100
    """
    return _run_query(sql)


def get_high_level_lineage_by_profile(
    dataset_key: str,
    resource_type: str,
    profile: str,
) -> pd.DataFrame:
    """
    High-level lineage filtered to Provenance records whose target
    is *resource_type* AND that resource carries the given *profile*.

    JOINs Provenance.target → {ResourceType} table on the typed ID,
    then filters by meta.profile on the resource.

    Returns: entity_type, activity, target_type, cnt
    """
    ds = DATASETS[dataset_key]
    id_field = _resource_type_to_id_field(resource_type)

    sql = f"""
    SELECT
      IFNULL(e.what.type, '(none)') AS entity_type,
      IFNULL(p.activity.text, '(unknown)') AS activity,
      t.type AS target_type,
      COUNT(*) AS cnt
    FROM `{ds}.Provenance` p,
      UNNEST(p.target) t
      LEFT JOIN UNNEST(p.entity) e
      JOIN `{ds}.{resource_type}` r
        ON t.{id_field} = r.id
    WHERE t.type = '{resource_type}'
      AND '{profile}' IN UNNEST(r.meta.profile)
      AND IFNULL(e.what.type, '') != 'Provenance'
    GROUP BY 1, 2, 3
    ORDER BY cnt DESC
    LIMIT 300
    """
    return _run_query(sql)


# ── Convenience ───────────────────────────────────────────────────────────────


def get_dataset_stats(dataset_key: str) -> dict:
    """Quick stats for a dataset's Provenance table."""
    ds = DATASETS[dataset_key]
    sql = f"""
    SELECT
      COUNT(*) AS total_provenance,
      COUNT(DISTINCT activity.text) AS distinct_activities,
      MIN(recorded) AS earliest,
      MAX(recorded) AS latest
    FROM `{ds}.Provenance`
    """
    df = _run_query(sql)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "total_provenance": f"{int(row['total_provenance']):,}",
        "distinct_activities": int(row["distinct_activities"]),
        "earliest": str(row["earliest"]),
        "latest": str(row["latest"]),
    }
