import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { DataPreview } from "../components/DataPreview";
import { ExplorePanel } from "../components/ExplorePanel";
import { ProfilingActions } from "../components/ProfilingActions";
import { SemProfileView } from "../components/SemProfile";
import { TechProfileView } from "../components/TechProfile";
import { Badge, Card, Tabs } from "../components/rds";
import { useChartSuggestions } from "../hooks/useCharts";
import { usePreview } from "../hooks/usePreview";
import { useProfileStatus, useSemProfile, useTechProfile } from "../hooks/useProfiles";

export default function TablePage() {
  const { project = "", dataset = "", table = "" } = useParams();
  const [tab, setTab] = useState(0);

  const { data: preview, loading: prevLoading, err: prevErr } = usePreview(project, dataset, table);
  const { status, reload } = useProfileStatus(project, dataset, table);
  const techAvailable = status?.technical === "available";
  const semAvailable = status?.semantic === "available";

  const { data: tech, err: techErr } = useTechProfile(project, dataset, table, techAvailable);
  const { data: sem, err: semErr } = useSemProfile(project, dataset, table, semAvailable);
  const { charts, loading: chartsLoading, err: chartsErr } = useChartSuggestions(tech, sem, techAvailable);

  const businessName = sem?.business_name;
  const tableDefinition = sem?.table_definition;

  const tabs = useMemo(() => ["Preview", "Technical", "Semantic", "Explore"], []);
  const disabled = useMemo(() => [false, false, !techAvailable, !techAvailable], [techAvailable]);

  return (
    <div style={{ padding: "32px 40px", maxWidth: 1000 }}>
      {/* Header */}
      <h1 style={{ margin: 0, fontSize: 24, fontWeight: 400, color: "var(--wb-text)" }}>
        {businessName ? (
          <>
            <span style={{ fontWeight: 700, color: "var(--wb-primary)" }}>{businessName}</span>
          </>
        ) : (
          <>
            Reviewing table:{" "}
            <span style={{ fontWeight: 700, color: "var(--wb-primary)" }}>{table}</span>
          </>
        )}
      </h1>
      <p style={{ color: "var(--wb-muted)", margin: "4px 0 0", fontSize: 14 }}>
        {project}.{dataset}.{table}
      </p>

      {/* Table-level summary card (when semantic profile exists) */}
      {(tableDefinition || tech) ? (
        <Card style={{ marginTop: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 32px" }}>
            {tableDefinition ? (
              <div style={{ flex: "1 1 320px" }}>
                <div style={{ fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--wb-muted)", marginBottom: 4 }}>
                  Description
                </div>
                <div style={{ fontSize: 14, lineHeight: 1.6, color: "var(--wb-text)" }}>{tableDefinition}</div>
              </div>
            ) : null}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 24px", alignItems: "flex-start" }}>
              {tech ? (
                <>
                  <Stat label="Rows" value={tech.row_count != null ? tech.row_count.toLocaleString() : "—"} />
                  <Stat label="Columns" value={tech.columns.length} />
                  <Stat label="Profiled" value={tech.profiled_at ? new Date(tech.profiled_at).toLocaleDateString() : "—"} />
                  <Stat label="Validation" value={<Badge tone={tech.validation.status === "pass" ? "success" : "warn"}>{tech.validation.status}</Badge>} />
                </>
              ) : null}
              {sem ? (
                <Stat label="Model" value={sem.model_used || "—"} />
              ) : null}
            </div>
          </div>
        </Card>
      ) : null}

      {/* Tabs */}
      <div style={{ marginTop: 20 }}>
        <Tabs labels={tabs} active={tab} onChange={setTab} disabled={disabled} />
      </div>

      {/* Tab content */}
      {tab === 0 ? <DataPreview data={preview} loading={prevLoading} err={prevErr} /> : null}

      {tab === 1 ? (
        <>
          <ProfilingActions
            project={project} dataset={dataset} table={table}
            status={status} onTriggered={reload}
            kind="technical"
          />
          {techErr ? <p style={{ color: "var(--wb-danger)" }}>{techErr}</p> : null}
          <TechProfileView data={tech} />
        </>
      ) : null}

      {tab === 2 ? (
        <>
          <ProfilingActions
            project={project} dataset={dataset} table={table}
            status={status} onTriggered={reload}
            kind="semantic"
          />
          {semErr ? <p style={{ color: "var(--wb-danger)" }}>{semErr}</p> : null}
          <SemProfileView data={sem} />
        </>
      ) : null}

      {tab === 3 ? (
        <ExplorePanel
          project={project}
          dataset={dataset}
          table={table}
          technical={tech}
          semantic={sem}
          suggestions={charts}
          sugLoading={chartsLoading}
          sugErr={chartsErr}
        />
      ) : null}
    </div>
  );
}

function Stat(props: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--wb-muted)" }}>
        {props.label}
      </div>
      <div style={{ fontSize: 14, fontWeight: 500, marginTop: 2 }}>{props.value}</div>
    </div>
  );
}
