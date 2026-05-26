import { useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { DataPreview } from "../components/DataPreview";
import { KeyInsightsPanel, InteractiveExplorerPanel } from "../components/ExplorePanel";
import { ProfilingActions } from "../components/ProfilingActions";
import { SemProfileView } from "../components/SemProfile";
import { TechProfileView } from "../components/TechProfile";
import { HelpIcon } from "../components/HelpSystem";
import { Badge, Tabs } from "../components/rds";
import { useChartSuggestions } from "../hooks/useCharts";
import { usePreview } from "../hooks/usePreview";
import { useProfileStatus, useSemProfile, useTechProfile } from "../hooks/useProfiles";

export default function TablePage() {
  const { project = "", dataset = "", table = "" } = useParams();
  const [tab, setTab] = useState(0);
  const visitedTabs = useRef(new Set<number>([0]));

  const handleTabChange = (t: number) => {
    visitedTabs.current.add(t);
    setTab(t);
  };

  const visited = (t: number) => visitedTabs.current.has(t);

  const { status, reload } = useProfileStatus(project, dataset, table);
  const techAvailable = status?.technical === "available";
  const semAvailable = status?.semantic === "available";

  const { data: preview, loading: prevLoading, err: prevErr } = usePreview(project, dataset, table);
  const { data: tech, err: techErr } = useTechProfile(project, dataset, table, techAvailable);
  const { data: sem, err: semErr, save: saveSem } = useSemProfile(project, dataset, table, semAvailable);
  const { charts, loading: chartsLoading, err: chartsErr } = useChartSuggestions(
    tech, sem, techAvailable && visited(3),
  );

  const businessName = sem?.business_name;
  const tableDefinition = sem?.table_definition;
  const pk = sem?.primary_key;
  const granularity = sem?.granularity;
  const domain = sem?.semantic_domain;

  const tabs = useMemo(() => ["Preview", "Technical", "Semantic", "Key Insights", "Interactive Explorer"], []);
  const disabled = useMemo(() => [false, false, !techAvailable, !techAvailable, !techAvailable], [techAvailable]);

  return (
    <div style={{ padding: "32px 40px" }}>
      {/* Header */}
      <h1 style={{ margin: 0, fontSize: 24, fontWeight: 400, color: "var(--wb-text)", display: "flex", alignItems: "center", gap: 8 }}>
        {businessName ? (
          <span style={{ fontWeight: 700, color: "var(--wb-primary)" }}>{businessName}</span>
        ) : (
          <>
            Reviewing table:{" "}
            <span style={{ fontWeight: 700, color: "var(--wb-primary)" }}>{table}</span>
          </>
        )}
        <HelpIcon title="Table Detail" content="Use the tabs to explore this table's preview, technical stats, semantic metadata, and visualizations. Click 'Profile' buttons to generate or refresh profiles." />
      </h1>
      <p style={{ color: "var(--wb-muted)", margin: "4px 0 0", fontSize: 14 }}>
        {project}.{dataset}.{table}
      </p>

      {/* Domain badge line */}
      {domain?.primary ? (
        <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <Badge tone="info">{domain.primary}</Badge>
          {domain.sub_domain ? (
            <span style={{ fontSize: 13, color: "var(--wb-muted)" }}>{domain.sub_domain}</span>
          ) : null}
        </div>
      ) : null}

      {/* Table-level summary — compact inline stats + collapsible detail */}
      {(tableDefinition || tech || granularity || pk) ? (
        <TableSummary
          tableDefinition={tableDefinition}
          granularity={granularity}
          pk={pk}
          tech={tech}
          sem={sem}
        />
      ) : null}

      {/* Tabs */}
      <div style={{ marginTop: 20 }}>
        <Tabs labels={tabs} active={tab} onChange={handleTabChange} disabled={disabled} />
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
          <SemProfileView data={sem} onSave={saveSem} />
        </>
      ) : null}

      {tab === 3 ? (
        <KeyInsightsPanel
          technical={tech}
          suggestions={charts}
          sugLoading={chartsLoading}
          sugErr={chartsErr}
        />
      ) : null}

      {tab === 4 ? (
        <InteractiveExplorerPanel
          project={project}
          dataset={dataset}
          table={table}
          technical={tech}
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

function TableSummary(props: {
  tableDefinition?: string | null;
  granularity?: string | null;
  pk?: { columns: string[]; pk_type: string; confidence: string } | null;
  tech?: { row_count: number | null; columns: { length: number }; profiled_at: string; validation: { status: string } } | null;
  sem?: { model_used: string } | null;
}) {
  const { tableDefinition, granularity, pk, tech, sem } = props;
  const [expanded, setExpanded] = useState(true);

  return (
    <div style={{ marginTop: 12 }}>
      {/* Always-visible stats row */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 24px", alignItems: "center", marginBottom: 4 }}>
        {tech ? (
          <>
            <Stat label="Rows" value={tech.row_count != null ? tech.row_count.toLocaleString() : "—"} />
            <Stat label="Columns" value={tech.columns.length} />
            <Stat label="Profiled" value={tech.profiled_at ? new Date(tech.profiled_at).toLocaleDateString() : "—"} />
            <Stat label="Validation" value={<Badge tone={tech.validation.status === "pass" ? "success" : "warn"}>{tech.validation.status}</Badge>} />
          </>
        ) : null}
        {sem ? <Stat label="Model" value={sem.model_used || "—"} /> : null}
        {pk && pk.columns?.length > 0 && pk.pk_type !== "none" ? (
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--wb-muted)" }}>PK</div>
            <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 2 }}>
              {pk.columns.map((col) => <Badge key={col} tone="neutral">{col}</Badge>)}
            </div>
          </div>
        ) : null}
        <button
          onClick={() => setExpanded(!expanded)}
          style={{ marginLeft: "auto", background: "none", border: "none", color: "var(--wb-primary)", cursor: "pointer", fontSize: 12, fontWeight: 500, padding: 0 }}
        >
          {expanded ? "Hide details" : "Show details"}
        </button>
      </div>

      {/* Collapsible description + granularity */}
      {expanded && (tableDefinition || granularity) ? (
        <div style={{ padding: "10px 0", fontSize: 13, color: "var(--wb-text)", lineHeight: 1.6, borderTop: "1px solid var(--wb-border)", marginTop: 4 }}>
          {tableDefinition}
          {tableDefinition && granularity ? " " : null}
          {granularity ? <span style={{ color: "var(--wb-muted)" }}>Granularity: {granularity}</span> : null}
        </div>
      ) : null}
    </div>
  );
}
