import { useCallback, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FilterBar } from "../components/FilterBar";
import { ProfilingWizard } from "../components/ProfilingWizard";
import Onboarding from "../components/Onboarding";
import { HelpIcon } from "../components/HelpSystem";
import { Badge, Card } from "../components/rds";
import type { DatasetStub } from "../hooks/useProgressiveCatalog";
import type { ApiConfig, CatalogTable } from "../types/catalog";

const PAGE_SIZE = 50;

function matchesFilter(t: CatalogTable, state: "all" | "none" | "tech" | "full") {
  const tech = t.profiling.technical;
  const sem = t.profiling.semantic;
  if (state === "all") return true;
  if (state === "none") return tech !== "available";
  if (state === "tech") return tech === "available" && sem !== "available";
  return tech === "available" && sem === "available";
}

function profilingBadge(t: CatalogTable) {
  const tech = t.profiling.technical;
  const sem = t.profiling.semantic;
  if (tech === "running" || sem === "running") return <Badge tone="running">Profiling…</Badge>;
  if (tech === "available" && sem === "available") return <Badge tone="success">Fully profiled</Badge>;
  if (tech === "available") return <Badge tone="info">Technical only</Badge>;
  return <Badge tone="neutral">Not profiled</Badge>;
}

function formatSize(bytes: number | null) {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export default function CatalogHome(props: {
  config: ApiConfig | null;
  catalog: { datasets: DatasetStub[]; loadingDatasets: boolean; loadDataset: (id: string) => void; loadAll: () => void };
  onRefresh?: () => void;
}) {
  const { datasets: dsStubs, loadingDatasets, loadDataset, loadAll } = props.catalog;
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState<"all" | "none" | "tech" | "full">("all");
  const [visibleCounts, setVisibleCounts] = useState<Record<string, number>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const [onboardingDismissed, setOnboardingDismissed] = useState(
    () => localStorage.getItem("dc_onboarding_dismissed") === "true"
  );

  const [showWizard, setShowWizard] = useState(false);

  const allTables = useMemo(() => dsStubs.flatMap((ds) => ds.tables), [dsStubs]);
  const totalTables = allTables.length;
  const totalDatasets = dsStubs.length;
  const profiledCount = useMemo(
    () => allTables.filter((t) => t.profiling.technical === "available").length,
    [allTables],
  );

  const showOnboarding = !loadingDatasets && totalDatasets > 0 && totalTables > 0 && profiledCount === 0 && !onboardingDismissed;

  const toggleExpand = useCallback((dsId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(dsId)) {
        next.delete(dsId);
      } else {
        next.add(dsId);
        loadDataset(dsId);
      }
      return next;
    });
  }, [loadDataset]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return dsStubs.map((ds) => ({
      ...ds,
      tables: ds.tables.filter((t: CatalogTable) => {
        if (!matchesFilter(t, stateFilter)) return false;
        if (!q) return true;
        return t.table_id.toLowerCase().includes(q) || t.fq_table.toLowerCase().includes(q);
      }),
    }));
  }, [dsStubs, search, stateFilter]);

  const filteredTables = filtered.reduce((n, d) => n + d.tables.length, 0);

  return (
    <div style={{ padding: "32px 40px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: "var(--wb-text)", display: "flex", alignItems: "center", gap: 8 }}>
            {props.config?.data_project_name || "Data Catalog"}
            <HelpIcon title="Data Catalog" content="Browse all BigQuery datasets in your project. Click a dataset to expand it, then profile tables to generate metadata. Use the sidebar to access Terminology, Cohort Builder, and the Data AMA Agent." />
            {props.config?.data_project_name && (
              <span style={{ fontSize: 14, fontWeight: 400, color: "var(--wb-muted)", marginLeft: 8 }}>
                ({props.config?.data_project})
              </span>
            )}
            {!props.config?.data_project_name && (
              <span style={{ color: "var(--wb-primary)", marginLeft: 8 }}>{props.config?.data_project}</span>
            )}
          </h1>
          <p style={{ color: "var(--wb-muted)", margin: "8px 0 0", fontSize: 14 }}>
            {loadingDatasets
              ? "Loading datasets…"
              : `${totalDatasets} dataset(s)${totalTables > 0 ? `, ${totalTables} table(s) loaded` : ""}. Click a dataset to load its tables.`}
          </p>
        </div>
        {!loadingDatasets && totalDatasets > 0 && (
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={() => { loadAll(); setShowWizard(true); }}
              style={{
                background: "var(--wb-primary)",
                color: "#fff",
                border: "1px solid var(--wb-primary)",
                borderRadius: 6,
                padding: "6px 14px",
                fontSize: 12,
                cursor: "pointer",
                fontWeight: 600,
                fontFamily: "var(--wb-font)",
              }}
            >
              Profile Wizard
            </button>
          </div>
        )}
      </div>

      {showOnboarding && (
        <Onboarding
          show
          totalTables={totalTables}
          onDismiss={() => {
            localStorage.setItem("dc_onboarding_dismissed", "true");
            setOnboardingDismissed(true);
          }}
        />
      )}

      <Card style={{ marginBottom: 24, marginTop: 16 }}>
        <FilterBar search={search} onSearch={setSearch} stateFilter={stateFilter} onStateFilter={setStateFilter} />
        {search || stateFilter !== "all" ? (
          <div style={{ fontSize: 13, color: "var(--wb-muted)", marginTop: 8 }}>
            Showing {filteredTables} of {totalTables} tables
          </div>
        ) : null}
      </Card>

      {!loadingDatasets && totalDatasets > 0 && totalTables === 0 && dsStubs.every((d) => d.loaded) && (
        <Card style={{ textAlign: "center", padding: 40, marginBottom: 24 }}>
          <div style={{ fontSize: 18, fontWeight: 600, color: "var(--wb-text)", marginBottom: 8 }}>
            No BigQuery tables found
          </div>
          <div style={{ fontSize: 14, color: "var(--wb-muted)", lineHeight: 1.6, maxWidth: 420, margin: "0 auto" }}>
            Data Catalog v2 currently supports BigQuery tables only. This project doesn't have any BQ datasets, or you may not have access.
          </div>
          <div style={{ marginTop: 16 }}>
            <button
              onClick={() => {/* handled by parent via settings */}}
              style={{
                background: "var(--wb-primary)",
                color: "#fff",
                border: "none",
                borderRadius: 6,
                padding: "8px 20px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Open Settings
            </button>
          </div>
        </Card>
      )}

      {filtered.map((ds) => {
        const isExpanded = expanded.has(ds.dataset_id);
        return (
          <div key={ds.dataset_id} style={{ marginBottom: 24 }}>
            <div
              style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: isExpanded ? 10 : 0, cursor: "pointer" }}
              onClick={() => toggleExpand(ds.dataset_id)}
            >
              <h2
                style={{
                  fontSize: 14,
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                  color: "var(--wb-muted)",
                  margin: 0,
                }}
              >
                <span style={{ marginRight: 8, fontSize: 12 }}>{isExpanded ? "▼" : "▶"}</span>
                {ds.dataset_id}
                <span style={{ fontWeight: 400, textTransform: "none", marginLeft: 8 }}>
                  {ds.loaded ? `${ds.tables.length} ${ds.tables.length === 1 ? "table" : "tables"}` : ds.loading ? "loading..." : "click to load"}
                </span>
              </h2>
            </div>

            {isExpanded && ds.loading && (
              <div style={{ padding: 16, color: "var(--wb-muted)", fontSize: 13 }}>Loading tables...</div>
            )}

            {isExpanded && ds.loaded && ds.tables.length === 0 && (
              <div style={{ padding: 16, color: "var(--wb-muted)", fontSize: 13 }}>No tables in this dataset.</div>
            )}

            {isExpanded && ds.loaded && ds.tables.length > 0 && (<>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                <thead>
                  <tr style={{ background: "var(--wb-surface)" }}>
                    {["Table", "Rows", "Size", "Columns", "Profiling"].map((h) => (
                      <th
                        key={h}
                        style={{
                          textAlign: "left",
                          padding: "10px 14px",
                          borderBottom: "2px solid var(--wb-border)",
                          color: "var(--wb-muted)",
                          fontWeight: 600,
                          fontSize: 12,
                          textTransform: "uppercase",
                          letterSpacing: "0.04em",
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ds.tables.slice(0, visibleCounts[ds.dataset_id] || PAGE_SIZE).map((t) => {
                    const to = `/table/${encodeURIComponent(t.project_id)}/${encodeURIComponent(t.dataset_id)}/${encodeURIComponent(t.table_id)}`;
                    return (
                      <tr
                        key={t.fq_table}
                        style={{
                          borderBottom: "1px solid var(--wb-border)",
                          cursor: "pointer",
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = "#f8fafb"; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                      >
                        <td style={{ padding: "10px 14px" }}>
                          <Link to={to} style={{ fontWeight: 600 }}>
                            {t.business_name || t.table_id}
                          </Link>
                          {t.business_name ? (
                            <div style={{ fontSize: 12, color: "var(--wb-muted)", fontWeight: 400, marginTop: 1 }}>
                              {t.table_id}
                            </div>
                          ) : null}
                          {t.table_definition ? (
                            <div style={{ fontSize: 12, color: "var(--wb-muted)", fontWeight: 400, marginTop: 3, lineHeight: 1.4, maxWidth: 360 }}>
                              {t.table_definition.length > 120 ? t.table_definition.slice(0, 120) + "…" : t.table_definition}
                            </div>
                          ) : null}
                        </td>
                        <td style={{ padding: "10px 14px", color: "var(--wb-muted)", verticalAlign: "top" }}>
                          {t.row_count != null ? t.row_count.toLocaleString() : "—"}
                        </td>
                        <td style={{ padding: "10px 14px", color: "var(--wb-muted)", verticalAlign: "top" }}>
                          {formatSize(t.size_bytes)}
                        </td>
                        <td style={{ padding: "10px 14px", color: "var(--wb-muted)", verticalAlign: "top" }}>
                          {t.column_count}
                        </td>
                        <td style={{ padding: "10px 14px", verticalAlign: "top" }}>{profilingBadge(t)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {ds.tables.length > (visibleCounts[ds.dataset_id] || PAGE_SIZE) && (
              <button
                onClick={() =>
                  setVisibleCounts((prev) => ({
                    ...prev,
                    [ds.dataset_id]: (prev[ds.dataset_id] || PAGE_SIZE) + PAGE_SIZE,
                  }))
                }
                style={{
                  display: "block",
                  margin: "8px auto",
                  background: "none",
                  border: "1px solid var(--wb-border)",
                  borderRadius: 6,
                  padding: "6px 20px",
                  fontSize: 12,
                  cursor: "pointer",
                  color: "var(--wb-primary)",
                  fontWeight: 500,
                }}
              >
                Show more ({ds.tables.length - (visibleCounts[ds.dataset_id] || PAGE_SIZE)} remaining)
              </button>
            )}
            </>)}
          </div>
        );
      })}

      {showWizard && (
        <ProfilingWizard
          datasets={dsStubs.filter((ds) => ds.loaded).map((ds) => ({
            dataset_id: ds.dataset_id,
            tables: ds.tables.map((t) => ({
              table_id: t.table_id,
              fq_table: t.fq_table,
              profiling: { technical: t.profiling.technical, semantic: t.profiling.semantic },
            })),
          }))}
          loadingDatasets={dsStubs.some((ds) => !ds.loaded)}
          project={props.config?.data_project || ""}
          onClose={() => setShowWizard(false)}
          onComplete={() => {
            setShowWizard(false);
            props.onRefresh?.();
          }}
        />
      )}
    </div>
  );
}
