import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FilterBar } from "../components/FilterBar";
import { Badge, Card } from "../components/rds";
import type { ApiConfig, CatalogDataset, CatalogTable } from "../types/catalog";

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
  datasets: CatalogDataset[];
  loading: boolean;
}) {
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState<"all" | "none" | "tech" | "full">("all");

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return props.datasets.map((ds) => ({
      ...ds,
      tables: ds.tables.filter((t) => {
        if (!matchesFilter(t, stateFilter)) return false;
        if (!q) return true;
        return t.table_id.toLowerCase().includes(q) || t.fq_table.toLowerCase().includes(q);
      }),
    }));
  }, [props.datasets, search, stateFilter]);

  const totalTables = props.datasets.reduce((n, d) => n + d.tables.length, 0);
  const filteredTables = filtered.reduce((n, d) => n + d.tables.length, 0);

  return (
    <div style={{ padding: "32px 40px", maxWidth: 960 }}>
      <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: "var(--wb-text)" }}>
        Browsing project:{" "}
        <span style={{ color: "var(--wb-primary)" }}>{props.config?.data_project}</span>
      </h1>
      <p style={{ color: "var(--wb-muted)", margin: "8px 0 24px", fontSize: 14 }}>
        {props.loading
          ? "Loading datasets…"
          : `${totalTables} table(s) across ${props.datasets.length} dataset(s). Select a table from the sidebar to see details.`}
      </p>

      <Card style={{ marginBottom: 24 }}>
        <FilterBar search={search} onSearch={setSearch} stateFilter={stateFilter} onStateFilter={setStateFilter} />
        {search || stateFilter !== "all" ? (
          <div style={{ fontSize: 13, color: "var(--wb-muted)", marginTop: 8 }}>
            Showing {filteredTables} of {totalTables} tables
          </div>
        ) : null}
      </Card>

      {filtered.map((ds) =>
        ds.tables.length === 0 ? null : (
          <div key={ds.dataset_id} style={{ marginBottom: 24 }}>
            <h2
              style={{
                fontSize: 14,
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.04em",
                color: "var(--wb-muted)",
                margin: "0 0 10px",
              }}
            >
              {ds.dataset_id}
              <span style={{ fontWeight: 400, textTransform: "none", marginLeft: 8 }}>
                {ds.tables.length} {ds.tables.length === 1 ? "table" : "tables"}
              </span>
            </h2>

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
                  {ds.tables.map((t) => {
                    const to = `/table/${encodeURIComponent(t.project_id)}/${encodeURIComponent(t.dataset_id)}/${encodeURIComponent(t.table_id)}`;
                    return (
                      <tr
                        key={t.fq_table}
                        style={{ borderBottom: "1px solid var(--wb-border)", cursor: "pointer" }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "#f8fafb")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
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
          </div>
        ),
      )}
    </div>
  );
}
