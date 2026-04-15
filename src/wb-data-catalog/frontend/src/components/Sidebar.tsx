import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import type { CatalogDataset, CatalogTable } from "../types/catalog";
import { SectionLabel } from "./rds";

function profilingDot(t: CatalogTable) {
  const tech = t.profiling.technical;
  const sem = t.profiling.semantic;
  if (tech === "running" || sem === "running") return "#f0c040";
  if (tech === "available" && sem === "available") return "var(--wb-success)";
  if (tech === "available") return "var(--wb-accent)";
  return "transparent";
}

function NavItem(props: { table: CatalogTable; active: boolean }) {
  const t = props.table;
  const to = `/table/${encodeURIComponent(t.project_id)}/${encodeURIComponent(t.dataset_id)}/${encodeURIComponent(t.table_id)}`;
  return (
    <Link
      to={to}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "7px 20px 7px 32px",
        fontSize: 14,
        color: "var(--wb-sidebar-text)",
        textDecoration: "none",
        background: props.active ? "var(--wb-sidebar-active)" : "transparent",
        borderRadius: 0,
        transition: "background 0.12s",
      }}
      onMouseEnter={(e) => {
        if (!props.active) e.currentTarget.style.background = "var(--wb-sidebar-hover)";
      }}
      onMouseLeave={(e) => {
        if (!props.active) e.currentTarget.style.background = "transparent";
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: profilingDot(t),
          border: profilingDot(t) === "transparent" ? "1.5px solid var(--wb-sidebar-muted)" : "none",
          flexShrink: 0,
        }}
      />
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
        {t.business_name || t.table_id}
      </span>
    </Link>
  );
}

function DatasetSection(props: { dataset: CatalogDataset; activeTable: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          width: "100%",
          padding: "8px 20px",
          border: "none",
          background: "transparent",
          color: "var(--wb-sidebar-text)",
          cursor: "pointer",
          fontSize: 14,
          fontWeight: 600,
          fontFamily: "var(--wb-font)",
          textAlign: "left",
        }}
      >
        <span style={{ fontSize: 10, width: 14 }}>{open ? "▾" : "▸"}</span>
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {props.dataset.dataset_id}
        </span>
        <span style={{ fontSize: 12, color: "var(--wb-sidebar-muted)", fontWeight: 400 }}>
          {props.dataset.tables.length} {props.dataset.tables.length === 1 ? "table" : "tables"}
        </span>
      </button>
      {open
        ? props.dataset.tables.map((t) => (
            <NavItem key={t.fq_table} table={t} active={props.activeTable === t.fq_table} />
          ))
        : null}
    </div>
  );
}

export function Sidebar(props: {
  projectId: string;
  datasets: CatalogDataset[];
  loading: boolean;
  onSettingsClick: () => void;
  onRefresh: () => void;
}) {
  const location = useLocation();
  const parts = location.pathname.split("/");
  const activeTable = parts.length >= 5 ? `${decodeURIComponent(parts[2])}.${decodeURIComponent(parts[3])}.${decodeURIComponent(parts[4])}` : "";

  return (
    <nav
      style={{
        width: "var(--wb-sidebar-width)",
        minWidth: "var(--wb-sidebar-width)",
        height: "100vh",
        background: "var(--wb-sidebar-bg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        position: "sticky",
        top: 0,
      }}
    >
      {/* Logo / brand */}
      <div style={{ padding: "20px 20px 12px" }}>
        <Link to="/" style={{ textDecoration: "none", color: "var(--wb-sidebar-text)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16, lineHeight: 1.2 }}>workbench</div>
              <div style={{ fontSize: 12, color: "var(--wb-sidebar-muted)", lineHeight: 1.2 }}>Data Catalog</div>
            </div>
          </div>
        </Link>
      </div>

      {/* Project badge */}
      {props.projectId ? (
        <div style={{ padding: "0 20px 8px" }}>
          <div
            style={{
              background: "var(--wb-sidebar-active)",
              borderRadius: "var(--wb-radius)",
              padding: "8px 12px",
              fontSize: 13,
              color: "var(--wb-sidebar-text)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {props.projectId}
            </span>
            <button
              type="button"
              onClick={props.onSettingsClick}
              style={{
                background: "none",
                border: "none",
                color: "var(--wb-sidebar-muted)",
                cursor: "pointer",
                fontSize: 14,
                padding: 2,
                lineHeight: 1,
              }}
              title="Settings"
            >
              ⚙
            </button>
          </div>
        </div>
      ) : null}

      {/* Navigation tree */}
      <div style={{ flex: 1, overflowY: "auto", paddingBottom: 16 }}>
        {props.loading ? (
          <div style={{ padding: "20px", color: "var(--wb-sidebar-muted)", fontSize: 13 }}>Loading datasets…</div>
        ) : props.datasets.length === 0 ? (
          <div style={{ padding: "20px", color: "var(--wb-sidebar-muted)", fontSize: 13 }}>
            No datasets found.
          </div>
        ) : (
          props.datasets.map((ds) => (
            <div key={ds.dataset_id}>
              <SectionLabel>{ds.dataset_id}</SectionLabel>
              <DatasetSection dataset={ds} activeTable={activeTable} />
            </div>
          ))
        )}
      </div>

      {/* Bottom toolbar */}
      <div
        style={{
          padding: "10px 20px",
          borderTop: "1px solid rgba(255,255,255,0.1)",
          display: "flex",
          gap: 8,
        }}
      >
        <button
          type="button"
          onClick={props.onRefresh}
          style={{
            flex: 1,
            padding: "6px 0",
            background: "var(--wb-sidebar-hover)",
            border: "none",
            borderRadius: "var(--wb-radius)",
            color: "var(--wb-sidebar-text)",
            cursor: "pointer",
            fontSize: 13,
            fontFamily: "var(--wb-font)",
          }}
        >
          ↻ Refresh
        </button>
        <button
          type="button"
          onClick={props.onSettingsClick}
          style={{
            flex: 1,
            padding: "6px 0",
            background: "var(--wb-sidebar-hover)",
            border: "none",
            borderRadius: "var(--wb-radius)",
            color: "var(--wb-sidebar-text)",
            cursor: "pointer",
            fontSize: 13,
            fontFamily: "var(--wb-font)",
          }}
        >
          ⚙ Settings
        </button>
      </div>
    </nav>
  );
}
