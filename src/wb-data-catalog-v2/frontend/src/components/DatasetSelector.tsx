import { useState, type CSSProperties } from "react";
import { Badge } from "./rds";

interface TableInfo {
  table_id: string;
  fq_table: string;
  profiling: { technical: string; semantic: string };
}

interface DatasetGroup {
  dataset_id: string;
  tables: TableInfo[];
}

export interface DatasetSelectorProps {
  datasets: DatasetGroup[];
  selected: string[];
  onSelectionChange: (tables: string[]) => void;
  force: boolean;
  onForceChange: (force: boolean) => void;
}

const headerStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "12px 16px",
  cursor: "pointer",
  borderBottom: "1px solid var(--wb-border)",
  userSelect: "none",
};

const tableRowStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "8px 16px 8px 40px",
  borderBottom: "1px solid var(--wb-border)",
  fontSize: 13,
};

const toggleRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "12px 0 16px",
  fontSize: 13,
  fontWeight: 500,
  color: "var(--wb-text)",
  cursor: "pointer",
  userSelect: "none",
};

export function DatasetSelector(props: DatasetSelectorProps) {
  const { datasets, selected, onSelectionChange, force, onForceChange } = props;
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(props.datasets.map((d) => d.dataset_id)));

  const toggleExpand = (dsId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(dsId)) next.delete(dsId);
      else next.add(dsId);
      return next;
    });
  };

  const toggleTable = (fq: string) => {
    if (selected.includes(fq)) {
      onSelectionChange(selected.filter((t) => t !== fq));
    } else {
      onSelectionChange([...selected, fq]);
    }
  };

  const toggleDataset = (ds: DatasetGroup) => {
    const fqs = ds.tables.map((t) => t.fq_table);
    const allSelected = fqs.every((fq) => selected.includes(fq));
    if (allSelected) {
      onSelectionChange(selected.filter((t) => !fqs.includes(t)));
    } else {
      const added = fqs.filter((fq) => !selected.includes(fq));
      onSelectionChange([...selected, ...added]);
    }
  };

  const selectAll = () => {
    const all = datasets.flatMap((ds) => ds.tables.map((t) => t.fq_table));
    if (selected.length === all.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange(all);
    }
  };

  const totalTables = datasets.reduce((n, ds) => n + ds.tables.length, 0);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600, color: "var(--wb-text)" }}>Select Tables to Profile</div>
          <div style={{ fontSize: 13, color: "var(--wb-muted)", marginTop: 4 }}>
            {selected.length} of {totalTables} tables selected
          </div>
        </div>
        <button
          onClick={selectAll}
          style={{
            background: "none",
            border: "1px solid var(--wb-border)",
            borderRadius: 6,
            padding: "6px 14px",
            fontSize: 12,
            cursor: "pointer",
            fontWeight: 500,
            fontFamily: "var(--wb-font)",
            color: "var(--wb-primary)",
          }}
        >
          {selected.length === totalTables && totalTables > 0 ? "Deselect all" : "Select all"}
        </button>
      </div>

      <label style={toggleRow}>
        <input
          type="checkbox"
          checked={force}
          onChange={(e) => onForceChange(e.target.checked)}
          style={{ width: 16, height: 16, accentColor: "var(--wb-primary)", cursor: "pointer" }}
        />
        Force re-profile (overwrite existing profiles)
      </label>

      <div style={{ border: "1px solid var(--wb-border)", borderRadius: "var(--wb-radius)", overflow: "hidden" }}>
        {datasets.map((ds) => {
          const isExpanded = expanded.has(ds.dataset_id);
          const fqs = ds.tables.map((t) => t.fq_table);
          const allChecked = fqs.length > 0 && fqs.every((fq) => selected.includes(fq));
          const someChecked = fqs.some((fq) => selected.includes(fq));

          return (
            <div key={ds.dataset_id}>
              <div style={headerStyle} onClick={() => toggleExpand(ds.dataset_id)}>
                <input
                  type="checkbox"
                  checked={allChecked}
                  ref={(el) => { if (el) el.indeterminate = someChecked && !allChecked; }}
                  onChange={() => toggleDataset(ds)}
                  onClick={(e) => e.stopPropagation()}
                  style={{ width: 16, height: 16, accentColor: "var(--wb-primary)", cursor: "pointer" }}
                />
                <span style={{ fontSize: 12, color: "var(--wb-muted)" }}>{isExpanded ? "▼" : "▶"}</span>
                <span style={{ fontWeight: 600, fontSize: 14, color: "var(--wb-text)" }}>{ds.dataset_id}</span>
                <Badge tone="neutral">{ds.tables.length} tables</Badge>
              </div>

              {isExpanded && ds.tables.map((t) => (
                <div key={t.fq_table} style={tableRowStyle}>
                  <input
                    type="checkbox"
                    checked={selected.includes(t.fq_table)}
                    onChange={() => toggleTable(t.fq_table)}
                    style={{ width: 14, height: 14, accentColor: "var(--wb-primary)", cursor: "pointer" }}
                  />
                  <span style={{ flex: 1, color: "var(--wb-text)" }}>{t.table_id}</span>
                  <Badge tone={t.profiling.technical === "available" ? "success" : "neutral"}>
                    Tech: {t.profiling.technical === "available" ? "done" : "none"}
                  </Badge>
                  <Badge tone={t.profiling.semantic === "available" ? "success" : "neutral"}>
                    Sem: {t.profiling.semantic === "available" ? "done" : "none"}
                  </Badge>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
