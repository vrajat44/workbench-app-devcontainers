import { Component, useMemo, useState, useEffect, useCallback, type ReactNode } from "react";
import type { IMutField, IDataQueryPayload, IRow } from "@kanaries/graphic-walker";
import { GraphicWalker } from "@kanaries/graphic-walker";
import "@kanaries/graphic-walker/dist/style.css";
import type { ChartSuggestion } from "../types/charts";
import type { TechColumn, TechProfile, SemProfile } from "../types/profile";
import { Badge, Button, Card } from "./rds";

const BQ_TO_GW: Record<string, IMutField["semanticType"]> = {
  INTEGER: "quantitative",
  INT64: "quantitative",
  FLOAT: "quantitative",
  FLOAT64: "quantitative",
  NUMERIC: "quantitative",
  BIGNUMERIC: "quantitative",
  BOOLEAN: "nominal",
  BOOL: "nominal",
  STRING: "nominal",
  BYTES: "nominal",
  DATE: "nominal",
  DATETIME: "nominal",
  TIMESTAMP: "nominal",
  TIME: "nominal",
  GEOGRAPHY: "nominal",
  JSON: "nominal",
  RECORD: "nominal",
  STRUCT: "nominal",
};

function bqColToField(col: TechColumn): IMutField {
  const upper = (col.data_type || "STRING").toUpperCase();
  const semanticType = BQ_TO_GW[upper] || "nominal";
  const isQuantitative = semanticType === "quantitative";

  return {
    fid: col.name,
    name: col.name,
    semanticType,
    analyticType: isQuantitative ? "measure" : "dimension",
  };
}

function buildFields(tech: TechProfile): IMutField[] {
  return tech.columns.map((col) => bqColToField(col));
}

function makeComputation(
  project: string,
  dataset: string,
  table: string,
): (payload: IDataQueryPayload) => Promise<IRow[]> {
  return async (payload: IDataQueryPayload): Promise<IRow[]> => {
    const resp = await fetch(
      `/api/gw/compute/${encodeURIComponent(project)}/${encodeURIComponent(dataset)}/${encodeURIComponent(table)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Computation error: ${text}`);
    }
    return resp.json();
  };
}

// ── Error boundary ───────────────────────────────────────────────────────────

interface EBProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (err: Error) => void;
}
interface EBState {
  error: Error | null;
}

class GWErrorBoundary extends Component<EBProps, EBState> {
  state: EBState = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    this.props.onError?.(error);
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div style={{ padding: 24, textAlign: "center" }}>
            <p style={{ color: "var(--wb-danger)", fontWeight: 600, marginBottom: 8 }}>
              Graphic Walker encountered an error
            </p>
            <p style={{ color: "var(--wb-muted)", fontSize: 13, marginBottom: 12 }}>
              {this.state.error.message}
            </p>
            <Button
              variant="primary"
              size="sm"
              onClick={() => this.setState({ error: null })}
            >
              Retry
            </Button>
          </div>
        )
      );
    }
    return this.props.children;
  }
}

// ── AI insights strip ────────────────────────────────────────────────────────

const INSIGHT_COLORS: Record<string, string> = {
  bar: "var(--wb-primary)",
  pie: "var(--wb-accent)",
  histogram: "var(--wb-success)",
  scatter: "var(--wb-warning)",
  composed: "var(--wb-info)",
};

function InsightStrip(props: {
  suggestions: ChartSuggestion[];
  loading: boolean;
  err: string | null;
}) {
  if (props.loading) {
    return (
      <div style={{ padding: "12px 0", color: "var(--wb-muted)", fontSize: 13 }}>
        Generating AI insights…
      </div>
    );
  }
  if (props.err) {
    return (
      <div style={{ padding: "12px 0", color: "var(--wb-danger)", fontSize: 13 }}>
        {props.err}
      </div>
    );
  }
  if (!props.suggestions.length) return null;

  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.04em",
          color: "var(--wb-muted)",
          marginBottom: 8,
        }}
      >
        AI-suggested insights
      </div>
      <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 4 }}>
        {props.suggestions.map((sug, idx) => (
          <div
            key={`${sug.title}-${idx}`}
            style={{
              minWidth: 200,
              maxWidth: 280,
              padding: "10px 14px",
              borderRadius: "var(--wb-radius)",
              border: "1px solid var(--wb-border)",
              background: "var(--wb-surface)",
              flexShrink: 0,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: INSIGHT_COLORS[sug.chart_type] || "var(--wb-muted)",
                  flexShrink: 0,
                }}
              />
              <span style={{ fontWeight: 600, fontSize: 13, lineHeight: 1.3 }}>
                {sug.title}
              </span>
            </div>
            <div style={{ fontSize: 12, color: "var(--wb-muted)", lineHeight: 1.4, marginBottom: 6 }}>
              {sug.rationale}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              <Badge tone="neutral">{sug.chart_type}</Badge>
              {sug.columns.slice(0, 3).map((c) => (
                <Badge key={c} tone="info">
                  {c}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Theme ────────────────────────────────────────────────────────────────────

const GW_THEME = {
  light: {
    background: "#ffffff",
    foreground: "#1a1a1a",
    primary: "#0f7b6c",
    "primary-foreground": "#ffffff",
    muted: "#f5f6f7",
    "muted-foreground": "#636363",
    border: "#dde1e6",
    ring: "#0f7b6c",
  },
  dark: {
    background: "#1a1a1a",
    foreground: "#f5f6f7",
    primary: "#0f7b6c",
    "primary-foreground": "#ffffff",
    muted: "#2a2a2a",
    "muted-foreground": "#a0a0a0",
    border: "#3a3a3a",
    ring: "#0f7b6c",
  },
};

// ── Main component ───────────────────────────────────────────────────────────

export function ExplorePanel(props: {
  project: string;
  dataset: string;
  table: string;
  technical: TechProfile | null;
  semantic: SemProfile | null;
  suggestions: ChartSuggestion[];
  sugLoading: boolean;
  sugErr: string | null;
}) {
  const [mode, setMode] = useState<"local" | "server">("local");
  const [previewRows, setPreviewRows] = useState<IRow[] | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [gwError, setGwError] = useState<string | null>(null);

  const fields = useMemo(
    () => (props.technical ? buildFields(props.technical) : []),
    [props.technical],
  );

  const computation = useMemo(
    () => makeComputation(props.project, props.dataset, props.table),
    [props.project, props.dataset, props.table],
  );

  useEffect(() => {
    if (!props.technical) return;
    setLoadingPreview(true);
    fetch(
      `/api/projects/${encodeURIComponent(props.project)}/datasets/${encodeURIComponent(props.dataset)}/tables/${encodeURIComponent(props.table)}/preview?limit=2000`,
    )
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((j: { rows?: IRow[] }) => setPreviewRows(j.rows || []))
      .catch(() => setPreviewRows([]))
      .finally(() => setLoadingPreview(false));
  }, [props.project, props.dataset, props.table, props.technical]);

  const handleGwError = useCallback((err: Error) => {
    console.error("GraphicWalker error:", err);
    setGwError(err.message);
  }, []);

  if (!props.technical) {
    return (
      <Card>
        <p style={{ color: "var(--wb-muted)" }}>
          Explore unlocks after technical profiling. Run it from the Technical tab.
        </p>
      </Card>
    );
  }

  const isLocal = mode === "local";
  const ready = isLocal ? previewRows !== null && !loadingPreview : true;

  return (
    <div>
      <InsightStrip
        suggestions={props.suggestions}
        loading={props.sugLoading}
        err={props.sugErr}
      />

      {/* Mode toggle */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 12,
          fontSize: 13,
        }}
      >
        <span style={{ color: "var(--wb-muted)" }}>Data source:</span>
        <button
          type="button"
          onClick={() => { setMode("local"); setGwError(null); }}
          style={{
            padding: "4px 12px",
            borderRadius: "var(--wb-radius)",
            border: `1px solid ${isLocal ? "var(--wb-primary)" : "var(--wb-border)"}`,
            background: isLocal ? "var(--wb-primary)" : "var(--wb-surface)",
            color: isLocal ? "#fff" : "var(--wb-text)",
            cursor: "pointer",
            fontSize: 13,
            fontWeight: 500,
          }}
        >
          Preview (fast)
        </button>
        <button
          type="button"
          onClick={() => { setMode("server"); setGwError(null); }}
          style={{
            padding: "4px 12px",
            borderRadius: "var(--wb-radius)",
            border: `1px solid ${!isLocal ? "var(--wb-primary)" : "var(--wb-border)"}`,
            background: !isLocal ? "var(--wb-primary)" : "var(--wb-surface)",
            color: !isLocal ? "#fff" : "var(--wb-text)",
            cursor: "pointer",
            fontSize: 13,
            fontWeight: 500,
          }}
        >
          Full dataset (BigQuery)
        </button>
        <span style={{ color: "var(--wb-muted)", fontSize: 12 }}>
          {isLocal
            ? `${previewRows?.length ?? "…"} sample rows`
            : "Queries full table via BigQuery"}
        </span>
      </div>

      {loadingPreview && isLocal ? (
        <Card>
          <p style={{ color: "var(--wb-muted)" }}>Loading preview data…</p>
        </Card>
      ) : ready ? (
        <Card style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ minHeight: 520 }}>
            <GWErrorBoundary
              key={`${mode}-${props.table}`}
              onError={handleGwError}
            >
              {isLocal ? (
                <GraphicWalker
                  key={`local-${props.table}`}
                  fields={fields}
                  data={previewRows || []}
                  appearance="light"
                  uiTheme={GW_THEME}
                  vizThemeConfig="vega"
                  onError={handleGwError}
                />
              ) : (
                <GraphicWalker
                  key={`server-${props.table}`}
                  fields={fields}
                  computation={computation}
                  appearance="light"
                  uiTheme={GW_THEME}
                  vizThemeConfig="vega"
                  onError={handleGwError}
                />
              )}
            </GWErrorBoundary>
            {gwError ? (
              <div style={{ padding: "12px 16px", background: "#fff8f8", borderTop: "1px solid var(--wb-border)", fontSize: 13, color: "var(--wb-danger)" }}>
                Error: {gwError}
              </div>
            ) : null}
          </div>
        </Card>
      ) : null}
    </div>
  );
}
