import type { SemProfile } from "../types/profile";
import { Badge, Card, Stack } from "./rds";

function sensTone(s: string): "neutral" | "info" | "success" | "warn" | "danger" {
  if (s === "PHI") return "danger";
  if (s === "PII") return "warn";
  if (s === "UID") return "info";
  return "neutral";
}

function confTone(c: string): "success" | "warn" | "danger" | "neutral" {
  if (c === "high") return "success";
  if (c === "medium") return "warn";
  if (c === "low") return "danger";
  return "neutral";
}

const th: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "2px solid var(--wb-border)",
  fontSize: 12,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.03em",
  color: "var(--wb-muted)",
  whiteSpace: "nowrap",
};

const td: React.CSSProperties = {
  padding: "8px 10px",
  borderBottom: "1px solid var(--wb-border)",
  verticalAlign: "top",
  fontSize: 13,
};

function StatPair(props: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 8, fontSize: 14 }}>
      <span style={{ color: "var(--wb-muted)", minWidth: 100 }}>{props.label}</span>
      <span style={{ fontWeight: 500 }}>{props.value}</span>
    </div>
  );
}

export function SemProfileView(props: { data: SemProfile | null; loading?: boolean }) {
  if (props.loading) return <p>Loading semantic profile…</p>;
  if (!props.data) return <p style={{ color: "var(--wb-muted)" }}>No semantic profile yet.</p>;

  const d = props.data;
  const v = d.validation;

  return (
    <Stack gap={16}>
      {/* Table-level stats */}
      <Card title="Semantic profile">
        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px 40px", marginBottom: 16 }}>
          <StatPair label="Table" value={d.table} />
          <StatPair label="Profiled at" value={d.profiled_at ? new Date(d.profiled_at).toLocaleString() : "—"} />
          <StatPair label="Model" value={d.model_used || "—"} />
          <StatPair label="Columns" value={d.columns.length} />
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
          <Badge tone={v.status === "pass" ? "success" : "warn"}>Validation: {v.status}</Badge>
        </div>

        {(v.issues?.length ?? 0) > 0 ? (
          <div style={{ fontSize: 14, marginTop: 10, color: "var(--wb-warning)" }}>
            <strong>Issues:</strong> {v.issues?.join("; ")}
          </div>
        ) : null}
      </Card>

      {/* Column-level detail table */}
      <Card title="Column semantics">
        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                {["Column", "Definition", "Sensitivity", "Terminology bindings", "Join paths", "Confidence"].map((h) => (
                  <th key={h} style={th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.columns.map((c) => (
                <tr key={c.name}>
                  <td style={{ ...td, fontWeight: 600, whiteSpace: "nowrap" }}>{c.name}</td>
                  <td style={{ ...td, maxWidth: 380, lineHeight: 1.5 }}>{c.definition}</td>
                  <td style={td}>
                    {c.sensitivity ? <Badge tone={sensTone(c.sensitivity)}>{c.sensitivity}</Badge> : "—"}
                  </td>
                  <td style={{ ...td, fontSize: 12 }}>
                    {(c.terminology_bindings || []).length > 0
                      ? c.terminology_bindings.map((b) => (
                          <div key={`${b.system}-${b.code}`} style={{ marginBottom: 2 }}>
                            <strong>{b.display}</strong>{" "}
                            <span style={{ color: "var(--wb-muted)" }}>({b.system}: {b.code})</span>
                          </div>
                        ))
                      : "—"}
                  </td>
                  <td style={{ ...td, fontSize: 12 }}>
                    {(c.join_paths || []).length > 0
                      ? c.join_paths.map((jp) => (
                          <div key={jp} style={{ fontFamily: "monospace" }}>{jp}</div>
                        ))
                      : "—"}
                  </td>
                  <td style={td}>
                    <Badge tone={confTone(c.confidence)}>{c.confidence}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </Stack>
  );
}
