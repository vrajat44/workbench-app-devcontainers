import { useMemo, useState, type CSSProperties } from "react";
import { Button, Badge, Card } from "./rds";
import { DatasetSelector } from "./DatasetSelector";
import { DomainSelector } from "./DomainSelector";
import { DocUploader } from "./DocUploader";
import { ProfilingLog } from "./ProfilingLog";
import type { BulkStatusResponse } from "../types/bulk";

/* ── Types ──────────────────────────────────────────────────────────────── */

export interface ProfilingWizardProps {
  datasets: Array<{
    dataset_id: string;
    tables: Array<{
      table_id: string;
      fq_table: string;
      profiling: { technical: string; semantic: string };
    }>;
  }>;
  loadingDatasets?: boolean;
  project: string;
  onClose: () => void;
  onComplete: () => void;
}

interface DocFile {
  filename: string;
  preview: string;
  size: number;
}

/* ── Step labels ────────────────────────────────────────────────────────── */

const STEPS = ["Select Tables", "Configure", "Review", "Profiling"];

/* ── Styles ─────────────────────────────────────────────────────────────── */

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  zIndex: 1000,
  background: "#fff",
  display: "flex",
  flexDirection: "column",
  fontFamily: "var(--wb-font)",
};

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "16px 32px",
  borderBottom: "1px solid var(--wb-border)",
  flexShrink: 0,
};

const stepBarStyle: CSSProperties = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  gap: 0,
  padding: "16px 32px",
  borderBottom: "1px solid var(--wb-border)",
  flexShrink: 0,
  background: "var(--wb-surface)",
};

const bodyStyle: CSSProperties = {
  flex: 1,
  overflow: "auto",
  padding: "24px 40px",
};

const footerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "16px 32px",
  borderTop: "1px solid var(--wb-border)",
  flexShrink: 0,
};

const closeBtnStyle: CSSProperties = {
  background: "none",
  border: "none",
  fontSize: 22,
  cursor: "pointer",
  color: "var(--wb-muted)",
  padding: "4px 8px",
  borderRadius: 4,
  fontFamily: "var(--wb-font)",
  lineHeight: 1,
};

/* ── Step indicator ─────────────────────────────────────────────────────── */

function StepIndicator(props: { step: number }) {
  return (
    <div style={stepBarStyle}>
      {STEPS.map((label, i) => {
        const isCurrent = i === props.step;
        const isDone = i < props.step;

        return (
          <div key={label} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 13,
                  fontWeight: 700,
                  background: isCurrent ? "var(--wb-primary)" : isDone ? "var(--wb-success, #1a7f37)" : "#e8ecef",
                  color: isCurrent || isDone ? "#fff" : "var(--wb-muted)",
                  transition: "background 0.2s",
                }}
              >
                {isDone ? "✓" : i + 1}
              </div>
              <span
                style={{
                  fontSize: 13,
                  fontWeight: isCurrent ? 600 : 400,
                  color: isCurrent ? "var(--wb-text)" : "var(--wb-muted)",
                }}
              >
                {label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div
                style={{
                  width: 48,
                  height: 2,
                  background: isDone ? "var(--wb-success, #1a7f37)" : "#e8ecef",
                  margin: "0 12px",
                  borderRadius: 1,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Review step ────────────────────────────────────────────────────────── */

function ReviewStep(props: {
  selectedTables: string[];
  selectedDomains: string[];
  uploadedDocs: DocFile[];
  forceReprofile: boolean;
  profilingMode: string;
  datasets: ProfilingWizardProps["datasets"];
}) {
  const { selectedTables, selectedDomains, uploadedDocs, forceReprofile, profilingMode, datasets } = props;

  // Count by dataset
  const datasetCounts: Record<string, number> = {};
  for (const ds of datasets) {
    const count = ds.tables.filter((t) => selectedTables.includes(t.fq_table)).length;
    if (count > 0) datasetCounts[ds.dataset_id] = count;
  }

  return (
    <div>
      <div style={{ fontSize: 16, fontWeight: 600, color: "var(--wb-text)", marginBottom: 20 }}>
        Review Configuration
      </div>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--wb-text)", marginBottom: 12 }}>Tables</div>
        <div style={{ fontSize: 13, color: "var(--wb-text)", marginBottom: 8 }}>
          {selectedTables.length} table(s) across {Object.keys(datasetCounts).length} dataset(s)
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {Object.entries(datasetCounts).map(([dsId, count]) => (
            <Badge key={dsId} tone="info">{dsId}: {count}</Badge>
          ))}
        </div>
        <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Badge tone="info">
            {profilingMode === "both" ? "Technical + Semantic" : profilingMode === "technical" ? "Technical only" : "Semantic only"}
          </Badge>
          {forceReprofile && <Badge tone="warn">Force re-profile enabled</Badge>}
        </div>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--wb-text)", marginBottom: 12 }}>Terminology Domains</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <Badge tone="info">Custom (always)</Badge>
          {selectedDomains.map((d) => (
            <Badge key={d} tone="info">{d}</Badge>
          ))}
          {selectedDomains.length === 0 && (
            <span style={{ fontSize: 13, color: "var(--wb-muted)" }}>No additional domains selected</span>
          )}
        </div>
      </Card>

      <Card>
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--wb-text)", marginBottom: 12 }}>Supporting Documents</div>
        {uploadedDocs.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {uploadedDocs.map((d) => (
              <div key={d.filename} style={{ fontSize: 13, color: "var(--wb-text)" }}>
                {d.filename}
                <span style={{ color: "var(--wb-muted)", marginLeft: 8, fontSize: 12 }}>
                  ({d.size < 1024 ? `${d.size} B` : `${(d.size / 1024).toFixed(1)} KB`})
                </span>
              </div>
            ))}
          </div>
        ) : (
          <span style={{ fontSize: 13, color: "var(--wb-muted)" }}>No documents uploaded</span>
        )}
      </Card>
    </div>
  );
}

/* ── Main wizard ────────────────────────────────────────────────────────── */

export function ProfilingWizard(props: ProfilingWizardProps) {
  const { datasets, onClose, onComplete } = props;

  const [step, setStep] = useState(0);
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [uploadedDocs, setUploadedDocs] = useState<DocFile[]>([]);
  const [docContext, setDocContext] = useState("");
  const [forceReprofile, setForceReprofile] = useState(false);
  const [profilingMode, setProfilingMode] = useState<"both" | "technical" | "semantic">("both");
  const [batchId, setBatchId] = useState<string | null>(null);
  const [batchStatus] = useState<BulkStatusResponse | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  const allTableMap = useMemo(() => {
    const map = new Map<string, { technical: string; semantic: string }>();
    for (const ds of datasets) {
      for (const t of ds.tables) map.set(t.fq_table, t.profiling);
    }
    return map;
  }, [datasets]);

  const tablesLackingTech = useMemo(() => {
    if (profilingMode !== "semantic") return [];
    return selectedTables.filter((fq) => allTableMap.get(fq)?.technical !== "available");
  }, [profilingMode, selectedTables, allTableMap]);

  const canNext =
    step === 0 ? selectedTables.length > 0 :
    step === 1 ? (profilingMode !== "semantic" || tablesLackingTech.length === 0) :
    step === 2 ? true :
    false;

  const handleStart = async () => {
    setStartError(null);
    try {
      const res = await fetch("/api/bulk-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tables: selectedTables,
          mode: profilingMode,
          force: forceReprofile,
          domains: selectedDomains,
          doc_context: docContext || undefined,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || `Failed: ${res.status}`);
      }
      const data = await res.json();
      setBatchId(data.batch_id);
      setStep(3);
    } catch (e: any) {
      setStartError(e.message || "Failed to start profiling");
    }
  };

  const handleNext = () => {
    if (step === 2) {
      handleStart();
    } else if (step < 3) {
      setStep(step + 1);
    }
  };

  const handleBack = () => {
    if (step > 0 && step < 3) {
      setStep(step - 1);
    }
  };

  const isProfilingStep = step === 3;

  return (
    <div style={overlayStyle}>
      {/* Header */}
      <div style={headerStyle}>
        <div style={{ fontSize: 18, fontWeight: 700, color: "var(--wb-text)" }}>
          Profiling Wizard
        </div>
        <button
          style={{ ...closeBtnStyle, opacity: isProfilingStep && !batchStatus ? 0.3 : 1, cursor: isProfilingStep && !batchStatus ? "not-allowed" : "pointer" }}
          onClick={() => { if (!isProfilingStep || batchStatus) onClose(); }}
          disabled={isProfilingStep && !batchStatus}
          title="Close"
        >
          X
        </button>
      </div>

      {/* Step indicator */}
      <StepIndicator step={step} />

      {/* Body */}
      <div style={bodyStyle}>
        {step === 0 && (
          <>
            <DatasetSelector
              datasets={datasets}
              selected={selectedTables}
              onSelectionChange={setSelectedTables}
              force={forceReprofile}
              onForceChange={setForceReprofile}
            />
            {props.loadingDatasets && (
              <div style={{ padding: "12px 16px", fontSize: 13, color: "var(--wb-muted)", display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ display: "inline-block", width: 14, height: 14, border: "2px solid var(--wb-border)", borderTopColor: "var(--wb-primary)", borderRadius: "50%", animation: "wb-splash-spin 0.8s linear infinite" }} />
                Loading datasets...
              </div>
            )}
          </>
        )}

        {step === 1 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 32, maxWidth: 720, margin: "0 auto" }}>
            <Card style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--wb-text)", marginBottom: 12 }}>Profiling Mode</div>
              {[
                { value: "both", label: "Technical + Semantic", desc: "Full profiling with column stats and AI metadata" },
                { value: "technical", label: "Technical only", desc: "Column stats, null rates, patterns, value distributions" },
                { value: "semantic", label: "Semantic only", desc: "AI definitions, terminology bindings, sensitivity (requires existing tech profiles)" },
              ].map((opt) => (
                <label key={opt.value} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "8px 0", cursor: "pointer" }}>
                  <input type="radio" name="profilingMode" value={opt.value} checked={profilingMode === opt.value} onChange={() => setProfilingMode(opt.value as any)} style={{ marginTop: 3 }} />
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 500 }}>{opt.label}</div>
                    <div style={{ fontSize: 12, color: "var(--wb-muted)" }}>{opt.desc}</div>
                  </div>
                </label>
              ))}
              {profilingMode === "semantic" && tablesLackingTech.length > 0 && (
                <div style={{ marginTop: 8, padding: "10px 14px", background: "#fff3e0", borderRadius: "var(--wb-radius)", border: "1px solid #ffe0b2" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#e65100", marginBottom: 4 }}>
                    {tablesLackingTech.length} selected table(s) have no technical profile
                  </div>
                  <div style={{ fontSize: 12, color: "#bf360c" }}>
                    Semantic profiling requires technical profiles first. Switch to "Technical + Semantic" or remove these tables.
                  </div>
                </div>
              )}
            </Card>
            <div style={profilingMode === "technical" ? { opacity: 0.4, pointerEvents: "none" as const } : undefined}>
              <DomainSelector selected={selectedDomains} onSelectionChange={setSelectedDomains} />
            </div>
            {profilingMode === "technical" && (
              <div style={{ fontSize: 12, color: "var(--wb-muted)", marginTop: -24 }}>Terminology domains only apply to semantic profiling.</div>
            )}
            <DocUploader docs={uploadedDocs} onDocsChange={setUploadedDocs} onContextChange={setDocContext} />
          </div>
        )}

        {step === 2 && (
          <div>
            <ReviewStep
              selectedTables={selectedTables}
              selectedDomains={selectedDomains}
              uploadedDocs={uploadedDocs}
              forceReprofile={forceReprofile}
              profilingMode={profilingMode}
              datasets={datasets}
            />
            {startError && (
              <div style={{ marginTop: 12, color: "var(--wb-danger)", fontSize: 13 }}>{startError}</div>
            )}
          </div>
        )}

        {step === 3 && batchId && (
          <ProfilingLog batchId={batchId} onComplete={onComplete} />
        )}
      </div>

      {/* Footer */}
      <div style={footerStyle}>
        <div>
          {step > 0 && step < 3 && (
            <Button variant="secondary" onClick={handleBack}>Back</Button>
          )}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {step < 3 && (
            <>
              {step < 2 && (
                <Button variant="primary" onClick={handleNext} disabled={!canNext}>
                  Next
                </Button>
              )}
              {step === 2 && (
                <Button variant="primary" onClick={handleNext} disabled={!canNext}>
                  Start Profiling
                </Button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
