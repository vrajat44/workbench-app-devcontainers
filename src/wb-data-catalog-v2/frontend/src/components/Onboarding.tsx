import { useState, type CSSProperties } from "react";

/* ── Step definitions ───────────────────────────────────────────────────── */

interface StepDef {
  label: string;
  icon: string;
  body: (totalTables: number) => React.ReactNode;
}

const STEPS: StepDef[] = [
  {
    label: "Profile Your Data",
    icon: "🔍",
    body: (totalTables) => (
      <>
        <p style={bodyText}>
          Click <strong>Profile Wizard</strong> to select tables, configure
          terminology domains, and generate metadata. You can choose technical
          profiling (column stats), semantic profiling (AI-powered definitions
          and sensitivity labels), or both.
        </p>
        <p style={{ ...bodyText, fontWeight: 600, color: "var(--wb-primary)" }}>
          You have {totalTables} table{totalTables === 1 ? "" : "s"} ready to
          profile.
        </p>
      </>
    ),
  },
  {
    label: "Explore Profiles",
    icon: "📊",
    body: () => (
      <p style={bodyText}>
        Click any profiled table to view its technical stats, semantic metadata,
        and key insights. Edit definitions and sensitivity labels directly in
        the UI.
      </p>
    ),
  },
  {
    label: "Discover & Query",
    icon: "💡",
    body: () => (
      <p style={bodyText}>
        Use <strong>Terminology</strong> to browse standardized codes across
        tables. Use <strong>Cohort Builder</strong> to filter and count subjects.
        Use <strong>Data AMA Agent</strong> to ask questions in natural language.
      </p>
    ),
  },
];

/* ── Styles ──────────────────────────────────────────────────────────────── */

const cardStyle: CSSProperties = {
  background: "#fff",
  border: "1px solid var(--wb-border)",
  borderRadius: 12,
  padding: "28px 32px 24px",
  boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
  marginBottom: 24,
};

const stepperRow: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "center",
  gap: 0,
  marginBottom: 28,
  position: "relative",
};

const bodyText: CSSProperties = {
  margin: "0 0 10px",
  fontSize: 14,
  lineHeight: 1.6,
  color: "var(--wb-text)",
};

function circleStyle(active: boolean): CSSProperties {
  return {
    width: 32,
    height: 32,
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 14,
    fontWeight: 700,
    flexShrink: 0,
    cursor: "pointer",
    transition: "background 0.15s, color 0.15s",
    border: "none",
    padding: 0,
    background: active ? "var(--wb-primary)" : "#e8ecef",
    color: active ? "#fff" : "#999",
  };
}

function labelStyle(active: boolean): CSSProperties {
  return {
    fontSize: 12,
    fontWeight: active ? 600 : 400,
    color: active ? "var(--wb-primary)" : "var(--wb-muted)",
    marginTop: 6,
    textAlign: "center",
    lineHeight: 1.3,
    transition: "color 0.15s",
  };
}

const connectorStyle: CSSProperties = {
  flex: 1,
  height: 2,
  background: "#e8ecef",
  alignSelf: "center",
  marginTop: -10,
  minWidth: 32,
  maxWidth: 80,
};

const dismissBtn: CSSProperties = {
  background: "var(--wb-primary)",
  color: "#fff",
  border: "none",
  borderRadius: "var(--wb-radius)",
  padding: "10px 28px",
  fontSize: 14,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "var(--wb-font)",
  transition: "background 0.15s",
};

/* ── Component ───────────────────────────────────────────────────────────── */

export interface OnboardingProps {
  show: boolean;
  onDismiss: () => void;
  totalTables: number;
}

export default function Onboarding({ show, onDismiss, totalTables }: OnboardingProps) {
  const [activeStep, setActiveStep] = useState(0);

  if (!show) return null;

  const step = STEPS[activeStep];

  return (
    <div style={cardStyle}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 4 }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: "var(--wb-text)" }}>
          Welcome to Data Catalog
        </div>
        <div style={{ fontSize: 13, color: "var(--wb-muted)", marginTop: 4 }}>
          Get started in three quick steps
        </div>
      </div>

      {/* Stepper */}
      <div style={stepperRow}>
        {STEPS.map((s, i) => (
          <div key={s.label} style={{ display: "contents" }}>
            {i > 0 && <div style={connectorStyle} />}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                width: 100,
                flexShrink: 0,
              }}
            >
              <button
                type="button"
                onClick={() => setActiveStep(i)}
                style={circleStyle(i === activeStep)}
                aria-label={`Step ${i + 1}: ${s.label}`}
              >
                {i + 1}
              </button>
              <div style={labelStyle(i === activeStep)}>{s.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Active step content */}
      <div
        style={{
          background: "var(--wb-bg, #f5f6f7)",
          borderRadius: 8,
          padding: "20px 24px",
          marginBottom: 20,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          <span style={{ fontSize: 22 }}>{step.icon}</span>
          <span style={{ fontSize: 15, fontWeight: 700, color: "var(--wb-text)" }}>
            {step.label}
          </span>
        </div>
        {step.body(totalTables)}
      </div>

      {/* Dismiss */}
      <div style={{ textAlign: "center" }}>
        <button
          type="button"
          onClick={onDismiss}
          style={dismissBtn}
          onMouseEnter={(e) => {
            (e.target as HTMLButtonElement).style.background = "var(--wb-primary-hover)";
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLButtonElement).style.background = "var(--wb-primary)";
          }}
        >
          Got it, let's start!
        </button>
      </div>
    </div>
  );
}
