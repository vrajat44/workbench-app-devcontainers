import {
  useState,
  useRef,
  useEffect,
  useCallback,
  type ReactNode,
  type CSSProperties,
} from "react";

/* ------------------------------------------------------------------ */
/*  1. Tooltip                                                        */
/* ------------------------------------------------------------------ */

type TooltipPosition = "top" | "bottom" | "left" | "right";

interface TooltipProps {
  text: string;
  children: ReactNode;
  position?: TooltipPosition;
}

const tooltipBase: CSSProperties = {
  position: "absolute",
  zIndex: 10000,
  maxWidth: 250,
  padding: "6px 10px",
  borderRadius: 4,
  background: "#333",
  color: "#fff",
  fontSize: 12,
  lineHeight: 1.4,
  whiteSpace: "pre-wrap",
  pointerEvents: "none",
};

const arrowSize = 5;

function tooltipPlacement(pos: TooltipPosition): CSSProperties {
  switch (pos) {
    case "top":
      return { bottom: "100%", left: "50%", transform: "translateX(-50%)", marginBottom: arrowSize + 2 };
    case "bottom":
      return { top: "100%", left: "50%", transform: "translateX(-50%)", marginTop: arrowSize + 2 };
    case "left":
      return { right: "100%", top: "50%", transform: "translateY(-50%)", marginRight: arrowSize + 2 };
    case "right":
      return { left: "100%", top: "50%", transform: "translateY(-50%)", marginLeft: arrowSize + 2 };
  }
}

function arrowStyle(pos: TooltipPosition): CSSProperties {
  const base: CSSProperties = {
    position: "absolute",
    width: 0,
    height: 0,
    borderStyle: "solid",
  };
  switch (pos) {
    case "top":
      return {
        ...base,
        bottom: -arrowSize,
        left: "50%",
        transform: "translateX(-50%)",
        borderWidth: `${arrowSize}px ${arrowSize}px 0 ${arrowSize}px`,
        borderColor: "#333 transparent transparent transparent",
      };
    case "bottom":
      return {
        ...base,
        top: -arrowSize,
        left: "50%",
        transform: "translateX(-50%)",
        borderWidth: `0 ${arrowSize}px ${arrowSize}px ${arrowSize}px`,
        borderColor: "transparent transparent #333 transparent",
      };
    case "left":
      return {
        ...base,
        right: -arrowSize,
        top: "50%",
        transform: "translateY(-50%)",
        borderWidth: `${arrowSize}px 0 ${arrowSize}px ${arrowSize}px`,
        borderColor: "transparent transparent transparent #333",
      };
    case "right":
      return {
        ...base,
        left: -arrowSize,
        top: "50%",
        transform: "translateY(-50%)",
        borderWidth: `${arrowSize}px ${arrowSize}px ${arrowSize}px 0`,
        borderColor: "transparent #333 transparent transparent",
      };
  }
}

export function Tooltip({ text, children, position = "top" }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = useCallback(() => {
    timerRef.current = setTimeout(() => setVisible(true), 200);
  }, []);

  const hide = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setVisible(false);
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <span
      style={{ position: "relative", display: "inline-flex" }}
      onMouseEnter={show}
      onMouseLeave={hide}
    >
      {children}
      {visible && (
        <span style={{ ...tooltipBase, ...tooltipPlacement(position) }}>
          {text}
          <span style={arrowStyle(position)} />
        </span>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  2. HelpIcon                                                       */
/* ------------------------------------------------------------------ */

interface HelpIconProps {
  title: string;
  content: string;
  size?: number;
}

export function HelpIcon({ title, content, size = 16 }: HelpIconProps) {
  const [open, setOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);
  const iconRef = useRef<HTMLSpanElement>(null);

  /* close on outside click */
  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node) &&
        iconRef.current &&
        !iconRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const iconStyle: CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    width: size,
    height: size,
    borderRadius: "50%",
    border: "1px solid var(--wb-muted, #637381)",
    color: "var(--wb-muted, #637381)",
    fontSize: size * 0.6,
    fontWeight: 700,
    cursor: "pointer",
    lineHeight: 1,
    userSelect: "none",
    flexShrink: 0,
  };

  const popoverStyle: CSSProperties = {
    position: "absolute",
    zIndex: 10001,
    top: size + 6,
    right: 0,
    width: 320,
    maxWidth: 320,
    background: "#fff",
    border: "1px solid var(--wb-border, #dde3e8)",
    borderRadius: 6,
    boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
    padding: "12px 16px",
  };

  const closeBtnStyle: CSSProperties = {
    position: "absolute",
    top: 8,
    right: 10,
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: 16,
    lineHeight: 1,
    color: "var(--wb-muted, #637381)",
    padding: 0,
  };

  return (
    <span ref={iconRef} style={{ position: "relative", display: "inline-flex" }}>
      <span style={iconStyle} onClick={() => setOpen((o) => !o)} role="button" aria-label="Help">
        ?
      </span>
      {open && (
        <div ref={popoverRef} style={popoverStyle}>
          <button style={closeBtnStyle} onClick={() => setOpen(false)} aria-label="Close">
            &times;
          </button>
          <div
            style={{
              fontWeight: 600,
              fontSize: 14,
              color: "var(--wb-text, #1d2d35)",
              marginBottom: 6,
              paddingRight: 18,
            }}
          >
            {title}
          </div>
          <div
            style={{
              fontSize: 13,
              color: "var(--wb-muted, #637381)",
              lineHeight: 1.6,
            }}
          >
            {content}
          </div>
        </div>
      )}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  3. HelpPanel                                                      */
/* ------------------------------------------------------------------ */

interface HelpTip {
  title: string;
  body: string;
}

const helpContent: Record<string, HelpTip[]> = {
  catalog: [
    {
      title: "Getting Started",
      body: "The Data Catalog shows all BigQuery datasets in your project. Click a dataset name to expand it and see its tables.",
    },
    {
      title: "Profiling",
      body: "Select tables and click ‘Profile entire project’ to generate metadata. Technical profiling captures column statistics (null rates, distinct counts, top values). Semantic profiling uses AI to add business names, definitions, and sensitivity labels.",
    },
    {
      title: "Filtering",
      body: "Use the search box to find tables by name. Use the profiling filter to show only profiled or unprofiled tables.",
    },
    {
      title: "Bulk Actions",
      body: "Select multiple tables using checkboxes, then use the floating action bar to profile them together.",
    },
  ],
  table: [
    {
      title: "Preview",
      body: "Shows a sample of 50 rows from the table. Use this to understand what the data looks like.",
    },
    {
      title: "Technical Profile",
      body: "Column-level statistics: data types, null rates, distinct counts, top values, patterns, and anomalies. Generated from BigQuery queries, no AI involved.",
    },
    {
      title: "Semantic Profile",
      body: "AI-generated metadata: business names, definitions, sensitivity labels (HIPAA Safe Harbor codes), terminology bindings, join paths, and cohort dimensions. You can edit definitions and sensitivity labels directly.",
    },
    {
      title: "Key Insights",
      body: "AI-suggested chart visualizations based on the table’s profile. Requires technical profiling to be complete.",
    },
    {
      title: "Interactive Explorer",
      body: "Drag-and-drop visual analytics powered by Graphic Walker. Create charts, pivot tables, and explore data interactively.",
    },
  ],
  chat: [
    {
      title: "Q&A Mode",
      body: "Ask natural language questions about your tables and metadata. The agent uses profiling data to answer accurately. Examples: ‘What tables have diagnosis data?’, ‘Explain the SUBJID column’.",
    },
    {
      title: "Agent Mode",
      body: "The agent can generate and execute BigQuery SQL queries. It uses your table profiles to write accurate queries. Examples: ‘Count patients with diabetes’, ‘Show top 10 diagnosis codes’.",
    },
    {
      title: "Context Loading",
      body: "Click ‘Load full details’ to give the agent access to complete column-level statistics. This enables more precise answers about specific columns, values, and patterns.",
    },
  ],
  terminology: [
    {
      title: "Terminology Registry",
      body: "Shows standardized codes (LOINC, SNOMED, ICD-10, etc.) discovered across all profiled tables. Each entry shows which columns in which tables use that code.",
    },
  ],
  cohorts: [
    {
      title: "Cohort Builder",
      body: "Build patient cohorts by applying filters across profiled tables. The builder uses semantic profiles to identify filterable dimensions and join paths between tables.",
    },
  ],
};

interface HelpPanelProps {
  page: string;
  open: boolean;
  onClose: () => void;
}

function TipSection({ tip }: { tip: HelpTip }) {
  const [expanded, setExpanded] = useState(false);

  const sectionStyle: CSSProperties = {
    borderLeft: "2px solid var(--wb-primary, #1a5c5e)",
    padding: "8px 0 8px 14px",
    marginBottom: 4,
  };

  const titleStyle: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    cursor: "pointer",
    fontWeight: 600,
    fontSize: 13,
    color: "var(--wb-text, #1d2d35)",
    userSelect: "none",
    background: "none",
    border: "none",
    padding: 0,
    width: "100%",
    textAlign: "left",
    fontFamily: "inherit",
  };

  const chevron = expanded ? "▾" : "▸";

  return (
    <div style={sectionStyle}>
      <button style={titleStyle} onClick={() => setExpanded((e) => !e)}>
        <span style={{ fontSize: 11, lineHeight: 1, flexShrink: 0 }}>{chevron}</span>
        {tip.title}
      </button>
      {expanded && (
        <div
          style={{
            marginTop: 6,
            fontSize: 13,
            color: "var(--wb-muted, #637381)",
            lineHeight: 1.6,
            paddingLeft: 17,
          }}
        >
          {tip.body}
        </div>
      )}
    </div>
  );
}

export function HelpPanel({ page, open, onClose }: HelpPanelProps) {
  const tips = helpContent[page] ?? [];

  /* prevent body scroll when open */
  useEffect(() => {
    if (open) {
      const prev = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = prev;
      };
    }
  }, [open]);

  if (!open) return null;

  const overlayStyle: CSSProperties = {
    position: "fixed",
    inset: 0,
    zIndex: 1499,
    background: "rgba(0,0,0,0.1)",
  };

  const panelStyle: CSSProperties = {
    position: "fixed",
    top: 0,
    right: 0,
    width: 340,
    height: "100vh",
    zIndex: 1500,
    display: "flex",
    flexDirection: "column",
    background: "var(--wb-surface, #f8f9fa)",
    boxShadow: "-4px 0 24px rgba(0,0,0,0.12)",
    animation: "helpPanelSlideIn 200ms ease-out",
  };

  const headerStyle: CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 18px",
    background: "var(--wb-primary, #1a5c5e)",
    color: "#fff",
    flexShrink: 0,
  };

  const closeBtnStyle: CSSProperties = {
    background: "none",
    border: "none",
    color: "#fff",
    fontSize: 20,
    cursor: "pointer",
    lineHeight: 1,
    padding: 0,
  };

  const bodyStyle: CSSProperties = {
    flex: 1,
    overflowY: "auto",
    padding: "16px 18px",
  };

  return (
    <>
      {/* Inject keyframes once */}
      <style>{`
        @keyframes helpPanelSlideIn {
          from { transform: translateX(100%); }
          to   { transform: translateX(0); }
        }
      `}</style>

      {/* Backdrop */}
      <div style={overlayStyle} onClick={onClose} />

      {/* Panel */}
      <div style={panelStyle}>
        <div style={headerStyle}>
          <span style={{ fontWeight: 600, fontSize: 15 }}>Help</span>
          <button style={closeBtnStyle} onClick={onClose} aria-label="Close help panel">
            &times;
          </button>
        </div>

        <div style={bodyStyle}>
          {tips.length === 0 && (
            <div style={{ color: "var(--wb-muted, #637381)", fontSize: 13 }}>
              No help tips available for this page.
            </div>
          )}
          {tips.map((tip) => (
            <TipSection key={tip.title} tip={tip} />
          ))}
        </div>
      </div>
    </>
  );
}
