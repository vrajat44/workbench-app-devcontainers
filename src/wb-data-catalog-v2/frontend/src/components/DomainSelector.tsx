import { useEffect, useState, type CSSProperties } from "react";
import { Badge } from "./rds";

interface DomainSystem {
  uri: string;
  use_case: string;
}

interface DomainPreset {
  key: string;
  label: string;
  description: string;
  systems: DomainSystem[];
}

export interface DomainSelectorProps {
  selected: string[];
  onSelectionChange: (domains: string[]) => void;
}

const domainCardStyle: CSSProperties = {
  border: "1px solid var(--wb-border)",
  borderRadius: "var(--wb-radius)",
  padding: "14px 16px",
  marginBottom: 10,
  background: "var(--wb-surface)",
};

const systemRowStyle: CSSProperties = {
  display: "flex",
  gap: 8,
  padding: "4px 0 4px 28px",
  fontSize: 12,
  color: "var(--wb-muted)",
  lineHeight: 1.5,
};

export function DomainSelector(props: DomainSelectorProps) {
  const { selected, onSelectionChange } = props;
  const [domains, setDomains] = useState<DomainPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/terminology-domains")
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((data) => {
        if (cancelled) return;
        setDomains(data.domains || data || []);
        setLoading(false);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message || "Failed to load domains");
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, []);

  const toggleDomain = (key: string) => {
    if (key === "custom") return;
    if (selected.includes(key)) {
      onSelectionChange(selected.filter((d) => d !== key));
    } else {
      onSelectionChange([...selected, key]);
    }
  };

  const toggleExpand = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  if (loading) {
    return <div style={{ padding: 20, color: "var(--wb-muted)", fontSize: 13 }}>Loading terminology domains...</div>;
  }

  if (error) {
    return <div style={{ padding: 20, color: "var(--wb-danger)", fontSize: 13 }}>Error: {error}</div>;
  }

  return (
    <div>
      <div style={{ fontSize: 16, fontWeight: 600, color: "var(--wb-text)", marginBottom: 4 }}>
        Terminology Domains
      </div>
      <div style={{ fontSize: 13, color: "var(--wb-muted)", marginBottom: 16 }}>
        Select which terminology systems to include during semantic profiling.
      </div>

      {/* Custom domain - always checked */}
      <div style={{ ...domainCardStyle, opacity: 0.7 }}>
        <label style={{ display: "flex", alignItems: "flex-start", gap: 10, cursor: "not-allowed" }}>
          <input
            type="checkbox"
            checked
            disabled
            style={{ width: 16, height: 16, marginTop: 2, accentColor: "var(--wb-primary)" }}
          />
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, color: "var(--wb-text)" }}>
              Custom
              <Badge tone="info">Always included</Badge>
            </div>
            <div style={{ fontSize: 12, color: "var(--wb-muted)", marginTop: 2 }}>
              Project-specific terminology and business rules.
            </div>
          </div>
        </label>
      </div>

      {domains.map((d) => {
        const isCustom = d.key === "custom";
        if (isCustom) return null;
        const checked = selected.includes(d.key);
        const isExpanded = expanded.has(d.key);

        return (
          <div key={d.key} style={domainCardStyle}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleDomain(d.key)}
                style={{ width: 16, height: 16, marginTop: 2, accentColor: "var(--wb-primary)", cursor: "pointer" }}
              />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 14, color: "var(--wb-text)" }}>{d.label}</span>
                  {d.systems && d.systems.length > 0 && (
                    <Badge tone="neutral">{d.systems.length} systems</Badge>
                  )}
                </div>
                <div style={{ fontSize: 12, color: "var(--wb-muted)", marginTop: 2, lineHeight: 1.5 }}>
                  {d.description}
                </div>
                {d.systems && d.systems.length > 0 && (
                  <button
                    onClick={() => toggleExpand(d.key)}
                    style={{
                      background: "none",
                      border: "none",
                      padding: "4px 0 0",
                      fontSize: 11,
                      color: "var(--wb-primary)",
                      cursor: "pointer",
                      fontFamily: "var(--wb-font)",
                      fontWeight: 500,
                    }}
                  >
                    {isExpanded ? "Hide systems" : "Show systems"}
                  </button>
                )}
              </div>
            </div>

            {isExpanded && d.systems && (
              <div style={{ marginTop: 8, borderTop: "1px solid var(--wb-border)", paddingTop: 8 }}>
                {d.systems.map((s, i) => (
                  <div key={i} style={systemRowStyle}>
                    <span style={{ fontFamily: "monospace", fontSize: 11, wordBreak: "break-all", flex: 1 }}>{s.uri}</span>
                    <span style={{ whiteSpace: "nowrap", color: "#636363" }}>{s.use_case}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
