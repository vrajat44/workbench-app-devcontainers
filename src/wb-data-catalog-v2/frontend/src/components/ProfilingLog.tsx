import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Button } from "./rds";
import type { BulkStatusResponse } from "../types/bulk";

export interface ProfilingLogProps {
  batchId: string;
  onComplete: () => void;
}

interface LogEntry {
  ts: string;
  level: "info" | "warn" | "error" | "success";
  msg: string;
}

const POLL_INTERVAL = 2000;

const containerStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  height: "100%",
  gap: 16,
};

const progressBarBg: CSSProperties = {
  height: 8,
  borderRadius: 4,
  background: "#e8ecef",
  overflow: "hidden",
};

const logContainerStyle: CSSProperties = {
  flex: 1,
  background: "#1e1e2e",
  borderRadius: "var(--wb-radius)",
  padding: "12px 16px",
  overflow: "auto",
  fontFamily: "monospace",
  fontSize: 12,
  lineHeight: 1.7,
  minHeight: 300,
};

const levelColors: Record<string, { bg: string; color: string }> = {
  info: { bg: "#3a3a4a", color: "#a0a0b0" },
  warn: { bg: "#4a3a20", color: "#f0c040" },
  error: { bg: "#4a2020", color: "#f06060" },
  success: { bg: "#204a20", color: "#60d060" },
};

const levelBadge = (level: string): CSSProperties => {
  const c = levelColors[level] || levelColors.info;
  return {
    display: "inline-block",
    padding: "1px 6px",
    borderRadius: 3,
    fontSize: 10,
    fontWeight: 700,
    textTransform: "uppercase",
    background: c.bg,
    color: c.color,
    marginRight: 8,
    fontFamily: "monospace",
    letterSpacing: "0.04em",
  };
};

const summaryCardStyle: CSSProperties = {
  background: "var(--wb-surface)",
  borderRadius: "var(--wb-radius)",
  padding: 20,
  border: "1px solid var(--wb-border)",
  marginTop: 16,
};

export function ProfilingLog(props: ProfilingLogProps) {
  const { batchId, onComplete } = props;
  const [status, setStatus] = useState<BulkStatusResponse | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevLogCount = useRef(0);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/bulk-profile/${batchId}`);
        if (!res.ok) {
          setError(`Status check failed: ${res.status}`);
          return;
        }
        const data: BulkStatusResponse & { logs?: LogEntry[] } = await res.json();
        setStatus(data);

        if (data.logs && data.logs.length > prevLogCount.current) {
          const newEntries = data.logs.slice(prevLogCount.current);
          setLogs((prev) => [...prev, ...newEntries]);
          prevLogCount.current = data.logs.length;
        }

        if (data.status !== "running") {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          // Add a synthetic completion log entry
          const done = (data.technical?.done || 0) + (data.semantic?.done || 0);
          const failed = (data.technical?.failed || 0) + (data.semantic?.failed || 0);
          setLogs((prev) => [
            ...prev,
            {
              ts: new Date().toISOString(),
              level: failed > 0 ? "warn" : "success",
              msg: `Batch ${data.status}: ${done} profiles generated, ${failed} failed.`,
            },
          ]);
        }
      } catch (e: any) {
        setError(e.message || "Polling error");
      }
    };

    // Initial fetch
    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [batchId]);

  // Auto-scroll to bottom
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  const total = status?.total || 0;
  const techDone = status?.technical?.done || 0;
  const semDone = status?.semantic?.done || 0;
  const techFailed = status?.technical?.failed || 0;
  const semFailed = status?.semantic?.failed || 0;
  const done = techDone + semDone;
  const failed = techFailed + semFailed;
  const totalJobs = total * (status?.mode === "both" ? 2 : 1);
  const completedJobs = done + failed + (status?.technical?.skipped || 0) + (status?.semantic?.skipped || 0);
  const pct = totalJobs > 0 ? Math.round((completedJobs / totalJobs) * 100) : 0;
  const isRunning = status?.status === "running";

  return (
    <div style={containerStyle}>
      {/* Progress header */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div style={{ fontSize: 16, fontWeight: 600, color: "var(--wb-text)" }}>
            {isRunning ? "Profiling in Progress..." : status?.status === "completed" ? "Profiling Complete" : status ? "Profiling Finished" : "Starting..."}
          </div>
          <div style={{ fontSize: 13, color: "var(--wb-muted)" }}>
            {completedJobs} / {totalJobs} tasks ({pct}%)
          </div>
        </div>
        <div style={progressBarBg}>
          <div
            style={{
              height: "100%",
              width: `${pct}%`,
              background: failed > 0 ? "var(--wb-warning)" : "var(--wb-primary)",
              borderRadius: 4,
              transition: "width 0.3s",
            }}
          />
        </div>
      </div>

      {error && (
        <div style={{ color: "var(--wb-danger)", fontSize: 13 }}>{error}</div>
      )}

      {/* Log viewer */}
      <div style={logContainerStyle}>
        {logs.length === 0 && (
          <div style={{ color: "#606070" }}>Waiting for log output...</div>
        )}
        {logs.map((entry, i) => (
          <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
            <span style={{ color: "#606070", whiteSpace: "nowrap", minWidth: 85 }}>
              {new Date(entry.ts).toLocaleTimeString()}
            </span>
            <span style={levelBadge(entry.level)}>{entry.level.toUpperCase()}</span>
            <span style={{ color: entry.level === "error" ? "#f06060" : entry.level === "warn" ? "#f0c040" : entry.level === "success" ? "#60d060" : "#c8c8d8" }}>
              {entry.msg}
            </span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>

      {/* Summary card when done */}
      {!isRunning && status && (
        <div style={summaryCardStyle}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--wb-text)", marginBottom: 12 }}>
            Summary
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--wb-muted)", textTransform: "uppercase", fontWeight: 600, marginBottom: 4 }}>Tables</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--wb-text)" }}>{total}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--wb-muted)", textTransform: "uppercase", fontWeight: 600, marginBottom: 4 }}>Completed</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--wb-success)" }}>{done}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--wb-muted)", textTransform: "uppercase", fontWeight: 600, marginBottom: 4 }}>Failed</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: failed > 0 ? "var(--wb-danger)" : "var(--wb-text)" }}>{failed}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--wb-muted)", textTransform: "uppercase", fontWeight: 600, marginBottom: 4 }}>Duration</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--wb-text)" }}>
                {status.started_at && status.finished_at
                  ? `${Math.round((new Date(status.finished_at).getTime() - new Date(status.started_at).getTime()) / 1000)}s`
                  : "--"}
              </div>
            </div>
          </div>
          <Button variant="primary" onClick={onComplete}>
            View Catalog
          </Button>
        </div>
      )}
    </div>
  );
}
