import { useState, useCallback, useEffect, useRef } from "react";
import type { BulkMode, BulkStatusResponse } from "../types/bulk";
import { useNotification } from "../components/Notifications";

const POLL_INTERVAL = 2000;

export function useBulkProfile(onComplete?: () => void) {
  const { showNotification } = useNotification();
  const [batchId, setBatchId] = useState<string | null>(null);
  const [status, setStatus] = useState<BulkStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startBatch = useCallback(
    async (tables: string[], mode: BulkMode, force = false) => {
      setError(null);
      setStatus(null);
      setLoading(true);
      stopPolling();

      try {
        const res = await fetch("/api/bulk-profile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tables, mode, force }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || `Failed: ${res.status}`);
        }
        const data = await res.json();
        setBatchId(data.batch_id);

        pollRef.current = setInterval(async () => {
          try {
            const r = await fetch(`/api/bulk-profile/${data.batch_id}`);
            if (r.ok) {
              const s: BulkStatusResponse = await r.json();
              setStatus(s);
              if (s.status !== "running") {
                stopPolling();
                setLoading(false);
                onCompleteRef.current?.();
                const done = (s.technical?.done || 0) + (s.semantic?.done || 0);
                const failed = (s.technical?.failed || 0) + (s.semantic?.failed || 0);
                if (failed > 0) {
                  showNotification(`Profiling finished with ${failed} error(s)`, "warning");
                } else {
                  showNotification(`Profiling complete: ${done} profile(s) generated`, "success");
                }
              }
            }
          } catch {}
        }, POLL_INTERVAL);
      } catch (e: any) {
        setError(e.message || "Failed to start bulk profiling");
        setLoading(false);
      }
    },
    [stopPolling],
  );

  const dismiss = useCallback(() => {
    stopPolling();
    setBatchId(null);
    setStatus(null);
    setLoading(false);
    setError(null);
  }, [stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  return { batchId, status, loading, error, startBatch, dismiss };
}
