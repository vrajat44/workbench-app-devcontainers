import { useEffect, useState } from "react";

export interface PreviewResult {
  fq_table: string;
  columns: { name: string; type: string; mode?: string; description?: string | null }[];
  rows: unknown[][];
  preview_row_count: number;
  total_rows: number | null;
  size_bytes: number | null;
}

export function usePreview(project: string, dataset: string, table: string) {
  const [data, setData] = useState<PreviewResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setErr(null);
    fetch(
      `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/preview`,
      { signal: controller.signal },
    )
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(body.detail || `Preview failed: ${r.status}`);
        }
        return r.json();
      })
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") setErr(String(e));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [project, dataset, table]);

  return { data, loading, err };
}
