import { useCallback, useEffect, useState } from "react";
import type { ApiConfig, CatalogResponse } from "../types/catalog";

export function useConfig() {
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then(setConfig)
      .catch(() => setErr("Failed to load /api/config"));
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const save = useCallback(
    async (patch: { billing_project?: string; data_project?: string; gemini_model?: string }) => {
      const r = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!r.ok) throw new Error(await r.text());
      const updated: ApiConfig = await r.json();
      setConfig(updated);
      return updated;
    },
    [],
  );

  return { config, err, reload, save };
}

export function useCatalog(refreshKey = 0) {
  const [data, setData] = useState<CatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (refreshKey < 0) return;
    setLoading(true);
    setErr(null);
    fetch("/api/catalog")
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then(setData)
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  return { data, loading, err };
}
