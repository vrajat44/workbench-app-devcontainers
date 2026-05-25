import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiConfig, CatalogResponse } from "../types/catalog";

export function useConfig() {
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const r = await fetch("/api/config");
      const data: ApiConfig = await r.json();
      setConfig(data);
      return data;
    } catch {
      setErr("Failed to load /api/config");
      return null;
    }
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

export function useCatalog(dataProject: string, refreshKey = 0) {
  const [data, setData] = useState<CatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const prevProject = useRef(dataProject);

  useEffect(() => {
    if (!dataProject) {
      setData(null);
      setLoading(false);
      return;
    }
    const projectChanged = prevProject.current !== dataProject;
    prevProject.current = dataProject;

    if (projectChanged) setData(null);
    setLoading(true);
    setErr(null);

    let cancelled = false;
    fetch(refreshKey > 0 ? "/api/catalog?refresh=true" : "/api/catalog")
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setErr(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [dataProject, refreshKey]);

  return { data, loading, err };
}
