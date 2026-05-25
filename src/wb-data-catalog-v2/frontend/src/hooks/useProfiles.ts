import { useCallback, useEffect, useState } from "react";
import type { SemColumn, SemProfile, TechProfile } from "../types/profile";

const _techCache = new Map<string, TechProfile>();
const _semCache = new Map<string, SemProfile>();

export interface ProfileStatus {
  technical: string;
  semantic: string;
}

export function useProfileStatus(project: string, dataset: string, table: string, pollMs = 2000) {
  const [status, setStatus] = useState<ProfileStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    return fetch(
      `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/status`,
    )
      .then((r) => r.json())
      .then(setStatus)
      .catch((e) => setErr(String(e)));
  }, [project, dataset, table]);

  const shouldPoll = !status || status.technical === "running" || status.semantic === "running";

  useEffect(() => {
    load();
    if (!shouldPoll) return;
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [load, pollMs, shouldPoll]);

  return { status, err, reload: load };
}

export function useTechProfile(project: string, dataset: string, table: string, enabled: boolean) {
  const cacheKey = `${project}.${dataset}.${table}`;
  const [data, setData] = useState<TechProfile | null>(() => _techCache.get(cacheKey) ?? null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) {
      if (!_techCache.has(cacheKey)) setData(null);
      return;
    }
    if (_techCache.has(cacheKey)) {
      setData(_techCache.get(cacheKey)!);
      return;
    }
    const controller = new AbortController();
    fetch(
      `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/technical`,
      { signal: controller.signal },
    )
      .then(async (r) => {
        if (r.status === 404) {
          setData(null);
          return null;
        }
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((j) => {
        if (j) {
          _techCache.set(cacheKey, j);
          setData(j);
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError") setErr(String(e));
      });
    return () => controller.abort();
  }, [project, dataset, table, enabled, cacheKey]);
  return { data, err };
}

export function useSemProfile(project: string, dataset: string, table: string, enabled: boolean) {
  const cacheKey = `${project}.${dataset}.${table}`;
  const [data, setData] = useState<SemProfile | null>(() => _semCache.get(cacheKey) ?? null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) {
      if (!_semCache.has(cacheKey)) setData(null);
      return;
    }
    if (_semCache.has(cacheKey)) {
      setData(_semCache.get(cacheKey)!);
      return;
    }
    const controller = new AbortController();
    fetch(
      `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/semantic`,
      { signal: controller.signal },
    )
      .then(async (r) => {
        if (r.status === 404) {
          setData(null);
          return null;
        }
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((j) => {
        if (j) {
          _semCache.set(cacheKey, j);
          setData(j);
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError") setErr(String(e));
      });
    return () => controller.abort();
  }, [project, dataset, table, enabled, cacheKey]);
  const save = useCallback(
    async (columns: SemColumn[]) => {
      const url = `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/semantic`;
      const r = await fetch(url, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          columns: columns.map((c) => ({
            name: c.name,
            definition: c.definition,
            sensitivity: c.sensitivity,
            confidence: c.confidence,
            unit_of_measure: c.unit_of_measure,
            measurement_method: c.measurement_method ?? "",
          })),
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const updated: SemProfile = await r.json();
      _semCache.set(cacheKey, updated);
      setData(updated);
      return updated;
    },
    [project, dataset, table],
  );

  return { data, err, save };
}

export async function triggerTechnical(project: string, dataset: string, table: string) {
  const r = await fetch(
    `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/technical`,
    { method: "POST" },
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ job_id: string; status: string }>;
}

export async function triggerSemantic(project: string, dataset: string, table: string) {
  const r = await fetch(
    `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/semantic`,
    { method: "POST" },
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ job_id: string; status: string }>;
}
