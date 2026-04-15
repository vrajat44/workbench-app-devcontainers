import { useCallback, useEffect, useState } from "react";
import type { SemProfile, TechProfile } from "../types/profile";

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

  useEffect(() => {
    load();
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [load, pollMs]);

  return { status, err, reload: load };
}

export function useTechProfile(project: string, dataset: string, table: string, enabled: boolean) {
  const [data, setData] = useState<TechProfile | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) {
      setData(null);
      return;
    }
    fetch(
      `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/technical`,
    )
      .then(async (r) => {
        if (r.status === 404) {
          setData(null);
          return null;
        }
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((j) => j && setData(j))
      .catch((e) => setErr(String(e)));
  }, [project, dataset, table, enabled]);
  return { data, err };
}

export function useSemProfile(project: string, dataset: string, table: string, enabled: boolean) {
  const [data, setData] = useState<SemProfile | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) {
      setData(null);
      return;
    }
    fetch(
      `/api/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/profile/semantic`,
    )
      .then(async (r) => {
        if (r.status === 404) {
          setData(null);
          return null;
        }
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((j) => j && setData(j))
      .catch((e) => setErr(String(e)));
  }, [project, dataset, table, enabled]);
  return { data, err };
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
