import { useCallback, useEffect, useRef, useState } from "react";
import type { CatalogDataset } from "../types/catalog";

export interface DatasetStub {
  dataset_id: string;
  tables: CatalogDataset["tables"];
  loaded: boolean;
  loading: boolean;
}

export function useProgressiveCatalog(dataProject: string, refreshKey = 0) {
  const [datasets, setDatasets] = useState<DatasetStub[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const prevProject = useRef(dataProject);

  useEffect(() => {
    if (!dataProject) {
      setDatasets([]);
      setLoadingDatasets(false);
      return;
    }
    const projectChanged = prevProject.current !== dataProject;
    prevProject.current = dataProject;
    if (projectChanged) setDatasets([]);

    setLoadingDatasets(true);
    setErr(null);

    let cancelled = false;
    fetch("/api/datasets")
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((d) => {
        if (cancelled) return;
        const ids: string[] = d.datasets || [];
        setDatasets(ids.map((id) => ({ dataset_id: id, tables: [], loaded: false, loading: false })));
      })
      .catch((e) => {
        if (!cancelled) setErr(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoadingDatasets(false);
      });
    return () => { cancelled = true; };
  }, [dataProject, refreshKey]);

  const loadDataset = useCallback((datasetId: string) => {
    setDatasets((prev) => {
      const ds = prev.find((d) => d.dataset_id === datasetId);
      if (!ds || ds.loaded || ds.loading) return prev;
      return prev.map((d) => d.dataset_id === datasetId ? { ...d, loading: true } : d);
    });

    fetch(`/api/datasets/${encodeURIComponent(datasetId)}/tables`)
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then((d) => {
        setDatasets((prev) =>
          prev.map((ds) =>
            ds.dataset_id === datasetId
              ? { ...ds, tables: d.tables || [], loaded: true, loading: false }
              : ds,
          ),
        );
      })
      .catch(() => {
        setDatasets((prev) =>
          prev.map((ds) =>
            ds.dataset_id === datasetId ? { ...ds, loaded: true, loading: false } : ds,
          ),
        );
      });
  }, []);

  const loadAll = useCallback(() => {
    datasets.forEach((ds) => {
      if (!ds.loaded && !ds.loading) loadDataset(ds.dataset_id);
    });
  }, [datasets, loadDataset]);

  return { datasets, loadingDatasets, err, loadDataset, loadAll };
}
