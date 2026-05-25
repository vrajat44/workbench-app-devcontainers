import { useEffect, useState } from "react";
import { Input, Select, Stack } from "./rds";

export function FilterBar(props: {
  search: string;
  onSearch: (v: string) => void;
  stateFilter: "all" | "none" | "tech" | "full";
  onStateFilter: (v: "all" | "none" | "tech" | "full") => void;
}) {
  const [local, setLocal] = useState(props.search);

  useEffect(() => {
    const timer = setTimeout(() => props.onSearch(local), 300);
    return () => clearTimeout(timer);
  }, [local]);

  useEffect(() => {
    setLocal(props.search);
  }, [props.search]);

  return (
    <Stack gap={8}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
        <Input placeholder="Search tables…" value={local} onChange={setLocal} />
        <Select
          value={props.stateFilter}
          onChange={(v) => props.onStateFilter(v as "all" | "none" | "tech" | "full")}
          options={[
            { value: "all", label: "Profiling: All" },
            { value: "none", label: "Not profiled" },
            { value: "tech", label: "Technical only" },
            { value: "full", label: "Technical + Semantic" },
          ]}
        />
      </div>
    </Stack>
  );
}
