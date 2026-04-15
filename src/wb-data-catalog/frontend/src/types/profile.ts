export interface TechColumn {
  name: string;
  data_type: string;
  nullable?: boolean;
  null_count?: number;
  null_percent?: number;
  distinct_count?: number;
  top_values?: string[];
  value_counts?: Record<string, number>;
  string_stats?: { min_length?: number; max_length?: number; avg_length?: number };
  numeric_stats?: { min?: number; max?: number; median?: number; stddev?: number };
  pattern?: string;
  anomalies?: string[];
}

export interface TechProfile {
  table: string;
  row_count: number;
  size_bytes?: number | null;
  profiled_at: string;
  validation: { status: string; anomalies?: string[]; warnings?: string[] };
  columns: TechColumn[];
}

export interface TerminologyBinding {
  system: string;
  code: string;
  display: string;
}

export interface SemColumn {
  name: string;
  definition: string;
  terminology_bindings: TerminologyBinding[];
  sensitivity: string;
  join_paths: string[];
  confidence: string;
}

export interface SemProfile {
  table: string;
  profiled_at: string;
  model_used: string;
  business_name?: string;
  table_definition?: string;
  validation: { status: string; issues?: string[] };
  columns: SemColumn[];
}
