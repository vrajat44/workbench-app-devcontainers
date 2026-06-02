import { SettingsPanel } from "../components/SettingsPanel";
import type { ApiConfig } from "../types/catalog";

interface SaveResult extends ApiConfig {
  bucket_status?: { bucket: string; action: string; error?: string };
}

export default function SettingsPage(props: {
  config: ApiConfig | null;
  onSave: (patch: { billing_project?: string; data_project?: string; gemini_model?: string }) => Promise<SaveResult>;
  onSaving?: () => void;
  onSaved: () => void;
}) {
  return (
    <div style={{ padding: "32px 40px", maxWidth: 640 }}>
      <h1 style={{ margin: "0 0 8px", fontSize: 24, fontWeight: 700, color: "var(--wb-text)" }}>Settings</h1>
      <p style={{ color: "var(--wb-muted)", margin: "0 0 20px", fontSize: 14 }}>
        Configure your data source, billing project, and AI model
      </p>
      <SettingsPanel config={props.config} onSave={props.onSave} onSaving={props.onSaving} onSaved={props.onSaved} />
    </div>
  );
}
