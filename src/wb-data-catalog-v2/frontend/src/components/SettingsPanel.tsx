import { useState } from "react";
import type { ApiConfig } from "../types/catalog";
import { Badge, Button, Card, Input, Stack } from "./rds";

interface SaveResult extends ApiConfig {
  bucket_status?: { bucket: string; action: string; error?: string };
}

export function SettingsPanel(props: {
  config: ApiConfig | null;
  onSave: (patch: { billing_project?: string; data_project?: string; gemini_model?: string }) => Promise<SaveResult>;
  onSaved: () => void;
}) {
  const c = props.config;
  const [billingProject, setBillingProject] = useState(c?.billing_project ?? "");
  const [dataProject, setDataProject] = useState(c?.data_project ?? "");
  const [model, setModel] = useState(c?.gemini_model ?? "");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [bucketMsg, setBucketMsg] = useState<string | null>(null);

  const derivedBucket = billingProject.trim() ? `metadata-json-${billingProject.trim()}` : null;

  const handleSave = async () => {
    setSaving(true);
    setMsg(null);
    setBucketMsg(null);
    try {
      const result = await props.onSave({
        billing_project: billingProject.trim(),
        data_project: dataProject.trim() || billingProject.trim(),
        gemini_model: model.trim(),
      });
      const bs = result.bucket_status;
      if (bs) {
        if (bs.action === "exists") {
          setBucketMsg(`Bucket "${bs.bucket}" verified.`);
        } else if (bs.action === "error") {
          setBucketMsg(`Bucket warning: ${bs.error}`);
        }
      }
      setMsg("Settings saved. Reloading catalog…");
      props.onSaved();
    } catch (e) {
      setMsg(`Error: ${e}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Stack gap={20}>
      <Card title="Settings" style={{ marginTop: 20 }}>
        <Stack gap={14}>
          <div style={{ fontSize: 14, color: "var(--wb-muted)" }}>
            Configure the GCP project and Gemini model. Profiles are stored in the
            project's existing metadata bucket{" "}
            {derivedBucket ? <Badge tone="info">{derivedBucket}</Badge> : <span>(derived from project ID)</span>}.
          </div>
          <label style={{ fontSize: 14 }}>
            <strong>GCP Project ID</strong> (billing + ADC project)
            <Input
              value={billingProject}
              onChange={setBillingProject}
              placeholder="e.g. my-gcp-project"
            />
          </label>
          <label style={{ fontSize: 14 }}>
            <strong>Data Project ID</strong> (project whose datasets to browse; defaults to GCP Project)
            <Input
              value={dataProject}
              onChange={setDataProject}
              placeholder="same as GCP Project if blank"
            />
          </label>
          <label style={{ fontSize: 14 }}>
            <strong>Gemini Model</strong> (leave blank for auto-detect)
            <Input
              value={model}
              onChange={setModel}
              placeholder="e.g. gemini-2.5-flash"
            />
          </label>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <Button variant="primary" onClick={handleSave} disabled={saving || !billingProject.trim()}>
              {saving ? "Saving…" : "Save & reload"}
            </Button>
            {msg ? <span style={{ fontSize: 14 }}>{msg}</span> : null}
          </div>
          {bucketMsg ? (
            <div
              style={{
                fontSize: 14,
                padding: "8px 12px",
                borderRadius: "var(--wb-radius)",
                background: bucketMsg.includes("warning") ? "#ffebe9" : "#dafbe1",
                color: bucketMsg.includes("warning") ? "var(--wb-danger)" : "var(--wb-success)",
              }}
            >
              {bucketMsg}
            </div>
          ) : null}
        </Stack>
      </Card>
    </Stack>
  );
}
