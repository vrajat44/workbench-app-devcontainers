import { useCallback, useRef, useState, type CSSProperties, type DragEvent } from "react";

interface DocFile {
  filename: string;
  preview: string;
  size: number;
}

export interface DocUploaderProps {
  docs: DocFile[];
  onDocsChange: (docs: DocFile[]) => void;
  onContextChange: (context: string) => void;
}

const ACCEPT = ".pdf,.md,.txt,.csv,.xlsx";

const dropZoneBase: CSSProperties = {
  border: "2px dashed var(--wb-border)",
  borderRadius: "var(--wb-radius)",
  padding: "32px 20px",
  textAlign: "center",
  cursor: "pointer",
  transition: "border-color 0.15s, background 0.15s",
};

const dropZoneHover: CSSProperties = {
  ...dropZoneBase,
  borderColor: "var(--wb-primary)",
  background: "#f0f9f9",
};

const fileRowStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "10px 14px",
  borderBottom: "1px solid var(--wb-border)",
  fontSize: 13,
};

const deleteBtnStyle: CSSProperties = {
  background: "none",
  border: "none",
  color: "var(--wb-danger)",
  cursor: "pointer",
  fontSize: 14,
  fontWeight: 700,
  padding: "2px 6px",
  borderRadius: 4,
  lineHeight: 1,
  fontFamily: "var(--wb-font)",
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocUploader(props: DocUploaderProps) {
  const { docs, onDocsChange, onContextChange } = props;
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    setError(null);
    setUploading(true);

    const newDocs: DocFile[] = [...docs];
    let combinedContext = docs.map((d) => d.preview).join("\n\n");

    for (const file of Array.from(files)) {
      try {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch("/api/profiling/docs", { method: "POST", body: form });
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || `Upload failed: ${res.status}`);
        }
        const data = await res.json();
        const doc: DocFile = {
          filename: data.filename || file.name,
          preview: data.preview || data.text || "",
          size: file.size,
        };
        newDocs.push(doc);
        combinedContext += (combinedContext ? "\n\n" : "") + doc.preview;
      } catch (e: any) {
        setError(e.message || `Failed to upload ${file.name}`);
      }
    }

    onDocsChange(newDocs);
    onContextChange(combinedContext);
    setUploading(false);
  }, [docs, onDocsChange, onContextChange]);

  const handleDelete = useCallback(async (filename: string) => {
    try {
      await fetch(`/api/profiling/docs/${encodeURIComponent(filename)}`, { method: "DELETE" });
    } catch {
      // best-effort delete
    }
    const remaining = docs.filter((d) => d.filename !== filename);
    onDocsChange(remaining);
    onContextChange(remaining.map((d) => d.preview).join("\n\n"));
  }, [docs, onDocsChange, onContextChange]);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files);
    }
  }, [uploadFiles]);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  return (
    <div>
      <div style={{ fontSize: 16, fontWeight: 600, color: "var(--wb-text)", marginBottom: 4 }}>
        Supporting Documents
      </div>
      <div style={{ fontSize: 13, color: "var(--wb-muted)", marginBottom: 16 }}>
        Upload data dictionaries, ERDs, or domain documentation to improve semantic profiling quality.
      </div>

      <div
        style={dragging ? dropZoneHover : dropZoneBase}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => inputRef.current?.click()}
      >
        <div style={{ fontSize: 28, marginBottom: 8, color: "var(--wb-muted)" }}>
          {uploading ? "Uploading..." : "Drop files here"}
        </div>
        <div style={{ fontSize: 13, color: "var(--wb-muted)", marginBottom: 12 }}>
          or
        </div>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
          disabled={uploading}
          style={{
            background: "var(--wb-surface)",
            border: "1px solid var(--wb-border)",
            borderRadius: 6,
            padding: "6px 16px",
            fontSize: 13,
            fontWeight: 500,
            cursor: uploading ? "not-allowed" : "pointer",
            fontFamily: "var(--wb-font)",
            color: "var(--wb-primary)",
          }}
        >
          Browse files
        </button>
        <div style={{ fontSize: 11, color: "var(--wb-muted)", marginTop: 10 }}>
          Accepts: .pdf, .md, .txt, .csv, .xlsx
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              uploadFiles(e.target.files);
              e.target.value = "";
            }
          }}
        />
      </div>

      {error && (
        <div style={{ color: "var(--wb-danger)", fontSize: 13, marginTop: 8 }}>{error}</div>
      )}

      {docs.length > 0 && (
        <div style={{ marginTop: 16, border: "1px solid var(--wb-border)", borderRadius: "var(--wb-radius)", overflow: "hidden" }}>
          {docs.map((d) => (
            <div key={d.filename} style={fileRowStyle}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, color: "var(--wb-text)" }}>{d.filename}</div>
                <div style={{ fontSize: 11, color: "var(--wb-muted)", marginTop: 2 }}>
                  {formatFileSize(d.size)}
                </div>
                {d.preview && (
                  <div style={{
                    fontSize: 11,
                    color: "var(--wb-muted)",
                    marginTop: 4,
                    lineHeight: 1.5,
                    overflow: "hidden",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                  }}>
                    {d.preview.slice(0, 200)}
                  </div>
                )}
              </div>
              <button style={deleteBtnStyle} onClick={() => handleDelete(d.filename)} title="Remove file">
                X
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
