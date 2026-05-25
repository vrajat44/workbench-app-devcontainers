import { useEffect, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Styles                                                             */
/* ------------------------------------------------------------------ */

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  zIndex: 2000,
  background: "#fff",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const wrapperStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 20,
  textAlign: "center",
};

const brandStyle: React.CSSProperties = {
  fontSize: 30,
  fontWeight: 700,
  color: "var(--wb-primary, #1a5c5e)",
  lineHeight: 1.2,
};

const subtitleStyle: React.CSSProperties = {
  fontSize: 14,
  color: "var(--wb-muted, #637381)",
  marginTop: 2,
};

const projectStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 600,
  color: "var(--wb-text, #1d2d35)",
};

const statusStyle: React.CSSProperties = {
  fontSize: 13,
  color: "var(--wb-muted, #637381)",
};

const spinnerSize = 40;
const spinnerBorder = 4;

const spinnerStyle: React.CSSProperties = {
  width: spinnerSize,
  height: spinnerSize,
  border: `${spinnerBorder}px solid #e0e0e0`,
  borderTopColor: "var(--wb-primary, #1a5c5e)",
  borderRadius: "50%",
  animation: "wb-splash-spin 0.8s linear infinite",
};

/* We inject the @keyframes rule once via a <style> tag so that inline
   styles can reference the animation name without a CSS file. */
const keyframesCSS = `
@keyframes wb-splash-spin {
  to { transform: rotate(360deg); }
}`;

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function LoadingSplash({
  visible,
  projectName,
  status,
}: {
  visible: boolean;
  projectName: string;
  status: string;
}) {
  /* Inject keyframes once when the component first mounts as visible. */
  const [injected, setInjected] = useState(false);

  useEffect(() => {
    if (!visible || injected) return;
    const id = "wb-splash-keyframes";
    if (!document.getElementById(id)) {
      const style = document.createElement("style");
      style.id = id;
      style.textContent = keyframesCSS;
      document.head.appendChild(style);
    }
    setInjected(true);
  }, [visible, injected]);

  if (!visible) return null;

  return (
    <div style={overlayStyle}>
      <div style={wrapperStyle}>
        {/* Brand */}
        <div>
          <div style={brandStyle}>workbench</div>
          <div style={subtitleStyle}>Data Catalog v2</div>
        </div>

        {/* Project name */}
        <div style={projectStyle}>{projectName}</div>

        {/* Spinner */}
        <div style={spinnerStyle} />

        {/* Status message */}
        <div style={statusStyle}>{status}</div>
      </div>
    </div>
  );
}
