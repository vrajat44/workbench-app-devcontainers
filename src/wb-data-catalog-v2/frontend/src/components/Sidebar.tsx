import { Link, useLocation } from "react-router-dom";

const navLinkBase: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "10px 14px",
  borderRadius: "var(--wb-radius)",
  fontSize: 14,
  fontWeight: 500,
  color: "var(--wb-sidebar-text)",
  textDecoration: "none",
  transition: "background 0.12s",
};

function NavLink(props: { to: string; label: string; active: boolean; onClick?: () => void }) {
  return (
    <Link
      to={props.to}
      onClick={props.onClick}
      style={{
        ...navLinkBase,
        background: props.active ? "var(--wb-sidebar-active)" : "transparent",
      }}
      onMouseEnter={(e) => {
        if (!props.active) e.currentTarget.style.background = "var(--wb-sidebar-hover)";
      }}
      onMouseLeave={(e) => {
        if (!props.active) e.currentTarget.style.background = "transparent";
      }}
    >
      {props.label}
    </Link>
  );
}

export function Sidebar(props: {
  projectId: string;
  projectName?: string;
  onRefresh: () => void;
  onNavigate?: () => void;
  onHelpClick?: () => void;
}) {
  const location = useLocation();
  const path = location.pathname;

  return (
    <nav
      style={{
        width: "var(--wb-sidebar-width)",
        minWidth: "var(--wb-sidebar-width)",
        height: "100vh",
        background: "var(--wb-sidebar-bg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        position: "sticky",
        top: 0,
      }}
    >
      {/* Logo / brand */}
      <div style={{ padding: "20px 20px 12px" }}>
        <Link to="/" style={{ textDecoration: "none", color: "var(--wb-sidebar-text)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <img src="/logo.png" alt="Data Catalog" width={28} height={28} style={{ borderRadius: 6 }} />
            <div>
              <div style={{ fontWeight: 700, fontSize: 16, lineHeight: 1.2 }}>workbench</div>
              <div style={{ fontSize: 12, color: "var(--wb-sidebar-muted)", lineHeight: 1.2 }}>Data Catalog v2</div>
            </div>
          </div>
        </Link>
      </div>

      {/* Project badge */}
      <div style={{ padding: "0 20px 12px" }}>
        {props.projectId ? (
          <div
            style={{
              background: "var(--wb-sidebar-active)",
              borderRadius: "var(--wb-radius)",
              padding: "8px 12px",
              fontSize: 13,
              color: "var(--wb-sidebar-text)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>
              {props.projectName ? (
                <span>
                  <span style={{ fontWeight: 600 }}>{props.projectName}</span>
                  <br />
                  <span style={{ fontSize: 11, color: "var(--wb-sidebar-muted)" }}>{props.projectId}</span>
                </span>
              ) : props.projectId}
            </span>
          </div>
        ) : null}
      </div>

      {/* Global nav */}
      <div style={{ padding: "0 12px", display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
        <NavLink to="/" label="Data Catalog" active={path === "/" || path.startsWith("/table/")} onClick={props.onNavigate} />
        <NavLink to="/terminology" label="Terminology" active={path === "/terminology"} onClick={props.onNavigate} />
        <NavLink to="/cohorts" label="Cohort Builder" active={path === "/cohorts"} onClick={props.onNavigate} />
        <NavLink to="/chat" label="Data AMA Agent" active={path === "/chat"} onClick={props.onNavigate} />
        <NavLink to="/settings" label="Settings" active={path === "/settings"} onClick={props.onNavigate} />
      </div>

      {/* Bottom toolbar */}
      <div
        style={{
          padding: "10px 20px",
          borderTop: "1px solid rgba(255,255,255,0.1)",
          display: "flex",
          gap: 8,
        }}
      >
        <button
          type="button"
          onClick={props.onRefresh}
          style={{
            flex: 1,
            padding: "6px 0",
            background: "var(--wb-sidebar-hover)",
            border: "none",
            borderRadius: "var(--wb-radius)",
            color: "var(--wb-sidebar-text)",
            cursor: "pointer",
            fontSize: 13,
            fontFamily: "var(--wb-font)",
          }}
        >
          Refresh
        </button>
        {props.onHelpClick && (
          <button
            type="button"
            onClick={props.onHelpClick}
            style={{
              padding: "6px 0",
              background: "var(--wb-sidebar-hover)",
              border: "none",
              borderRadius: "var(--wb-radius)",
              color: "var(--wb-sidebar-text)",
              cursor: "pointer",
              fontSize: 13,
              fontFamily: "var(--wb-font)",
              width: 36,
            }}
            title="Help"
          >
            ?
          </button>
        )}
      </div>
    </nav>
  );
}
