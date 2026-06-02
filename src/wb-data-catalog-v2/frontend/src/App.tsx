import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { HelpPanel } from "./components/HelpSystem";
import { LoadingSplash } from "./components/LoadingSplash";
import { NotificationProvider } from "./components/Notifications";
import { Sidebar } from "./components/Sidebar";
import { SettingsPanel } from "./components/SettingsPanel";
import { useConfig } from "./hooks/useDatasets";
import { useProgressiveCatalog } from "./hooks/useProgressiveCatalog";
import CatalogHome from "./pages/CatalogPage";
import ChatPage from "./pages/ChatPage";
import CohortsPage from "./pages/CohortsPage";
import SettingsPage from "./pages/SettingsPage";
import TablePage from "./pages/TablePage";
import TerminologyPage from "./pages/TerminologyPage";

export default function App() {
  const { config, save: saveConfig, reload: reloadConfig } = useConfig();
  const [refreshKey, setRefreshKey] = useState(0);
  const [helpOpen, setHelpOpen] = useState(false);
  const [splashHold, setSplashHold] = useState(false);
  const location = useLocation();

  const configured = config?.configured ?? false;
  const dataProject = config?.data_project ?? "";

  const catalog = useProgressiveCatalog(
    configured ? dataProject : "",
    refreshKey,
  );

  const handleSaving = () => setSplashHold(true);

  const handleRefresh = () => {
    setSplashHold(true);
    reloadConfig();
    setRefreshKey((k) => k + 1);
  };

  const wasLoading = useRef(false);
  const catalogReady = !catalog.loadingDatasets && catalog.datasets.length > 0;
  useEffect(() => {
    if (catalog.loadingDatasets) wasLoading.current = true;
    if (splashHold && wasLoading.current && catalogReady) {
      wasLoading.current = false;
      setSplashHold(false);
    }
  }, [splashHold, catalog.loadingDatasets, catalogReady]);

  const isInitialLoad = splashHold || (catalog.loadingDatasets && catalog.datasets.length === 0);

  const helpPage = location.pathname.startsWith("/table/") ? "table"
    : location.pathname === "/terminology" ? "terminology"
    : location.pathname === "/cohorts" ? "cohorts"
    : location.pathname === "/chat" ? "chat"
    : location.pathname === "/settings" ? "catalog"
    : "catalog";

  if (!configured) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
        <div style={{ maxWidth: 600, width: "100%" }}>
          <div style={{ textAlign: "center", marginBottom: 24 }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "var(--wb-primary)" }}>workbench</div>
            <div style={{ color: "var(--wb-muted)", fontSize: 14 }}>Data Catalog v2</div>
          </div>
          <SettingsPanel
            config={config}
            onSave={saveConfig}
            onSaving={handleSaving}
            onSaved={handleRefresh}
          />
        </div>
      </div>
    );
  }

  return (
    <NotificationProvider>
      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        {/* Feedback banner */}
        <div style={{
          background: "#1a5c5e",
          color: "#fff",
          fontSize: 13,
          fontWeight: 500,
          textAlign: "center",
          padding: "6px 16px",
          flexShrink: 0,
        }}>
          This is an early preview.{" "}
          <a
            href="https://forms.gle/6ttFsVRzUR1jse4C6"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#a5d6a7", fontWeight: 700, textDecoration: "underline" }}
          >
            Share feedback or report bugs
          </a>
        </div>

        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <LoadingSplash
          visible={isInitialLoad}
          projectName={config?.data_project_name || dataProject}
          status="Loading datasets..."
        />

        <Sidebar
          projectId={dataProject}
          projectName={config?.data_project_name || ""}
          onRefresh={handleRefresh}
          onHelpClick={() => setHelpOpen(true)}
        />

        <HelpPanel page={helpPage} open={helpOpen} onClose={() => setHelpOpen(false)} />

        <main
          style={{
            flex: 1,
            overflow: "auto",
          }}
        >
          <ErrorBoundary>
            <Routes>
              <Route
                path="/"
                element={
                  <CatalogHome
                    config={config}
                    catalog={catalog}
                    onRefresh={handleRefresh}
                  />
                }
              />
              <Route path="/table/:project/:dataset/:table" element={<TablePage />} />
              <Route path="/terminology" element={<TerminologyPage />} />
              <Route path="/cohorts" element={<CohortsPage />} />
              <Route path="/chat" element={<ChatPage dataProject={dataProject} />} />
              <Route
                path="/settings"
                element={
                  <SettingsPage
                    config={config}
                    onSave={saveConfig}
                    onSaving={handleSaving}
                    onSaved={handleRefresh}
                  />
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
        </div>
      </div>
    </NotificationProvider>
  );
}
