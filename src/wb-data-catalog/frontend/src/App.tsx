import { Navigate, Route, Routes } from "react-router-dom";
import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { SettingsPanel } from "./components/SettingsPanel";
import { useCatalog, useConfig } from "./hooks/useDatasets";
import CatalogHome from "./pages/CatalogPage";
import TablePage from "./pages/TablePage";

export default function App() {
  const { config, save: saveConfig, reload: reloadConfig } = useConfig();
  const [refreshKey, setRefreshKey] = useState(0);
  const [showSettings, setShowSettings] = useState(false);

  const configured = config?.configured ?? false;
  const { data, loading } = useCatalog(configured ? refreshKey : -1);

  const handleRefresh = () => {
    reloadConfig();
    setRefreshKey((k) => k + 1);
  };

  if (!configured) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
        <div style={{ maxWidth: 520, width: "100%" }}>
          <div style={{ textAlign: "center", marginBottom: 24 }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "var(--wb-primary)" }}>workbench</div>
            <div style={{ color: "var(--wb-muted)", fontSize: 14 }}>Data Catalog</div>
          </div>
          <SettingsPanel
            config={config}
            onSave={saveConfig}
            onSaved={handleRefresh}
          />
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar
        projectId={config?.data_project ?? ""}
        datasets={data?.datasets ?? []}
        loading={loading}
        onSettingsClick={() => setShowSettings((s) => !s)}
        onRefresh={handleRefresh}
      />

      <main
        style={{
          flex: 1,
          overflow: "auto",
          height: "100vh",
        }}
      >
        {showSettings ? (
          <div style={{ padding: 32, maxWidth: 640 }}>
            <SettingsPanel
              config={config}
              onSave={saveConfig}
              onSaved={() => {
                handleRefresh();
                setShowSettings(false);
              }}
            />
          </div>
        ) : (
          <Routes>
            <Route
              path="/"
              element={
                <CatalogHome
                  config={config}
                  datasets={data?.datasets ?? []}
                  loading={loading}
                />
              }
            />
            <Route path="/table/:project/:dataset/:table" element={<TablePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        )}
      </main>
    </div>
  );
}
