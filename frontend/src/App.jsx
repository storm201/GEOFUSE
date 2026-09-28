import React, { useState, useEffect, useRef } from "react";
import { Map, UploadCloud, RotateCcw, Zap, Layers } from "lucide-react";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import SceneNavigator from "./components/SceneNavigator";
import HeroViewer from "./components/HeroViewer";
import TrustPanel from "./components/TrustPanel";
import ReceiptModal from "./components/ReceiptModal";
import BenchmarkView from "./components/BenchmarkView";
import TrustMatrixView from "./components/TrustMatrixView";
import PipelineDiagnosticsView from "./components/PipelineDiagnosticsView";
import BuildingAnalyticsView from "./components/BuildingAnalyticsView";
import IngestionModal from "./components/IngestionModal";
import UserUpload from "./components/UserUpload";
import { getSystemStatus, getScenes, getSceneGrid, runTileInference } from "./services/api";

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [selectedSceneId, setSelectedSceneId] = useState("urban_core");
  const [gridData, setGridData] = useState(null);
  const [selectedTileId, setSelectedTileId] = useState(0);
  const [preloadedResult, setPreloadedResult] = useState(null);

  // Active View Navigation across the 7 Stitch Screens:
  // "workstation" | "trust-matrix" | "pipeline" | "benchmark" | "building-analytics"
  const [activeView, setActiveView] = useState("workstation");

  // Dual workflow: "preloaded" vs "user_input"
  const [inputWorkflow, setInputWorkflow] = useState("preloaded");
  const [customSceneInfo, setCustomSceneInfo] = useState(null);
  const [customTileId, setCustomTileId] = useState(0);
  const [customResult, setCustomResult] = useState(null);
  const [customLoading, setCustomLoading] = useState(false);
  const [customError, setCustomError] = useState(null);
  const [customIsForceLiveRunning, setCustomIsForceLiveRunning] = useState(false);

  const [loading, setLoading] = useState(false);
  const [isForceLiveRunning, setIsForceLiveRunning] = useState(false);
  const [error, setError] = useState(null);

  // Modals
  const [receiptModalOpen, setReceiptModalOpen] = useState(false);
  const [ingestionModalOpen, setIngestionModalOpen] = useState(false);
  const [isPresentationMode, setIsPresentationMode] = useState(false);

  // Request tokens and AbortControllers
  const tileAbortRef = useRef(null);
  const tileRequestTokenRef = useRef(0);
  const customTileAbortRef = useRef(null);
  const customTileRequestTokenRef = useRef(0);

  // 1. Initial Application Bootstrap
  useEffect(() => {
    let isMounted = true;

    // Load System Telemetry
    getSystemStatus()
      .then((status) => {
        if (isMounted) setSystemStatus(status);
      })
      .catch((err) => console.error("Could not fetch system status:", err));

    // Load Available Scenes
    getScenes()
      .then((sceneList) => {
        if (!isMounted) return;
        setScenes(sceneList.scenes || []);
        const defaultId = sceneList.default_scene_id || "urban_core";
        setSelectedSceneId(defaultId);
        loadGridAndInitialTile(defaultId);
      })
      .catch((err) => {
        if (isMounted) setError(`Backend connection failed: ${err.message}`);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Helper to load preloaded scene grid and fetch tile #0
  const loadGridAndInitialTile = async (sceneId) => {
    if (tileAbortRef.current) {
      tileAbortRef.current.abort();
    }
    const abortCtrl = new AbortController();
    tileAbortRef.current = abortCtrl;
    const currentToken = ++tileRequestTokenRef.current;

    try {
      setLoading(true);
      setError(null);
      const grid = await getSceneGrid(sceneId);
      if (currentToken !== tileRequestTokenRef.current) return;

      setGridData(grid);
      setSelectedTileId(0);

      const initialInference = await runTileInference(sceneId, 0, false, abortCtrl.signal);
      if (currentToken === tileRequestTokenRef.current) {
        setPreloadedResult(initialInference);
      }
    } catch (err) {
      if (err.name === "AbortError" || err.message?.includes("aborted")) {
        return;
      }
      if (currentToken === tileRequestTokenRef.current) {
        setError(`Failed to load scene grid: ${err.message}`);
      }
    } finally {
      if (currentToken === tileRequestTokenRef.current) {
        setLoading(false);
      }
    }
  };

  // Handle Scene Change in Preloaded mode
  const handleSceneChange = (newSceneId) => {
    setSelectedSceneId(newSceneId);
    loadGridAndInitialTile(newSceneId);
  };

  // Handle Preloaded Tile Selection with Stale Guard & AbortController
  const handleTileSelect = async (tileId) => {
    if (tileId === selectedTileId && preloadedResult && !error && !loading) {
      return;
    }

    if (tileAbortRef.current) {
      tileAbortRef.current.abort();
    }
    const abortCtrl = new AbortController();
    tileAbortRef.current = abortCtrl;
    const currentToken = ++tileRequestTokenRef.current;

    setSelectedTileId(tileId);
    setLoading(true);
    setError(null);

    try {
      const res = await runTileInference(selectedSceneId, tileId, false, abortCtrl.signal);
      if (currentToken === tileRequestTokenRef.current) {
        setPreloadedResult(res);
      }
    } catch (err) {
      if (err.name === "AbortError" || err.message?.includes("aborted")) {
        return;
      }
      if (currentToken === tileRequestTokenRef.current) {
        setError(`Inference failed on tile #${tileId}: ${err.message}`);
      }
    } finally {
      if (currentToken === tileRequestTokenRef.current) {
        setLoading(false);
      }
    }
  };

  // Handle Preloaded Live GPU Force Rerun
  const handlePreloadedForceLive = async (tileId) => {
    if (tileAbortRef.current) {
      tileAbortRef.current.abort();
    }
    const abortCtrl = new AbortController();
    tileAbortRef.current = abortCtrl;
    const currentToken = ++tileRequestTokenRef.current;

    setIsForceLiveRunning(true);
    setLoading(true);
    setError(null);

    try {
      const res = await runTileInference(selectedSceneId, tileId, true, abortCtrl.signal);
      if (currentToken === tileRequestTokenRef.current) {
        setPreloadedResult(res);
      }
    } catch (err) {
      if (err.name === "AbortError" || err.message?.includes("aborted")) {
        return;
      }
      if (currentToken === tileRequestTokenRef.current) {
        setError(`Live GPU forward pass failed on tile #${tileId}: ${err.message}`);
      }
    } finally {
      if (currentToken === tileRequestTokenRef.current) {
        setIsForceLiveRunning(false);
        setLoading(false);
      }
    }
  };

  // Handle Custom Scene Validation Completion
  const handleCustomSceneValidated = async (valInfo) => {
    if (customTileAbortRef.current) {
      customTileAbortRef.current.abort();
    }
    const abortCtrl = new AbortController();
    customTileAbortRef.current = abortCtrl;
    const currentToken = ++customTileRequestTokenRef.current;

    setInputWorkflow("user_input");
    setActiveView("workstation");
    setCustomSceneInfo(valInfo);
    setCustomTileId(0);
    setCustomLoading(true);
    setCustomError(null);
    setCustomResult(null);

    try {
      const initialCustomRes = await runTileInference(valInfo.scene_id, 0, false, abortCtrl.signal);
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomResult(initialCustomRes);
      }
    } catch (err) {
      if (err.name === "AbortError" || err.message?.includes("aborted")) {
        return;
      }
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomError(`Inference failed on uploaded scene tile #00: ${err.message}`);
      }
    } finally {
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomLoading(false);
      }
    }
  };

  // Handle Custom Tile Selection
  const handleCustomTileSelect = async (tileId) => {
    if (!customSceneInfo) return;
    if (tileId === customTileId && customResult && !customError && !customLoading) {
      return;
    }

    if (customTileAbortRef.current) {
      customTileAbortRef.current.abort();
    }
    const abortCtrl = new AbortController();
    customTileAbortRef.current = abortCtrl;
    const currentToken = ++customTileRequestTokenRef.current;

    setCustomTileId(tileId);
    setCustomLoading(true);
    setCustomError(null);

    try {
      const res = await runTileInference(customSceneInfo.scene_id, tileId, false, abortCtrl.signal);
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomResult(res);
      }
    } catch (err) {
      if (err.name === "AbortError" || err.message?.includes("aborted")) {
        return;
      }
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomError(`Custom inference failed on tile #${tileId}: ${err.message}`);
      }
    } finally {
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomLoading(false);
      }
    }
  };

  // Handle Custom Live GPU Force Rerun
  const handleCustomForceLive = async (tileId) => {
    if (!customSceneInfo) return;
    if (customTileAbortRef.current) {
      customTileAbortRef.current.abort();
    }
    const abortCtrl = new AbortController();
    customTileAbortRef.current = abortCtrl;
    const currentToken = ++customTileRequestTokenRef.current;

    setCustomIsForceLiveRunning(true);
    setCustomLoading(true);
    setCustomError(null);

    try {
      const res = await runTileInference(customSceneInfo.scene_id, tileId, true, abortCtrl.signal);
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomResult(res);
      }
    } catch (err) {
      if (err.name === "AbortError" || err.message?.includes("aborted")) {
        return;
      }
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomError(`Live GPU forward pass failed on custom tile #${tileId}: ${err.message}`);
      }
    } finally {
      if (currentToken === customTileRequestTokenRef.current) {
        setCustomIsForceLiveRunning(false);
        setCustomLoading(false);
      }
    }
  };

  // Reset custom upload state
  const handleResetCustomUpload = () => {
    if (customTileAbortRef.current) {
      customTileAbortRef.current.abort();
    }
    setCustomSceneInfo(null);
    setCustomResult(null);
    setCustomTileId(0);
    setCustomError(null);
    setCustomLoading(false);
    setCustomIsForceLiveRunning(false);
    setInputWorkflow("preloaded");
  };

  // Active result for the receipt modal & HUD
  const activeResult =
    inputWorkflow === "preloaded" ? preloadedResult : customResult;
  const activeReceiptId = activeResult?.receipt_id;
  const activeTileId = inputWorkflow === "preloaded" ? selectedTileId : customTileId;
  const activeSceneId = inputWorkflow === "preloaded" ? selectedSceneId : (customSceneInfo?.scene_id || "custom");

  return (
    <div className="stitch-app-shell">
      {/* Tactical Left Sidebar Navigation (Screens 1 to 7 selector) */}
      {!isPresentationMode && (
        <Sidebar
          activeView={activeView}
          onViewChange={setActiveView}
          onOpenIngestionModal={() => setIngestionModalOpen(true)}
          systemStatus={systemStatus}
          preloadedResult={preloadedResult}
          customResult={customResult}
          inputWorkflow={inputWorkflow}
        />
      )}

      {/* Main Layout Area */}
      <div className={`main-wrapper-layout ${isPresentationMode ? "fullscreen" : ""}`}>
        {/* Top Header Command Bar */}
        {!isPresentationMode && (
          <Header
            systemStatus={systemStatus}
            activeView={activeView}
            onViewChange={setActiveView}
            inputWorkflow={inputWorkflow}
            onWorkflowChange={setInputWorkflow}
            onOpenReceipt={() => setReceiptModalOpen(true)}
            onOpenIngestionModal={() => setIngestionModalOpen(true)}
            isPresentationMode={isPresentationMode}
            onTogglePresentation={() => setIsPresentationMode((prev) => !prev)}
            onForceLive={inputWorkflow === "preloaded" ? handlePreloadedForceLive : handleCustomForceLive}
            isForceLiveRunning={inputWorkflow === "preloaded" ? isForceLiveRunning : customIsForceLiveRunning}
            isLoading={inputWorkflow === "preloaded" ? (loading || isForceLiveRunning) : (customLoading || customIsForceLiveRunning)}
            selectedTileId={activeTileId}
          />
        )}

        {/* View Router */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          {/* SCREEN 1: Workstation */}
          {activeView === "workstation" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", flex: 1, padding: "0.5rem" }}>
              {/* Dual Workflow Switcher Strip */}
              {!isPresentationMode && (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    flexWrap: "wrap",
                    gap: "0.75rem",
                    padding: "0.25rem 0.5rem",
                  }}
                >
                  <div className="mode-switcher-bar">
                    <button
                      className={`mode-switch-btn ${inputWorkflow === "preloaded" ? "active" : ""}`}
                      onClick={() => setInputWorkflow("preloaded")}
                      type="button"
                    >
                      <Map size={15} />
                      <span>Preloaded Scenes (5×5 Grid)</span>
                    </button>
                    <button
                      className={`mode-switch-btn ${inputWorkflow === "user_input" ? "active" : ""}`}
                      onClick={() => setInputWorkflow("user_input")}
                      type="button"
                    >
                      <UploadCloud size={15} />
                      <span>User Input (Upload GeoTIFF)</span>
                      {customResult && <span className="source-tag user_input">Ready</span>}
                    </button>
                  </div>

                  {inputWorkflow === "user_input" && customSceneInfo && (
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <button
                        className="btn-secondary"
                        style={{ fontSize: "0.8rem", padding: "0.35rem 0.8rem" }}
                        onClick={handleResetCustomUpload}
                        type="button"
                      >
                        <RotateCcw size={14} />
                        Upload Another Scene
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Workflow A: Preloaded Sentinel-2 Scenes */}
              {inputWorkflow === "preloaded" && (
                <main className={`main-workspace ${isPresentationMode ? "presentation-active" : ""}`}>
                  {!isPresentationMode && (
                    <SceneNavigator
                      scenes={scenes}
                      selectedSceneId={selectedSceneId}
                      onSceneChange={handleSceneChange}
                      gridData={gridData}
                      selectedTileId={selectedTileId}
                      onTileSelect={handleTileSelect}
                      loading={loading}
                    />
                  )}

                  <HeroViewer
                    inferenceResult={preloadedResult}
                    loading={loading}
                    error={error}
                    onForceLive={handlePreloadedForceLive}
                    isForceLiveRunning={isForceLiveRunning}
                    isPresentationMode={isPresentationMode}
                    onTogglePresentation={() => setIsPresentationMode((prev) => !prev)}
                    onLoadDemoCache={() => handleTileSelect(0)}
                    sceneMetadata={scenes.find((s) => s.scene_id === selectedSceneId)}
                    tileInfo={gridData?.tiles?.find((t) => t.tile_id === selectedTileId)}
                  />

                  {!isPresentationMode && (
                    <TrustPanel
                      inferenceResult={preloadedResult}
                      onOpenReceipt={() => setReceiptModalOpen(true)}
                      onForceLive={handlePreloadedForceLive}
                      isForceLiveRunning={isForceLiveRunning}
                    />
                  )}
                </main>
              )}

              {/* Workflow B: User-Provided Input */}
              {inputWorkflow === "user_input" && (
                <>
                  {customSceneInfo ? (
                    <main className={`main-workspace ${isPresentationMode ? "presentation-active" : ""}`}>
                      {!isPresentationMode && (
                        <SceneNavigator
                          scenes={[]}
                          selectedSceneId={customSceneInfo.scene_id}
                          onSceneChange={() => {}}
                          gridData={customSceneInfo.grid}
                          selectedTileId={customTileId}
                          onTileSelect={handleCustomTileSelect}
                          loading={customLoading}
                        />
                      )}

                      <HeroViewer
                        inferenceResult={customResult}
                        loading={customLoading}
                        error={customError}
                        onForceLive={handleCustomForceLive}
                        isForceLiveRunning={customIsForceLiveRunning}
                        isPresentationMode={isPresentationMode}
                        onTogglePresentation={() => setIsPresentationMode((prev) => !prev)}
                        sceneMetadata={customSceneInfo}
                        tileInfo={customSceneInfo?.grid?.tiles?.find((t) => t.tile_id === customTileId)}
                      />

                      {!isPresentationMode && (
                        <TrustPanel
                          inferenceResult={customResult}
                          onOpenReceipt={() => setReceiptModalOpen(true)}
                          onForceLive={handleCustomForceLive}
                          isForceLiveRunning={customIsForceLiveRunning}
                        />
                      )}
                    </main>
                  ) : (
                    <UserUpload onSceneValidated={handleCustomSceneValidated} />
                  )}
                </>
              )}
            </div>
          )}

          {/* SCREEN 2: Trust & Verification Matrix */}
          {activeView === "trust-matrix" && (
            <TrustMatrixView
              inferenceResult={activeResult}
              onOpenReceipt={() => setReceiptModalOpen(true)}
              onForceLive={inputWorkflow === "preloaded" ? handlePreloadedForceLive : handleCustomForceLive}
              isForceLiveRunning={inputWorkflow === "preloaded" ? isForceLiveRunning : customIsForceLiveRunning}
              selectedSceneId={activeSceneId}
              selectedTileId={activeTileId}
            />
          )}

          {/* SCREEN 4: Benchmark Lab */}
          {activeView === "benchmark" && <BenchmarkView />}

          {/* SCREEN 5: Spectral Pipeline Diagnostics */}
          {activeView === "pipeline" && (
            <PipelineDiagnosticsView
              systemStatus={systemStatus}
              preloadedResult={activeResult}
              selectedSceneId={activeSceneId}
              selectedTileId={activeTileId}
            />
          )}

          {/* SCREEN 7: Downstream Building Analytics */}
          {activeView === "building-analytics" && (
            <BuildingAnalyticsView
              inferenceResult={activeResult}
              onForceLive={inputWorkflow === "preloaded" ? handlePreloadedForceLive : handleCustomForceLive}
              isForceLiveRunning={inputWorkflow === "preloaded" ? isForceLiveRunning : customIsForceLiveRunning}
              selectedSceneId={activeSceneId}
              selectedTileId={activeTileId}
              onOpenReceipt={() => setReceiptModalOpen(true)}
            />
          )}
        </div>
      </div>

      {/* SCREEN 3: Authoritative Cryptographic Trust Receipt Modal */}
      {receiptModalOpen && activeReceiptId && (
        <ReceiptModal
          receiptId={activeReceiptId}
          onClose={() => setReceiptModalOpen(false)}
        />
      )}

      {/* SCREEN 6: GeoTIFF Ingestion Modal */}
      <IngestionModal
        isOpen={ingestionModalOpen}
        onClose={() => setIngestionModalOpen(false)}
        onSceneValidated={handleCustomSceneValidated}
      />
    </div>
  );
}
