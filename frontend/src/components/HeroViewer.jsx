import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Eye,
  Layers,
  Compass,
  ShieldAlert,
  Building2,
  Trees,
  Columns,
  SplitSquareVertical,
  Zap,
  MapPin,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Move,
  Maximize2,
  Minimize2,
  AlertTriangle,
  RefreshCw,
  Database,
} from "lucide-react";
import TileInsight from "./TileInsight";

export default function HeroViewer({
  inferenceResult,
  loading,
  error,
  onForceLive,
  isForceLiveRunning = false,
  isPresentationMode = false,
  onTogglePresentation,
  onLoadDemoCache,
  sceneMetadata = null,
  tileInfo = null,
}) {
  const [activeTab, setActiveTab] = useState("rgb");
  const [viewMode, setViewMode] = useState("side-by-side"); // "side-by-side" | "curtain"
  const [curtainPos, setCurtainPos] = useState(50); // percentage 0..100

  // Synchronized Spatial Viewport Transform State
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);

  const isDraggingCurtainRef = useRef(false);
  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0 });
  const curtainCanvasRef = useRef(null);

  const assets = inferenceResult?.assets;
  const tileId = inferenceResult?.tile_id ?? 0;
  const sceneId = inferenceResult?.scene_id ?? "urban_core";
  const source = inferenceResult?.source || "demo_cache";
  const trustScore = inferenceResult?.evidence_metrics?.trust_score_pct;

  // Reset spatial zoom and pan when tile or scene changes
  useEffect(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, [tileId, sceneId]);

  // --- Presentation Hotkeys (1: RGB, 2: CIR, 3: NDVI, Space: Curtain Toggle, F: Presentation Mode) ---
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore if user is currently interacting with an input/textarea/select
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        document.activeElement?.isContentEditable
      ) {
        return;
      }
      // Ignore if system/browser modifier keys are pressed
      if (e.ctrlKey || e.altKey || e.metaKey) {
        return;
      }

      if (e.key === "1") {
        setActiveTab("rgb");
      } else if (e.key === "2") {
        setActiveTab("cir");
      } else if (e.key === "3") {
        setActiveTab("ndvi");
      } else if (e.code === "Space") {
        e.preventDefault();
        setViewMode((prev) => (prev === "curtain" ? "side-by-side" : "curtain"));
      } else if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        if (onTogglePresentation) {
          onTogglePresentation();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onTogglePresentation]);

  // Resolve left and right images based on active diagnostic tab
  let leftImage = null;
  let rightImage = null;
  let leftLabel = "10m Sentinel-2 Original";
  let rightLabel = "4m GeoFUSE Super-Resolution";

  if (assets) {
    if (activeTab === "rgb") {
      leftImage = assets.rgb_input;
      rightImage = assets.rgb_sr;
      leftLabel = "10m Input (Natural RGB: 4-3-2)";
      rightLabel = "4m GeoFUSE SR (Natural RGB: 4-3-2)";
    } else if (activeTab === "cir") {
      leftImage = assets.cir_input;
      rightImage = assets.cir_sr;
      leftLabel = "10m False-Color CIR (8-4-3)";
      rightLabel = "4m False-Color CIR (8-4-3)";
    } else if (activeTab === "ndvi") {
      leftImage = assets.ndvi_input;
      rightImage = assets.ndvi_sr;
      leftLabel = "10m NDVI Vegetation Index";
      rightLabel = "4m GeoFUSE NDVI (Calibrated)";
    } else if (activeTab === "uncertainty") {
      leftImage = assets.rgb_sr;
      rightImage = assets.uncertainty;
      leftLabel = "4m GeoFUSE Reconstruction";
      rightLabel = "Epistemic Uncertainty (Ensemble σ)";
    } else if (activeTab === "trust") {
      leftImage = assets.rgb_sr;
      rightImage = assets.trust_map;
      leftLabel = "4m GeoFUSE Reconstruction";
      rightLabel = "Composite Trust / Risk Map";
    } else if (activeTab === "buildings") {
      leftImage = assets.rgb_sr;
      rightImage = assets.building_mask || assets.rgb_sr;
      leftLabel = "4m GeoFUSE Reconstruction";
      rightLabel = assets.building_mask
        ? "Building Footprints (Morphological Mask)"
        : "Building Footprints (No Detections in Tile)";
    }
  }

  // --- Synchronized Zoom Controls ---
  const handleZoomIn = () => {
    setZoom((z) => Math.min(8.0, Number((z * 1.3).toFixed(2))));
  };

  const handleZoomOut = () => {
    setZoom((z) => {
      const next = Math.max(1.0, Number((z / 1.3).toFixed(2)));
      if (next <= 1.0) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleResetTransform = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((prev) => {
      const next = Math.max(1.0, Math.min(8.0, Number((prev * factor).toFixed(2))));
      if (next <= 1.0) {
        setPan({ x: 0, y: 0 });
      }
      return next;
    });
  };

  // --- Synchronized Pan Handling for Side-by-Side Viewport ---
  const handleSidePanStart = (e) => {
    if (e.button === 0 || e.button === 1) {
      isPanningRef.current = true;
      setIsPanning(true);
      panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const handleSidePanMove = (e) => {
    if (!isPanningRef.current) return;
    setPan({
      x: e.clientX - panStartRef.current.x,
      y: e.clientY - panStartRef.current.y,
    });
  };

  const handleSidePanEnd = () => {
    if (isPanningRef.current) {
      isPanningRef.current = false;
      setIsPanning(false);
    }
  };

  // --- Curtain Slider Pointer Interactions ---
  const updateCurtainFromPointer = useCallback((clientX) => {
    if (!curtainCanvasRef.current) return;
    const rect = curtainCanvasRef.current.getBoundingClientRect();
    const offsetX = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (offsetX / rect.width) * 100));
    setCurtainPos(pct);
  }, []);

  const handleCurtainPointerDown = (e) => {
    if (!curtainCanvasRef.current) return;
    const rect = curtainCanvasRef.current.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const offsetX = clientX - rect.left;
    const pct = (offsetX / rect.width) * 100;

    // Check if click is within +/- 6% of divider or handle
    if (Math.abs(pct - curtainPos) <= 6 || zoom === 1.0) {
      isDraggingCurtainRef.current = true;
      updateCurtainFromPointer(clientX);
    } else {
      // Pan canvas when zoomed in and clicking away from divider
      isPanningRef.current = true;
      setIsPanning(true);
      panStartRef.current = {
        x: clientX - pan.x,
        y: (e.clientY || (e.touches && e.touches[0].clientY)) - pan.y,
      };
    }
  };

  const handleCurtainPointerMove = (e) => {
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);

    if (isDraggingCurtainRef.current) {
      updateCurtainFromPointer(clientX);
    } else if (isPanningRef.current) {
      setPan({
        x: clientX - panStartRef.current.x,
        y: clientY - panStartRef.current.y,
      });
    }
  };

  const handleCurtainPointerUp = () => {
    isDraggingCurtainRef.current = false;
    isPanningRef.current = false;
    setIsPanning(false);
  };

  // Shared geometric CSS transform applied synchronously to all layers
  const sharedTransformStyle = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    transformOrigin: "center center",
    transition: isPanning ? "none" : "transform 0.08s ease-out",
  };

  return (
    <div className={`hero-canvas-container ${isPresentationMode ? "presentation-mode" : ""}`}>
      {/* Presentation Mode Topbar Overlay */}
      {isPresentationMode && (
        <div className="presentation-topbar">
          <div className="presentation-meta">
            <span className="presentation-title">GeoFUSE SentinelGuard</span>
            <span className="presentation-pill">
              <MapPin size={11} /> Tile #{tileId.toString().padStart(2, "0")} · 1280m × 1280m
            </span>
            {trustScore !== undefined && (
              <span className="presentation-pill trust">
                Trust Score: {trustScore.toFixed(1)}%
              </span>
            )}
            <span className={`source-tag ${source}`}>{source.replace("_", " ").toUpperCase()}</span>
          </div>

          <div className="presentation-hotkeys">
            <span className="hotkey-pill"><b>1</b> RGB</span>
            <span className="hotkey-pill"><b>2</b> CIR</span>
            <span className="hotkey-pill"><b>3</b> NDVI</span>
            <span className="hotkey-pill"><b>Space</b> Curtain</span>
            <span className="hotkey-pill"><b>Scroll</b> Zoom</span>
          </div>

          <button
            className="presentation-exit-btn"
            onClick={onTogglePresentation}
            title="Exit Presentation Mode (Press F)"
          >
            <Minimize2 size={13} />
            <span>Exit (F)</span>
          </button>
        </div>
      )}

      {/* Multi-Band and Diagnostic Tabs */}
      <div className="view-tabs">
        <button
          className={`tab-btn ${activeTab === "rgb" ? "active" : ""}`}
          onClick={() => setActiveTab("rgb")}
          title="Shortcut: Press 1"
        >
          <Eye size={14} /> Natural RGB
        </button>

        <button
          className={`tab-btn ${activeTab === "cir" ? "active" : ""}`}
          onClick={() => setActiveTab("cir")}
          title="Shortcut: Press 2"
        >
          <Layers size={14} /> Color Infrared (CIR)
        </button>

        <button
          className={`tab-btn ${activeTab === "ndvi" ? "active" : ""}`}
          onClick={() => setActiveTab("ndvi")}
          title="Shortcut: Press 3"
        >
          <Trees size={14} /> NDVI Vegetation
        </button>

        <button
          className={`tab-btn ${activeTab === "uncertainty" ? "active" : ""}`}
          onClick={() => setActiveTab("uncertainty")}
        >
          <Compass size={14} /> Uncertainty (σ)
        </button>

        <button
          className={`tab-btn ${activeTab === "trust" ? "active" : ""}`}
          onClick={() => setActiveTab("trust")}
        >
          <ShieldAlert size={14} /> Trust / Risk Map
        </button>

        <button
          className={`tab-btn ${activeTab === "buildings" ? "active" : ""}`}
          onClick={() => setActiveTab("buildings")}
        >
          <Building2 size={14} /> Building Footprints
        </button>
      </div>

      {/* Viewer Control & Live Execution Strip */}
      <div className="viewer-toolbar">
        {/* View Mode: Side-by-Side vs Curtain Slider */}
        <div className="viewer-mode-toggle">
          <button
            className={`viewer-mode-btn ${viewMode === "side-by-side" ? "active" : ""}`}
            onClick={() => setViewMode("side-by-side")}
            title="Symmetrical Side-by-Side Viewport (Space to toggle)"
          >
            <Columns size={13} /> Side-by-Side
          </button>
          <button
            className={`viewer-mode-btn ${viewMode === "curtain" ? "active" : ""}`}
            onClick={() => setViewMode("curtain")}
            title="Shared Viewport Curtain Slider (Space to toggle)"
          >
            <SplitSquareVertical size={13} /> Curtain Slider
          </button>
        </div>

        {/* Synchronized Zoom & Pan Controls */}
        <div className="viewer-zoom-controls">
          <button
            className="zoom-btn"
            onClick={handleZoomOut}
            disabled={zoom <= 1.0}
            title="Zoom Out (or scroll down)"
          >
            <ZoomOut size={13} />
          </button>
          <span className="zoom-level-badge" title="Active Zoom Level">
            {Math.round(zoom * 100)}%
          </span>
          <button
            className="zoom-btn"
            onClick={handleZoomIn}
            disabled={zoom >= 8.0}
            title="Zoom In (or scroll up)"
          >
            <ZoomIn size={13} />
          </button>
          <button
            className="zoom-btn"
            onClick={handleResetTransform}
            disabled={zoom === 1.0 && pan.x === 0 && pan.y === 0}
            title="Reset Zoom & Pan (100%)"
          >
            <RotateCcw size={12} />
          </button>
        </div>

        {/* Spatial Footprint Badge */}
        <div className="spatial-footprint-badge">
          <MapPin size={12} className="icon-cyan" />
          <span>Tile #{tileId.toString().padStart(2, "0")} · 1280m × 1280m Extent</span>
          {zoom > 1.0 && (
            <span
              style={{
                color: "var(--accent-cyan)",
                marginLeft: "4px",
                display: "inline-flex",
                alignItems: "center",
                gap: "2px",
              }}
            >
              <Move size={10} /> Drag to Pan
            </span>
          )}
        </div>

        {/* Presentation Mode Toggle Button */}
        {onTogglePresentation && (
          <button
            className={`viewer-mode-btn ${isPresentationMode ? "active" : ""}`}
            onClick={onTogglePresentation}
            title="Toggle Presentation Mode (Shortcut: F)"
          >
            {isPresentationMode ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
            <span>{isPresentationMode ? "Normal (F)" : "Present (F)"}</span>
          </button>
        )}

        {/* Force Live Inference Button */}
        {onForceLive && (
          <button
            className="btn-live-gpu"
            onClick={() => onForceLive(tileId)}
            disabled={loading || isForceLiveRunning}
            title="Bypass all caches and execute fresh forward pass on GPU"
          >
            {isForceLiveRunning ? (
              <>
                <div
                  className="spinner"
                  style={{ width: "12px", height: "12px", borderWidth: "2px" }}
                />
                <span>Running GPU Forward Pass...</span>
              </>
            ) : (
              <>
                <Zap size={13} />
                <span>Run Using GPU (Force Live)</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Main Shared Spatial Viewport */}
      <div
        className="image-viewport"
        onWheel={handleWheel}
        onDoubleClick={handleResetTransform}
      >
        {loading ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "1rem",
              color: "var(--text-secondary)",
            }}
          >
            <div className="spinner" />
            <div style={{ textAlign: "center" }}>
              <div style={{ fontWeight: "600", color: "var(--text-primary)", fontSize: "0.9rem", marginBottom: "0.25rem" }}>
                PROCESSING TILE #{tileId.toString().padStart(2, "0")}
              </div>
              <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)", letterSpacing: "0.01em" }}>
                Executing direct 10m → 4m learned reconstruction & empirical trust checks...
              </span>
            </div>
          </div>
        ) : error ? (
          /* Graceful Failure & Recovery State */
          <div className="error-recovery-card">
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--accent-red)", marginBottom: "0.5rem" }}>
              <AlertTriangle size={20} />
              <span style={{ fontWeight: "600", fontSize: "0.95rem" }}>Inference Execution Notice</span>
            </div>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem", lineHeight: "1.4" }}>
              {error}
            </p>
            <div style={{ display: "flex", gap: "0.6rem", justifyContent: "center", flexWrap: "wrap" }}>
              {onForceLive && (
                <button
                  className="btn-secondary"
                  style={{ fontSize: "0.8rem", padding: "0.35rem 0.8rem" }}
                  onClick={() => onForceLive(tileId)}
                >
                  <RefreshCw size={13} />
                  Retry Live Inference
                </button>
              )}
              {onLoadDemoCache && !sceneId.startsWith("custom_") && (
                <button
                  className="btn-primary"
                  style={{ fontSize: "0.8rem", padding: "0.35rem 0.8rem" }}
                  onClick={onLoadDemoCache}
                >
                  <Database size={13} />
                  Load Offline Demo Cache
                </button>
              )}
            </div>
            {typeof sceneId === "string" && sceneId.startsWith("custom_") && (
              <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "0.75rem" }}>
                Scientific integrity rule: Custom uploaded scenes cannot be substituted with precomputed demo data.
              </p>
            )}
          </div>
        ) : assets ? (
          viewMode === "side-by-side" ? (
            /* Mode 1: Symmetrical Equal-Scale Side-by-Side Viewport with Synchronized Pan/Zoom */
            <div
              className={`comparison-side-by-side ${zoom > 1.0 ? (isPanning ? "panning" : "grabbable") : ""}`}
              onPointerDown={handleSidePanStart}
              onPointerMove={handleSidePanMove}
              onPointerUp={handleSidePanEnd}
              onPointerLeave={handleSidePanEnd}
            >
              {/* Left Panel: 10m Input Layer */}
              <div className="viewport-frame">
                <span className="image-badge">{leftLabel}</span>
                <div className="normalized-spatial-canvas">
                  <div className="spatial-transform-layer" style={sharedTransformStyle}>
                    {leftImage && (
                      <img
                        src={leftImage}
                        alt={leftLabel}
                        className="raster-layer layer-10m"
                        draggable={false}
                      />
                    )}
                  </div>
                </div>
              </div>

              {/* Right Panel: 4m Super-Resolution Layer */}
              <div className="viewport-frame">
                <span className="image-badge sr">{rightLabel}</span>
                <div className="normalized-spatial-canvas">
                  <div className="spatial-transform-layer" style={sharedTransformStyle}>
                    {rightImage && (
                      <img
                        src={rightImage}
                        alt={rightLabel}
                        className="raster-layer layer-4m"
                        draggable={false}
                      />
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Mode 2: Interactive Shared Spatial Curtain Slider with Synchronized Pan/Zoom */
            <div
              className="curtain-viewport-container"
              onPointerMove={handleCurtainPointerMove}
              onPointerUp={handleCurtainPointerUp}
              onPointerLeave={handleCurtainPointerUp}
            >
              <div
                ref={curtainCanvasRef}
                className={`curtain-canvas ${zoom > 1.0 ? (isPanning ? "panning" : "grabbable") : ""}`}
                onPointerDown={handleCurtainPointerDown}
              >
                {/* Unified Transformed Canvas Layer */}
                <div className="spatial-transform-layer curtain-shared-layer" style={sharedTransformStyle}>
                  {/* Left/Bottom Layer: 10m Input */}
                  {leftImage && (
                    <img
                      src={leftImage}
                      alt={leftLabel}
                      className="curtain-layer layer-10m"
                      draggable={false}
                    />
                  )}

                  {/* Right/Top Layer: 4m SR (Clipped by slider position) */}
                  {rightImage && (
                    <img
                      src={rightImage}
                      alt={rightLabel}
                      className="curtain-layer layer-4m"
                      draggable={false}
                      style={{
                        clipPath: `inset(0 0 0 ${curtainPos}%)`,
                        WebkitClipPath: `inset(0 0 0 ${curtainPos}%)`,
                      }}
                    />
                  )}
                </div>

                {/* Vertical Divider Line with Glowing Interactive Handle */}
                <div
                  className="curtain-divider"
                  style={{ left: `${curtainPos}%` }}
                >
                  <div className="curtain-handle" title="Drag to adjust curtain comparison">
                    ↔
                  </div>
                </div>

                {/* Labels indicating active layers */}
                <span className="image-badge">
                  {leftLabel} (10m)
                </span>
                <span className="image-badge sr top-right">
                  {rightLabel} (4m)
                </span>
              </div>
            </div>
          )
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
            Select a tile from the left navigator to begin inspection.
          </div>
        )}
      </div>

      {/* Grounded Educational & Explainability Layer */}
      {inferenceResult && (
        <TileInsight
          inferenceResult={inferenceResult}
          activeTab={activeTab}
          sceneMetadata={sceneMetadata}
          tileInfo={tileInfo}
        />
      )}
    </div>
  );
}
