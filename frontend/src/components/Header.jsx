import React from "react";
import {
  Cpu,
  Activity,
  ShieldCheck,
  Layers,
  BarChart3,
  Tv,
  FileCheck2,
  RefreshCw,
  UploadCloud,
  Compass
} from "lucide-react";

export default function Header({
  systemStatus,
  activeView,
  onViewChange,
  inputWorkflow,
  onWorkflowChange,
  onOpenReceipt,
  onOpenIngestionModal,
  isPresentationMode,
  onTogglePresentation,
  onForceLive,
  isForceLiveRunning,
  isLoading,
  selectedTileId
}) {
  const isCuda = systemStatus?.cuda_available ?? true;
  const devName = systemStatus?.device_name || "NVIDIA RTX 4060";
  const vramTotal = systemStatus?.vram_total_gb ?? 8.0;
  const vramAlloc = systemStatus?.vram_allocated_gb ?? 5.8;
  const checkpoints = systemStatus?.ensemble_checkpoints_found ?? 3;

  return (
    <header className="tactical-top-header">
      {/* Upper Command Strip */}
      <div className="header-upper-strip">
        {/* Brand & Subtitle */}
        <div className="header-brand-wrap">
          <div className="header-title-box">
            <span className="brand-title-text">GeoFUSE SentinelGuard</span>
            <span className="sih-tag">SIH 2026</span>
          </div>
          <div className="header-divider-v"></div>
          <span className="header-mission-subtext">
            Trust-Aware Satellite Super-Resolution | 10m → 4m Geospatial Reconstruction
          </span>
        </div>

        {/* Live Hardware Telemetry Pills */}
        <div className="header-telemetry-pills">
          <div className="telemetry-pill live-pill">
            <span className="pulse-dot-green"></span>
            <span className="pill-tag text-emerald-400">LIVE GPU</span>
          </div>

          <div className="header-divider-v"></div>

          <div className="telemetry-pill">
            <span className="pill-muted">GPU:</span>
            <span className="pill-val font-mono">{devName}</span>
          </div>

          <div className="header-divider-v"></div>

          <div className="telemetry-pill">
            <span className="pill-muted">VRAM:</span>
            <span className="pill-val font-mono">{vramAlloc} / {vramTotal} GB</span>
          </div>

          <div className="header-divider-v"></div>

          <div className="telemetry-pill">
            <span className="pill-muted">ENS:</span>
            <span className="pill-val-cyan font-mono">{checkpoints}/3 (EDSR, RCAN, SwinIR)</span>
          </div>
        </div>

        {/* Command Action Buttons */}
        <div className="header-actions-wrap">
          <button
            className={`action-btn presentation-btn ${isPresentationMode ? "active" : ""}`}
            onClick={onTogglePresentation}
            title="Toggle presentation fullscreen HUD"
            type="button"
          >
            <Tv size={14} />
            <span>{isPresentationMode ? "EXIT PRESENTATION" : "PRESENTATION"}</span>
          </button>

          <button
            className="action-btn receipt-pill-btn"
            onClick={onOpenReceipt}
            title="View authoritative cryptographic trust receipt"
            type="button"
          >
            <FileCheck2 size={14} />
            <span>TRUST RECEIPT</span>
          </button>

          {onForceLive && (
            <button
              className="action-btn recalibrate-btn"
              onClick={() => onForceLive(selectedTileId)}
              disabled={isForceLiveRunning || isLoading}
              title="Force live PyTorch GPU forward pass"
              type="button"
            >
              <RefreshCw size={13} className={(isForceLiveRunning || isLoading) ? "spin-animation" : ""} />
              <span>{(isForceLiveRunning || isLoading) ? "INFERRING..." : "LIVE PASS"}</span>
            </button>
          )}

          <div className="header-avatar" title="Commander Operator">
            <span>OP</span>
          </div>
        </div>
      </div>

      {/* Lower Mode & Coordinate Strip */}
      <div className="header-lower-strip">
        <div className="workflow-tab-pills">
          <button
            className={`workflow-tab ${inputWorkflow === "preloaded" && activeView === "workstation" ? "active" : ""}`}
            onClick={() => {
              onWorkflowChange("preloaded");
              onViewChange("workstation");
            }}
            type="button"
          >
            PRELOADED SCENES • 5×5 GRID
          </button>

          <button
            className={`workflow-tab ${inputWorkflow === "user_input" || activeView === "ingestion" ? "active" : ""}`}
            onClick={() => {
              if (onOpenIngestionModal) onOpenIngestionModal();
            }}
            type="button"
          >
            USER INPUT • UPLOAD GEOTIFF
          </button>

          <button
            className={`workflow-tab ${activeView === "benchmark" ? "active" : ""}`}
            onClick={() => onViewChange("benchmark")}
            type="button"
          >
            BENCHMARK LAB
          </button>

          <button
            className={`workflow-tab ${activeView === "building-analytics" ? "active" : ""}`}
            onClick={() => onViewChange("building-analytics")}
            type="button"
          >
            DOWNSTREAM CADASTRAL
          </button>
        </div>

        <div className="header-coords-box font-mono">
          <span className="coord-label">COORDINATE FRAME:</span>
          <span className="coord-val">WGS 84 / UTM ZONE 43N</span>
          <span className="coord-label ml-2">BAND:</span>
          <span className="coord-val text-cyan-400">B02, B03, B04, B08</span>
        </div>
      </div>
    </header>
  );
}
