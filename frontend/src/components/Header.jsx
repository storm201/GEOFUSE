import React from "react";
import { Cpu, Activity, ShieldCheck, Layers, BarChart3 } from "lucide-react";

export default function Header({ systemStatus, currentMode, onModeChange }) {
  const isCuda = systemStatus?.cuda_available;
  const devName = systemStatus?.device_name || "Detecting...";
  const vramTotal = systemStatus?.vram_total_gb ? `${systemStatus.vram_total_gb} GB` : "N/A";
  const checkpoints = systemStatus?.ensemble_checkpoints_found ?? 0;

  return (
    <header className="top-bar">
      <div className="brand-section">
        <span className="brand-badge">SIH 2026</span>
        <div>
          <h1 className="brand-title">GeoFUSE SentinelGuard</h1>
          <p className="brand-subtitle">Trust-Aware Satellite Super-Resolution (10m → 4m GSD)</p>
        </div>
      </div>

      <div className="telemetry-strip">
        <div className="telemetry-item" title="Hardware Compute Device">
          <Cpu size={14} className={isCuda ? "text-cyan-400" : "text-slate-400"} />
          <span className="font-mono">{devName}</span>
          <span className={`telemetry-dot ${isCuda ? "" : "warning"}`} />
        </div>

        {isCuda && (
          <div className="telemetry-item" title="Dedicated GPU VRAM">
            <Activity size={14} />
            <span className="font-mono">VRAM: {vramTotal}</span>
          </div>
        )}

        <div className="telemetry-item" title="Loaded Ensemble Models">
          <ShieldCheck size={14} />
          <span className="font-mono">Ensemble: {checkpoints}/3 Models</span>
        </div>
      </div>

      <div className="mode-switcher">
        <button
          className={`mode-btn ${currentMode === "real" ? "active" : ""}`}
          onClick={() => onModeChange("real")}
        >
          <Layers size={14} style={{ display: "inline", marginRight: "4px", verticalAlign: "-2px" }} />
          Live Real SR
        </button>
        <button
          className={`mode-btn ${currentMode === "benchmark" ? "active" : ""}`}
          onClick={() => onModeChange("benchmark")}
        >
          <BarChart3 size={14} style={{ display: "inline", marginRight: "4px", verticalAlign: "-2px" }} />
          Benchmark Lab
        </button>
      </div>
    </header>
  );
}
