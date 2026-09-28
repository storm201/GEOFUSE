import React from "react";
import {
  Compass,
  UploadCloud,
  Layers,
  ShieldCheck,
  BarChart3,
  Building2,
  Activity,
  Cpu,
  Radio,
  FileCheck
} from "lucide-react";

export default function Sidebar({
  activeView,
  onViewChange,
  onOpenIngestionModal,
  systemStatus,
  preloadedResult,
  customResult,
  inputWorkflow
}) {
  const activeResult = inputWorkflow === "preloaded" ? preloadedResult : customResult;
  const latency = activeResult?.latency_ms ?? 14.2;

  const navItems = [
    {
      id: "workstation",
      label: "Scene Browser",
      subtitle: "Workstation 5×5 Grid",
      icon: Layers,
      badge: "LIVE",
      action: () => onViewChange("workstation")
    },
    {
      id: "ingestion",
      label: "GeoTIFF Ingestion",
      subtitle: "L2A Multi-Band Upload",
      icon: UploadCloud,
      badge: customResult ? "READY" : null,
      action: () => {
        onViewChange("workstation");
        if (onOpenIngestionModal) onOpenIngestionModal();
      }
    },
    {
      id: "pipeline",
      label: "Spectral Pipeline",
      subtitle: "Flow & Tri-Ensemble",
      icon: Activity,
      badge: "TRT",
      action: () => onViewChange("pipeline")
    },
    {
      id: "trust-matrix",
      label: "Trust & Verification",
      subtitle: "4-Pillar Gate Matrix",
      icon: ShieldCheck,
      badge: "92.3%",
      action: () => onViewChange("trust-matrix")
    },
    {
      id: "benchmark",
      label: "Benchmark Lab",
      subtitle: "SOTA Model Comparison",
      icon: BarChart3,
      badge: "S2",
      action: () => onViewChange("benchmark")
    },
    {
      id: "building-analytics",
      label: "Building Analytics",
      subtitle: "Downstream Footprints",
      icon: Building2,
      badge: "10m→4m",
      action: () => onViewChange("building-analytics")
    }
  ];

  return (
    <aside className="tactical-sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand-box">
        <div className="sidebar-brand-content">
          <div className="sidebar-brand-logo">
            <Radio size={18} className="text-cyan-400 animate-pulse" />
          </div>
          <div className="sidebar-brand-text">
            <span className="brand-name">GeoFUSE</span>
            <span className="brand-tagline">SentinelGuard // 10M→4M</span>
          </div>
        </div>
        <span className="brand-pill">SIH-2026</span>
      </div>

      {/* Workspace Rack Status */}
      <div className="sidebar-rack-status">
        <span className="rack-label">WORKSPACE RACK</span>
        <span className="rack-val">
          <span className="status-pulse-dot"></span>
          SECURE_MODE
        </span>
      </div>

      {/* Navigation List */}
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              className={`sidebar-nav-item ${isActive ? "active" : ""}`}
              onClick={item.action}
              type="button"
            >
              <div className="nav-item-left">
                <span className="nav-icon-wrap">
                  <Icon size={18} />
                </span>
                <div className="nav-item-labels">
                  <span className="nav-item-title">{item.label}</span>
                  <span className="nav-item-sub">{item.subtitle}</span>
                </div>
              </div>
              {item.badge && (
                <span className={`nav-item-badge ${isActive ? "active" : ""}`}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Telemetry Footer */}
      <div className="sidebar-footer">
        <div className="footer-telemetry-row">
          <span className="telemetry-label">TELEMETRY LINK</span>
          <span className="telemetry-val-green">OPTICAL_99.98%</span>
        </div>
        <div className="telemetry-progress-bar">
          <div className="telemetry-progress-fill" style={{ width: "99.9%" }}></div>
        </div>
        <div className="footer-coords-row">
          <span>LAT: 28.6139° N</span>
          <span>LON: 77.2090° E</span>
        </div>
        <div className="footer-latency-row">
          <span className="latency-label">INFERENCE:</span>
          <span className="latency-val font-mono">{latency.toFixed(1)} ms/tile</span>
        </div>
      </div>
    </aside>
  );
}
