import React from "react";
import { ShieldCheck, AlertTriangle, FileText, CheckCircle2, Clock } from "lucide-react";

export default function TrustPanel({ inferenceResult, onOpenReceipt, onForceLive, isForceLiveRunning }) {
  const metrics = inferenceResult?.evidence_metrics;
  const isTrusted = metrics?.is_trusted;
  const source = inferenceResult?.source || "unknown";
  const latency = inferenceResult?.latency_ms ?? 0;
  const tileId = inferenceResult?.tile_id ?? 0;

  const isCached = source === "demo_cache" || source === "api_cache" || source === "user_input_cache";

  return (
    <div className="card-panel">
      <div className="panel-header">
        <span className="panel-title">
          <ShieldCheck size={15} />
          Trust & Verification HUD
        </span>
        {inferenceResult && (
          <span className={`source-tag ${source}`}>
            {source === "demo_cache"
              ? "Demo Cache"
              : source === "api_cache"
              ? "API Cache"
              : source === "user_input"
              ? "User Input (Live)"
              : source === "user_input_cache"
              ? "User Input (Cached)"
              : source === "live_inference"
              ? "Live GPU Forward Pass"
              : "Live Inference"}
          </span>
        )}
      </div>

      <div className="panel-body">
        {metrics ? (
          <>
            {/* Top Composite Trust Score Banner */}
            <div className={`trust-score-banner ${isTrusted ? "" : "warning"}`}>
              <div className={`trust-score-number ${isTrusted ? "" : "warning"}`}>
                {metrics.trust_score_pct.toFixed(2)}%
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                Empirical Multi-Criteria Trust Score
              </div>
              <span className={`trust-status-badge ${isTrusted ? "" : "warning"}`}>
                {metrics.status_label}
              </span>
            </div>

            {/* Latency & Hardware Stats */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <Clock size={12} /> Execution Latency:
              </span>
              <span className="font-mono" style={{ color: "var(--text-primary)" }}>
                {source === "demo_cache" ? "Instant (Precomputed)" : `${latency.toFixed(1)} ms`}
              </span>
            </div>

            {/* Empirical Signals Breakdown */}
            <div style={{ marginBottom: "1rem" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "block", marginBottom: "0.4rem" }}>
                EMPIRICAL EVIDENCE METRICS
              </span>

              <div className="metric-row">
                <span className="metric-label">Ensemble Disagreement (σ):</span>
                <span className="metric-val font-mono">{metrics.mean_disagreement.toFixed(4)}</span>
              </div>

              <div className="metric-row">
                <span className="metric-label">Spectral ΔNDVI Mean:</span>
                <span className="metric-val font-mono">{metrics.delta_ndvi_mean.toFixed(4)}</span>
              </div>

              <div className="metric-row">
                <span className="metric-label">Spectral In-Tolerance (%):</span>
                <span className="metric-val font-mono">{metrics.pct_spectral_consistent.toFixed(1)}%</span>
              </div>

              <div className="metric-row">
                <span className="metric-label">Canny Edge IoU:</span>
                <span className="metric-val font-mono">{metrics.edge_iou.toFixed(4)}</span>
              </div>

              <div className="metric-row">
                <span className="metric-label">Gradient Correlation (r):</span>
                <span className="metric-val font-mono">{metrics.gradient_correlation.toFixed(4)}</span>
              </div>
            </div>

            {/* Plain-Language Warnings & Advisories */}
            {metrics.warnings && metrics.warnings.length > 0 && (
              <div className="warning-box">
                <div style={{ display: "flex", alignItems: "center", gap: "5px", fontWeight: "600", marginBottom: "4px", color: "var(--accent-amber)" }}>
                  <AlertTriangle size={13} />
                  Reliability Advisory Detected
                </div>
                <ul style={{ paddingLeft: "1.1rem", margin: 0 }}>
                  {metrics.warnings.map((w, idx) => (
                    <li key={idx} style={{ marginBottom: "4px" }}>
                      {w}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Inspect Trust Receipt Button */}
            <button
              onClick={onOpenReceipt}
              style={{
                width: "100%",
                marginTop: "1.25rem",
                padding: "0.6rem 1rem",
                backgroundColor: "var(--bg-primary)",
                border: "1px solid var(--accent-cyan)",
                color: "var(--accent-cyan)",
                borderRadius: "6px",
                fontSize: "0.8rem",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.5rem",
                transition: "all 0.15s ease",
              }}
            >
              <FileText size={14} />
              Inspect Authoritative Receipt
            </button>
          </>
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: "0.8rem", textAlign: "center", padding: "2rem 0" }}>
            Awaiting tile inference telemetry...
          </div>
        )}
      </div>
    </div>
  );
}
