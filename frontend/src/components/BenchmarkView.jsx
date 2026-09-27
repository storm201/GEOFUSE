import React, { useState, useEffect } from "react";
import { BarChart3, AlertCircle, Award, Database, CheckCircle2 } from "lucide-react";
import { getBenchmarkSummary } from "../services/api";

export default function BenchmarkView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    getBenchmarkSummary()
      .then((res) => {
        if (isMounted) setData(res);
      })
      .catch((err) => console.error("Failed to load benchmark:", err))
      .finally(() => {
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div style={{ maxWidth: "1100px", margin: "1.5rem auto", padding: "0 1.5rem" }}>
      {/* Banner */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(16, 185, 129, 0.05))",
          border: "1px solid var(--border-accent)",
          borderRadius: "8px",
          padding: "1.25rem 1.5rem",
          marginBottom: "1.5rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.25rem" }}>
            <span
              style={{
                backgroundColor: "var(--accent-cyan)",
                color: "#041019",
                fontSize: "0.7rem",
                fontWeight: "700",
                padding: "2px 6px",
                borderRadius: "3px",
              }}
            >
              CONTROLLED EVALUATION
            </span>
            <h2 style={{ fontSize: "1.25rem", fontWeight: "700", color: "#ffffff" }}>
              Synthetic Degrade-and-Recover Benchmark Lab
            </h2>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            {data?.description || "Controlled evaluation on contiguous Southeast Quadrant hold-out split."}
          </p>
        </div>
        <BarChart3 size={32} style={{ color: "var(--accent-cyan)", opacity: 0.8 }} />
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: "4rem 0" }}>
          <div className="spinner" style={{ margin: "0 auto 1rem" }} />
          <p style={{ color: "var(--text-muted)" }}>Loading verified benchmark results...</p>
        </div>
      ) : data ? (
        <div style={{ display: "grid", gap: "1.5rem" }}>
          {/* Quantitative Comparison Table */}
          <div className="card-panel">
            <div className="panel-header">
              <span className="panel-title">
                <Award size={15} />
                Quantitative Model Comparison (Hold-Out Evaluation)
              </span>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)", backgroundColor: "var(--bg-primary)" }}>
                    <th style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>Architecture / Method</th>
                    <th style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>Parameters</th>
                    <th style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>Val Loss</th>
                    <th style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>Val PSNR</th>
                    <th style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>Val SSIM</th>
                    <th style={{ padding: "0.75rem 1rem", color: "var(--text-muted)" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.baseline_methods?.map((m, idx) => (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: "1px solid var(--border-subtle)",
                        backgroundColor: m.is_ours ? "rgba(6, 182, 212, 0.04)" : "transparent",
                      }}
                    >
                      <td style={{ padding: "0.85rem 1rem", fontWeight: m.is_ours ? "600" : "400", color: m.is_ours ? "var(--accent-cyan)" : "var(--text-primary)" }}>
                        {m.name}
                      </td>
                      <td className="font-mono" style={{ padding: "0.85rem 1rem", color: "var(--text-secondary)" }}>
                        {m.parameters > 0 ? `${(m.parameters / 1e6).toFixed(2)}M` : "0 (Interpolation)"}
                      </td>
                      <td className="font-mono" style={{ padding: "0.85rem 1rem", color: "var(--text-secondary)" }}>
                        {m.val_loss?.toFixed(5) || "N/A"}
                      </td>
                      <td className="font-mono" style={{ padding: "0.85rem 1rem", fontWeight: "600", color: m.is_ours ? "var(--accent-green)" : "var(--text-primary)" }}>
                        {m.val_psnr_db ? `${m.val_psnr_db.toFixed(2)} dB` : "N/A"}
                        {m.delta_psnr_db && (
                          <span style={{ fontSize: "0.75rem", color: "var(--accent-green)", marginLeft: "4px" }}>
                            (+{m.delta_psnr_db.toFixed(2)} dB)
                          </span>
                        )}
                      </td>
                      <td className="font-mono" style={{ padding: "0.85rem 1rem", fontWeight: "600", color: m.is_ours ? "var(--accent-green)" : "var(--text-primary)" }}>
                        {m.val_ssim ? m.val_ssim.toFixed(4) : "N/A"}
                        {m.delta_ssim && (
                          <span style={{ fontSize: "0.75rem", color: "var(--accent-green)", marginLeft: "4px" }}>
                            (+{m.delta_ssim.toFixed(4)})
                          </span>
                        )}
                      </td>
                      <td style={{ padding: "0.85rem 1rem" }}>
                        {m.is_ours ? (
                          <span style={{ fontSize: "0.75rem", backgroundColor: "var(--accent-cyan-dim)", color: "var(--accent-cyan)", padding: "2px 6px", borderRadius: "3px" }}>
                            GeoFUSE (Ours)
                          </span>
                        ) : (
                          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Baseline</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dataset & Spatial Split Card */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
            <div className="card-panel">
              <div className="panel-header">
                <span className="panel-title">
                  <Database size={15} />
                  Geographic Hold-Out Strategy
                </span>
              </div>
              <div className="panel-body">
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.75rem" }}>
                  Random patch splitting causes spatial autocorrelation leakage in Earth Observation imagery.
                  GeoFUSE SentinelGuard enforces an independent spatial split:
                </p>
                <div className="metric-row">
                  <span className="metric-label">Hold-Out Zone:</span>
                  <span className="metric-val">{data.evaluation_dataset?.geographic_hold_out}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Evaluation Patches:</span>
                  <span className="metric-val font-mono">{data.evaluation_dataset?.test_patches} Non-Leaking</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Training Patches:</span>
                  <span className="metric-val font-mono">{data.evaluation_dataset?.training_patches} Patches</span>
                </div>
              </div>
            </div>

            <div className="card-panel">
              <div className="panel-header">
                <span className="panel-title">
                  <CheckCircle2 size={15} />
                  Hold-Out Structural Diagnostics
                </span>
              </div>
              <div className="panel-body">
                <div className="metric-row">
                  <span className="metric-label">Canny Edge IoU:</span>
                  <span className="metric-val font-mono">{data.structural_metrics?.edge_iou}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Canny Edge F1 Score:</span>
                  <span className="metric-val font-mono">{data.structural_metrics?.edge_f1}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Sobel Gradient Correlation (r):</span>
                  <span className="metric-val font-mono">{data.structural_metrics?.gradient_correlation_r}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Mean Radiometric ΔNDVI:</span>
                  <span className="metric-val font-mono">{data.radiometric_metrics?.mean_delta_ndvi}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Scientific Disclaimer */}
          <div
            style={{
              backgroundColor: "rgba(245, 158, 11, 0.05)",
              border: "1px solid var(--accent-amber)",
              borderRadius: "6px",
              padding: "1rem",
              display: "flex",
              alignItems: "flex-start",
              gap: "0.75rem",
            }}
          >
            <AlertCircle size={18} style={{ color: "var(--accent-amber)", flexShrink: 0, marginTop: "2px" }} />
            <div>
              <strong style={{ fontSize: "0.85rem", color: "#fed7aa", display: "block", marginBottom: "2px" }}>
                Scientific Transparency Notice:
              </strong>
              <p style={{ fontSize: "0.8rem", color: "#fbd38d" }}>
                {data.scientific_disclaimer}
              </p>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
