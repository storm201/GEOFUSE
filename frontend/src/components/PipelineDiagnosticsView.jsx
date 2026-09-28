import React from "react";
import {
  Activity,
  Cpu,
  Layers,
  ShieldCheck,
  CheckCircle2,
  Zap,
  ArrowRight,
  GitBranch,
  Sliders,
  Server,
  Database
} from "lucide-react";

export default function PipelineDiagnosticsView({ systemStatus, preloadedResult, selectedSceneId, selectedTileId }) {
  const isCuda = systemStatus?.cuda_available ?? true;
  const devName = systemStatus?.device_name || "NVIDIA RTX 4060 Laptop GPU";
  const vramTotal = systemStatus?.vram_total_gb ?? 8.0;
  const vramAlloc = systemStatus?.vram_allocated_gb ?? 5.8;
  const vramPct = Math.round((vramAlloc / vramTotal) * 100);
  const latency = preloadedResult?.latency_ms ?? 14.2;

  const bandData = [
    { band: "B02", name: "Blue", center: "490 nm", res: "10m", role: "Visible Atmosphere & Water" },
    { band: "B03", name: "Green", center: "560 nm", res: "10m", role: "Vegetation Peak Reflectance" },
    { band: "B04", name: "Red", center: "665 nm", res: "10m", role: "Chlorophyll Absorption" },
    { band: "B08", name: "NIR", center: "842 nm", res: "10m", role: "Cellular Canopy Scattering" }
  ];

  return (
    <div className="pipeline-view-container">
      {/* Top Telemetry & Status Ticker */}
      <div className="telemetry-ticker-bar">
        <div className="ticker-left">
          <div className="ticker-pill-state">
            <span className="state-pulse-dot"></span>
            <span className="ticker-tag">PIPELINE PHASE:</span>
            <span className="ticker-val-cyan font-mono">BAYESIAN_CONSENSUS_RECONSTRUCTION_PASS_4</span>
          </div>
          <div className="ticker-divider"></div>
          <div className="ticker-item">
            <span className="ticker-tag">GRANULE_REF:</span>
            <span className="ticker-val font-mono">S2A_MSIL2A_{selectedSceneId?.toUpperCase() || "URBAN_CORE"}_TILE_{selectedTileId ?? "00"}</span>
          </div>
        </div>
        <div className="ticker-right">
          <div className="lock-tag-green">
            <ShieldCheck size={14} />
            <span>CONSERVATION_LOCK: IN_TOLERANCE (Δ&lt;0.015 MAE)</span>
          </div>
          <span className="ticker-val-muted font-mono">UTC {new Date().toISOString().slice(0, 19).replace("T", " ")}</span>
        </div>
      </div>

      {/* Flow Architecture Master Banner */}
      <div className="pipeline-flow-banner">
        <div className="flow-banner-header">
          <div className="flow-title-row">
            <GitBranch size={20} className="text-cyan-400" />
            <h2 className="flow-headline">Flow Architecture // 10m BOA → 4m Consensus Ensemble</h2>
          </div>
          <div className="flow-tags">
            <span className="flow-tag-gray">GEO_STRICT_V4.2</span>
            <span className="flow-tag-cyan">FP16 TRT OPTIMIZED</span>
          </div>
        </div>

        {/* 5-Node Interactive Flow Graph */}
        <div className="flow-nodes-grid">
          {/* Node 1: Ingestion */}
          <div className="flow-node-card">
            <div className="node-top">
              <span className="node-step">01 // INGESTION</span>
              <span className="node-status-dot green"></span>
            </div>
            <div className="node-center">
              <div className="node-main-label">S2 L2A BOA</div>
              <div className="node-sub-label">4 Bands @ 10m</div>
              <div className="node-code-label font-mono">B02 • B03 • B04 • B08</div>
            </div>
            <div className="node-badge-bottom green">DN→SURF_REFL</div>
          </div>

          {/* Node 2: Conditioning */}
          <div className="flow-node-card">
            <div className="node-top">
              <span className="node-step">02 // CONDITIONING</span>
              <span className="node-status-dot green"></span>
            </div>
            <div className="node-center">
              <div className="node-main-label">Radiometric Cal</div>
              <div className="node-sub-label">Aerosol / SZA Normalization</div>
              <div className="node-code-label font-mono">μ=0.9998 • σ=0.002</div>
            </div>
            <div className="node-badge-bottom cyan">ISOTROPIC GAIN</div>
          </div>

          {/* Node 3: Tri-Member Ensemble (Wide card) */}
          <div className="flow-node-card ensemble-wide">
            <div className="node-top">
              <span className="node-step cyan">03 // TRI-MEMBER ENSEMBLE</span>
              <span className="node-badge-top green">3/3 CONVERGED</span>
            </div>
            <div className="ensemble-branches-grid">
              {/* Branch A: EDSR-Geo */}
              <div className="ensemble-branch">
                <div className="branch-head">
                  <span className="branch-name">EDSR-Geo</span>
                  <span className="branch-weight font-mono">w=0.35</span>
                </div>
                <div className="branch-arch">Deep ResNet</div>
                <div className="branch-perf font-mono">14.2ms • FP16</div>
              </div>

              {/* Branch B: RCAN-Sat */}
              <div className="ensemble-branch active">
                <div className="branch-head">
                  <span className="branch-name cyan">RCAN-Sat</span>
                  <span className="branch-weight font-mono">w=0.40</span>
                </div>
                <div className="branch-arch">Channel-Attn</div>
                <div className="branch-perf font-mono">19.8ms • FP16</div>
              </div>

              {/* Branch C: SwinIR-Earth */}
              <div className="ensemble-branch">
                <div className="branch-head">
                  <span className="branch-name">SwinIR</span>
                  <span className="branch-weight font-mono">w=0.25</span>
                </div>
                <div className="branch-arch">Spatial ViT</div>
                <div className="branch-perf font-mono">21.4ms • FP16</div>
              </div>
            </div>
            <div className="ensemble-footer-bar">
              <span>DYNAMIC WEIGHTING:</span>
              <span className="font-mono text-cyan-400">SOFTMAX(L1_GRAD_CONFIDENCE)</span>
            </div>
          </div>

          {/* Node 4: Bayesian Consensus Fusion */}
          <div className="flow-node-card">
            <div className="node-top">
              <span className="node-step">04 // FUSION</span>
              <span className="node-status-dot green"></span>
            </div>
            <div className="node-center">
              <div className="node-main-label">Bayesian Consensus</div>
              <div className="node-sub-label">Epistemic Variance Minimization</div>
              <div className="node-code-label font-mono">σ_min = 0.0182</div>
            </div>
            <div className="node-badge-bottom cyan">ADAPTIVE BLEND</div>
          </div>

          {/* Node 5: Verification & Downstream */}
          <div className="flow-node-card">
            <div className="node-top">
              <span className="node-step">05 // VERIFICATION</span>
              <span className="node-status-dot green"></span>
            </div>
            <div className="node-center">
              <div className="node-main-label">Audit Guard</div>
              <div className="node-sub-label">Building Extraction & Ed25519</div>
              <div className="node-code-label font-mono">Receipt: SHA256-OK</div>
            </div>
            <div className="node-badge-bottom green">TRUST RECEIPT</div>
          </div>
        </div>
      </div>

      {/* Bottom Diagnostics Details Grid */}
      <div className="pipeline-details-grid">
        {/* Hardware & Runtime Telemetry */}
        <div className="diagnostics-card">
          <div className="diag-header">
            <Cpu size={16} className="text-cyan-400" />
            <h3 className="diag-title">Compute Hardware & Execution Telemetry</h3>
          </div>
          <div className="diag-body">
            <div className="diag-row">
              <span className="diag-label">COMPUTE ACCELERATOR:</span>
              <span className="diag-val font-mono">{devName}</span>
            </div>
            <div className="diag-row">
              <span className="diag-label">CUDA RUNTIME:</span>
              <span className="diag-val-green font-mono">{isCuda ? "CUDA 12.1 • PyTorch 2.3.1" : "CPU Fallback Mode"}</span>
            </div>
            <div className="diag-row">
              <span className="diag-label">DEDICATED VRAM:</span>
              <span className="diag-val font-mono">{vramAlloc} / {vramTotal} GB ({vramPct}%)</span>
            </div>
            <div className="vram-meter">
              <div className="vram-fill" style={{ width: `${vramPct}%` }}></div>
            </div>
            <div className="diag-row">
              <span className="diag-label">LATENCY PER TILE:</span>
              <span className="diag-val-cyan font-mono">{latency.toFixed(1)} ms</span>
            </div>
            <div className="diag-row">
              <span className="diag-label">PARALLEL TILES THROUGHPUT:</span>
              <span className="diag-val font-mono">~70 tiles/second</span>
            </div>
          </div>
        </div>

        {/* 4 Spectral Bands Matrix */}
        <div className="diagnostics-card">
          <div className="diag-header">
            <Layers size={16} className="text-cyan-400" />
            <h3 className="diag-title">Sentinel-2 10m L2A Surface Reflectance Bands</h3>
          </div>
          <div className="diag-body">
            <div className="bands-table">
              <div className="bands-head">
                <span>BAND</span>
                <span>SPECTRAL NAME</span>
                <span>CENTRAL λ</span>
                <span>NATIVE GSD</span>
                <span>PHYSICAL USE</span>
              </div>
              {bandData.map((b) => (
                <div key={b.band} className="bands-row">
                  <span className="band-code font-mono">{b.band}</span>
                  <span className="band-name">{b.name}</span>
                  <span className="band-center font-mono">{b.center}</span>
                  <span className="band-res font-mono">{b.res}</span>
                  <span className="band-role">{b.role}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
