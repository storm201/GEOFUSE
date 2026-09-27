# GeoFUSE SentinelGuard (SIH 2026)

**Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imagery.**  
*Smart India Hackathon 2026 (SIH 2026)*

> **Core Philosophy:** *"Sharper imagery, with evidence attached."*

---

## Overview

GeoFUSE SentinelGuard is a trust-aware satellite image super-resolution pipeline designed for Sentinel-2 L2A medium-resolution imagery. Rather than merely producing nominally sharper reconstructions, GeoFUSE attaches empirical evidence to every reconstructed pixel:
- **Baseline Comparison:** Bicubic interpolation vs. lightweight deep learning SR.
- **Uncertainty Proxy:** Sequential ensemble disagreement (variance/std map).
- **Stability Analysis:** Perturbation-based sensitivity testing.
- **Spectral Consistency:** Band ratio and delta-NDVI fidelity checks.
- **Structural Consistency:** Edge gradient and local SSIM alignment.
- **Downstream Evaluation:** Light geospatial task (e.g. building footprint / feature extraction) comparison.
- **Verification & Receipts:** Synthetic degrade-and-recover validation, Trust Map fusion, and machine-readable JSON **Trust Receipt**.

---

## Directory Layout

```
GeoFUSE/
├── .gitignore              # Ignores raw/interim data, checkpoints, outputs, caches
├── requirements.txt        # Minimal, reproducible dependencies
├── config.yaml             # Single source of truth for paths, hyperparams, & thresholds
├── README.md               # Project documentation
├── src/
│   ├── __init__.py
│   ├── utils/              # Configuration parsing & device resolution
│   ├── data/               # Sentinel-2 ingest, tiling, and synthetic degradation
│   ├── models/             # Lightweight SR architectures & ensemble training
│   ├── evaluation/         # Consistency checks, uncertainty proxies, Trust Receipt
│   └── dashboard/          # Interactive Streamlit inspection dashboard
└── scripts/
    └── smoke_test.py       # Dependency & hardware validation suite
```

---

## Getting Started

### 1. One-Click Production Launcher (Recommended)
Double-click `run me.bat` in the project root (or execute via shell):
```cmd
"run me.bat"
```
Or directly start the production gateway:
```bash
python scripts/launch_server.py
```
This automatically verifies checkpoints, checks port 8000, boots the FastAPI GPU backend, serves the production React SPA, displays hardware telemetry (CUDA / VRAM / Checkpoints / 4 local datasets), and opens `http://127.0.0.1:8000` in your default browser.

### 2. Execution Workflows
- **Workflow A: Preloaded Demonstration Scenes**
  - Instant inspection of 4 local scenes (`urban_core`, `agriculture`, `temporal_april2024`, `primary_raw`).
  - Interactive 5×5 spatial partition grid (25 tiles of 128×128 px each).
  - Instant demo cache retrieval (< 1ms) with full multi-band layers (RGB, CIR, NDVI, Uncertainty, Risk, Building footprint mask).
  - Real-time `Force Live GPU Inference` bypass with the 3-member ensemble on local RTX 4060.
  - Interactive split-curtain comparison and synchronized pan/zoom at identical 1:1 geospatial footprints.
  - Cryptographically signed Trust Receipts with direct JSON export.
- **Workflow B: Custom User-Provided Imagery**
  - Drag-and-drop support for multi-spectral Sentinel-2 Level-2A rasters (4 separate band files `B02`, `B03`, `B04`, `B08` or a single 4-band GeoTIFF / JP2).
  - Strict Level-2A validation (GSD ~10m, minimum 128×128 pixels required for learned 2.5× super-resolution).
  - Automatic macro scene overview and 5×5 interactive tile grid partitioning.
  - Actual GPU inference with complete provenance isolation (never substitutes demo results for custom uploads).

### 3. Presentation Hotkeys
While using the Hero Viewer, presentation hotkeys allow rapid demonstration:
- `1` : Switch to **RGB Natural Color** (B04, B03, B02)
- `2` : Switch to **CIR False Color Infrared** (B08, B04, B03)
- `3` : Switch to **NDVI Vegetation Index** ((B08 - B04) / (B08 + B04))
- `Space` : Toggle between **Interactive Split Curtain** and **Side-by-Side View**
- `F` : Toggle **Fullscreen Presentation Mode** (hides sidebars/headers, displays floating HUD)

### 4. Storage & Retention Management
- Configurable 7-day TTL and 5GB LRU storage cap protects disk usage during long demo sessions.
- Inspect storage: `GET /api/system/storage`
- Trigger cleanup: `POST /api/system/cleanup` (supports `dry_run=true` simulation).

### 5. Streamlit Dashboard Fallback
The original interactive Streamlit dashboard remains fully functional:
```bash
streamlit run src/dashboard/app.py
```
(Or select option `[2]` in `run me.bat`).

---

## Scientific Rigor & Geographic Hold-Out Strategy

In Earth Observation (EO) and satellite imagery super-resolution, standard random patch splitting is scientifically flawed due to spatial autocorrelation: adjacent overlapping tiles in training and validation sets cause data leakage, leading to artificially inflated accuracy metrics.

GeoFUSE SentinelGuard enforces an **independent contiguous geographic hold-out split**:
- **Scene Dimensions**: 512 × 512 pixels (approx. 5.12 km × 5.12 km) at 10m Ground Sample Distance (GSD).
- **Validation Hold-Out Zone**: Strictly contiguous Southeast Quadrant (`rows 256..512, cols 256..512`). Total: 49 non-leaking evaluation patches.
- **Training Zone**: North and West sub-regions. Total: 120 training patches.
- **Zero Spatial Leakage**: Strict spatial separation guarantees that model generalizability is tested on completely unseen landscape geometry.

### Quantitative Benchmark Comparison (Hold-Out Evaluation)

| Architecture / Method | Parameters | Val Loss (Compound L1+Sobel) | Val PSNR (dB) | Val SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Bicubic Baseline (2x)** | 0 (Interpolation) | 0.01920 | 38.19 dB | 0.9208 |
| **ResidualSRNet (Ours)** | 273,700 (~0.27M) | **0.01726** | **38.68 dB** | **0.9270** |
| *Delta vs. Baseline* | — | *-0.00194* | *+0.49 dB* | *+0.0062* |

---

## Verification & Consistency Checks (Phase 7)

To ensure super-resolution models do not introduce radiometric distortion or structural hallucinations, GeoFUSE SentinelGuard performs two independent verification checks against pre-degradation reference imagery:

1. **Radiometric / Spectral Consistency**:
   - Computes $\Delta\text{NDVI} = |\text{NDVI}_{SR} - \text{NDVI}_{GT}|$ using red (B04) and near-infrared (B08) bands.
   - Flags pixels exceeding tolerance threshold ($\tau = 0.05$).
   - Computes multi-spectral Green/Red ratio consistency.
   - Raises explicit `SpectralBandError` if required spectral bands are missing or mismatched.
   - **Empirical Results**: Mean $\Delta\text{NDVI} \approx 0.021$ across diverse tiles, with $>92\%$ of pixels within tolerance.

2. **Structural & Edge Alignment**:
   - Quantifies boundary preservation using Canny edge maps (IoU, Precision, Recall, and F1 score).
   - Evaluates spatial high-frequency correlation via Sobel gradient magnitude Pearson correlation ($r$).
   - Generates multi-color diagnostic overlays highlighting matched edges (Green), hallucinated/shifted edges (Red), and missed edges (Cyan).
   - **Empirical Results**: Edge IoU $\approx 0.40 - 0.42$, Edge F1 $\approx 0.57 - 0.59$, Gradient Correlation $r \approx 0.78 - 0.81$.

---

## Multi-Evidence Trust/Risk Map Fusion (Phase 8)

GeoFUSE SentinelGuard synthesizes all four independent reliability signals into a unified spatial **Trust/Risk Map**:

$$\text{Risk}(x, y) = w_1 R_{\text{disag}}(x, y) + w_2 R_{\text{stab}}(x, y) + w_3 R_{\text{spec}}(x, y) + w_4 R_{\text{struct}}(x, y)$$

$$\text{Trust}(x, y) = 1.0 - \text{Risk}(x, y)$$

> [!NOTE]
> **Scientific Transparency**: This composite is an **empirical heuristic fusion**, not a calibrated Bayesian posterior probability.

- **Scale Dominance Guard**: Each individual signal is normalized to $[0.0, 1.0]$ via min-max scaling prior to combination, ensuring that higher-magnitude metrics (e.g. gradient difference) do not overpower subtle signals (e.g. stability variance).
- **Configurable Weights**: Defined in `config.yaml` ($w_{\text{disag}}=0.25, w_{\text{stab}}=0.25, w_{\text{spec}}=0.25, w_{\text{struct}}=0.25$).
- **Interpretable Output**: Provides a single per-tile scalar **Trust Score** ($0 - 100\%$) and spatial risk heatmaps (Green = High Trust, Red = High Risk).
- **Empirical Benchmark Across Held-Out Tiles**:
  - Sample #0: Trust Score **86.53%**, Mean Risk 0.1347, High-Risk Flagged: 0.00%
  - Sample #1: Trust Score **86.98%**, Mean Risk 0.1302, High-Risk Flagged: 0.00%
  - Sample #2: Trust Score **86.11%**, Mean Risk 0.1389, High-Risk Flagged: 0.00%
  - Sample #3: Trust Score **85.49%**, Mean Risk 0.1451, High-Risk Flagged: 0.00%

---

## Downstream Task Evaluation: Building Footprint Analysis (Phase 9)

To verify the real-world operational utility of super-resolved imagery, GeoFUSE SentinelGuard executes a canonical downstream Earth Observation task: morphological building footprint extraction (white top-hat filtering + NDVI vegetation rejection + component filtering).

> [!IMPORTANT]
> **Scientific Honesty**:
> In the absence of certified independent high-resolution vector building footprints, all metrics are reported strictly as **Bicubic vs. SR Reconstruction Agreement (IoU / Dice)** and **Relative Agreement against Pre-Degradation Reference HR Tile**, never as fabricated "ground-truth accuracy".

- **Trust Map Correlation**: Footprint agreement is stratified across High-Trust ($\ge 0.85$) and Low-Trust ($< 0.85$) geographic zones from Phase 8.
- **Empirical Results Across Held-Out Tiles**:
  - Sample #0: Bic-SR IoU **0.9285**, High-Trust IoU **0.9248**, Low-Trust IoU **0.9364**, SR-Ref IoU **0.4174** (65.1% High-Trust)
  - Sample #1: Bic-SR IoU **0.9461**, High-Trust IoU **0.9508**, Low-Trust IoU **0.9343**, SR-Ref IoU **0.6137** (67.5% High-Trust)
  - Sample #2: Bic-SR IoU **0.9253**, High-Trust IoU **0.9225**, Low-Trust IoU **0.9316**, SR-Ref IoU **0.5691** (70.8% High-Trust)
  - Sample #3: Bic-SR IoU **0.9058**, High-Trust IoU **0.9004**, Low-Trust IoU **0.9131**, SR-Ref IoU **0.5268** (63.6% High-Trust)
- **Overlay Diagnostics**: Multi-panel previews (`outputs/previews/downstream_footprint_overlay_sample_*.png`) visualize model agreement (Cyan) vs. boundary discrepancy (Orange) atop the scene.

---

## Interactive Demonstration Dashboard (Phase 10)

GeoFUSE SentinelGuard includes an interactive Streamlit dashboard (`src/dashboard/app.py`) for live comparative inspection:

- **Side-by-Side Verification**: Simultaneous 4-column display of:
  1. **Original Reference (10m)** [Pre-degradation Sentinel-2]
  2. **Bicubic Baseline (2x)** [Standard interpolation]
  3. **GeoFUSE SR (2x)** [Ensemble Mean Reconstruction]
  4. **Trust / Risk Map Overlay** [RdYlGn colormap: Green = High Trust, Red = High Risk]
- **Downstream Task Toggle**: Interactive inspection of building footprint contours, consensus masks, and trust stratification statistics.
- **Evidence Breakdown Toggle**: Live inspection of individual evidence maps (disagreement, stability, $\Delta$NDVI, gradient error).
- **Cached Inference**: Utilizes Streamlit resource caching to ensure rapid, responsive tile navigation without timeouts or redundant compute.

### Launching the Dashboard:
```bash
streamlit run src/dashboard/app.py
```

---

## Auditable Trust Receipts (Phase 11)

In mission-critical geospatial analysis, super-resolved imagery should never be delivered as an unverified visual output. GeoFUSE SentinelGuard implements an auditable **Trust Receipt generator** that pairs every reconstructed tile with a cryptographically timestamped, machine-readable JSON receipt (`outputs/receipts/trust_receipt_sample_*.json` and `examples/trust_receipt_sample_3.json`) and an interactive HTML report card.

### Receipt Components
1. **Model Provenance**: Records the architecture (`ResidualSRNet`), total trainable parameter count (~0.27M), scaling factor (2x), and exact ensemble checkpoint paths (`outputs/checkpoints/ensemble_member_{0,1,2}.pth`).
2. **Geospatial & Acquisition Metadata**: Extracted directly from GeoTIFF headers (`EPSG:32643`, 10m GSD, MGRS `43PGQ`, Sentinel-2A, spatial bounding box coordinates).
3. **Strict Scientific Honesty Mandate**: Metadata fields not present in standalone band GeoTIFF headers (such as `cloud_cover_percentage`, `sun_elevation_angle_deg`, `sun_azimuth_angle_deg`, and `satellite_orbit_number`, which reside in XML SAFE manifests) are explicitly marked as `"unavailable"` rather than guessed or fabricated.
4. **Empirical Evidence Summary**: Disagreement std ($0.0163 - 0.0195$), stability variance ($0.00018 - 0.00020$), mean $\Delta\text{NDVI}$ ($0.021 - 0.022$), edge gradient correlation ($0.78 - 0.81$), and fused trust score ($86.02\% - 86.75\%$).
5. **Downstream Task Metrics**: Bicubic vs. SR building footprint agreement (IoU $\approx 0.91 - 0.95$), stratified by high-trust and low-trust zones.
6. **Automated Risk & Trust Warnings**: Evaluates the fused trust score against a configurable threshold (`min_trust_score_threshold: 86.5%`). If trust is compromised, an unambiguous plain-language warning is attached to the receipt (e.g. Sample #2 at 86.41% and Sample #3 at 86.02% trigger `LOW TRUST WARNING`).

### Generating Trust Receipts via CLI:
```bash
python scripts/generate_trust_receipts.py
```

### Interactive Dashboard Viewer:
Inside the Streamlit dashboard (`src/dashboard/app.py`), navigate to **Tab 2: "📜 Auditable Trust Receipt (JSON & HTML)"** to view:
- Color-coded HTML status banner (High Trust Approved vs. Low Trust Warning).
- Comprehensive metadata and metrics summary tables.
- Interactive JSON schema tree.
- Direct **"Download Trust Receipt (JSON)"** button for automated ingestion into downstream GIS workflows.

---

## End-to-End Reproducibility & Evaluation (Phase 12)

A dedicated, comprehensive reproduction guide is available in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

### 1-Line Full Pipeline Reproduction:
To verify the entire pipeline end-to-end (data inspection, baseline generation, ensemble inference, stability checks, consistency checks, trust fusion, downstream evaluation, trust receipts, and test suite):
```bash
python scripts/reproduce_all.py --skip-training
```

To force complete sequential retraining of all 3 ensemble members from scratch:
```bash
python scripts/reproduce_all.py --force-retrain
```

### Modular Pipeline Commands:
```bash
# 1. Inspect data and generate RGB previews
python scripts/inspect_sentinel2.py

# 2. Generate 2x degrade-and-recover baseline triplets
python scripts/generate_baseline_triplets.py

# 3. Train all 3 ensemble members sequentially (seeds [42, 101, 2024])
python scripts/train.py

# 4. Run ensemble inference and generate disagreement map
python scripts/run_ensemble_inference.py

# 5. Run input-perturbation stability tests
python scripts/test_stability.py

# 6. Run spectral and structural consistency checks
python scripts/test_consistency_checks.py

# 7. Generate fused Trust / Risk maps
python scripts/generate_trust_risk_maps.py

# 8. Evaluate downstream building footprint agreement
python scripts/evaluate_downstream_task.py

# 9. Generate auditable Trust Receipts
python scripts/generate_trust_receipts.py

# 10. Run automated PyTest test suite (57 tests)
pytest tests/ -v

# 11. Launch interactive Streamlit dashboard
streamlit run src/dashboard/app.py
```

### Empirical Verification Benchmark (Zero Fabrication)

> [!IMPORTANT]
> **Scientific Honesty Mandate**:
> The PSNR and SSIM metrics below are **supplementary metrics only**, computed strictly within the **synthetic degrade-and-recover setting**, and are **NOT** mathematical proof of true high-resolution recovery. All values are pulled directly from verified training and evaluation logs:

| Model / Method | Seed | Parameters | Final Val Loss (L1 + 0.1·Sobel) | Val PSNR (dB) | Val SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bicubic Baseline (2x)** | — | 0 (Interpolation) | 0.01920 | 38.19 dB | 0.9208 |
| **ResidualSRNet (Single)** | 42 | 273,700 (~0.27M) | **0.01726** | **38.68 dB** | **0.9270** |
| **Ensemble Member 0** | 42 | 273,700 (~0.27M) | 0.01728 | 38.67 dB | 0.9272 |
| **Ensemble Member 1** | 101 | 273,700 (~0.27M) | 0.01733 | 38.66 dB | 0.9265 |
| **Ensemble Member 2** | 2024 | 273,700 (~0.27M) | 0.01737 | 38.62 dB | 0.9261 |
| **Ensemble Mean** | — | 821,100 (3×0.27M) | **0.01733** | **38.65 dB** | **0.9266** |
| *Advantage vs. Baseline* | — | — | *-0.00187* | *+0.46 dB* | *+0.0058* |

---

## Demo Hardening & Live Presentation (Phase 13)

To ensure presentation reliability, GeoFUSE SentinelGuard includes an offline demo hardening layer designed to eliminate live inference latency and runtime failure points:

### 1. Precomputed Offline Demo Cache (`outputs/demo_cache/`):
- All 4 demo tiles are precomputed and serialized (reconstructions, uncertainty maps, stability maps, spectral/edge consistency, downstream footprints, and trust receipts).
- The Streamlit dashboard automatically detects and loads precomputed bundles (`outputs/demo_cache/demo_tile_*.pkl`) in **< 10 milliseconds**.
- Presentation mode operates **100% offline**, requiring **zero live model forward passes**, eliminating GPU VRAM exhaustion, CPU bottlenecks, and timeout risks.
- Recompute or refresh the demo cache at any time:
  ```bash
  python scripts/precompute_demo_cache.py
  ```

### 2. Fault-Tolerant Error Handling:
- Deleting raw GeoTIFF files or model checkpoints **will not crash the application**.
- All file access and processing routines are protected by graceful fallback handlers that display clear, actionable UI guidance rather than unhandled Python exceptions.

### 3. Confirmed Live Demonstration of the Trust Guard Mechanism:
- **High-Trust Approved Examples**:
  - `Tile #0` — Central Settlement Cluster (Trust Score: **86.75%** $\ge$ 86.50% threshold)
  - `Tile #8` — Agricultural & Rural Roads (Trust Score: **86.70%** $\ge$ 86.50% threshold)
- **Deliberately Flagged Low-Trust Demonstration Examples**:
  - `Tile #16` — Rural River Corridor (Trust Score: **86.41%** < 86.50% threshold $\rightarrow$ **`LOW TRUST WARNING`**)
  - `Tile #24` — Complex Terrain Transition (Trust Score: **86.02%** < 86.50% threshold $\rightarrow$ **`LOW TRUST WARNING`**)
- Selecting Tile #16 or Tile #24 in the dashboard immediately triggers a prominent **Live Demonstration Trust Guard Warning Banner**, visually illustrating how GeoFUSE SentinelGuard flags unreliable super-resolution reconstructions to protect downstream automated decisions.

---

## Project Summary & Presentation Assets (Phase 14)

For the final hackathon submission and rapid review, GeoFUSE SentinelGuard provides a standalone one-page summary document ([`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)) and an exported high-resolution presentation gallery in [`outputs/presentation/`](outputs/presentation/):

### Standalone Documentation:
- **One-Page Project Summary**: [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) — Synthesizes the core problem, engineering approach, empirical results as actually measured, and the explicit scientific honesty statement.
- **Scientific Honesty Mandate**: Section 4 of [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) details *"What This System Does NOT Claim"*, stating boundaries regarding synthetic degrade-and-recover evaluation, heuristic trust fusion vs. Bayesian posteriors, and absence of fabricated metadata.

### Exported Presentation Assets:
The figures below are generated directly from verified offline cache artifacts at 150 DPI with custom dark-themed styling matching the Streamlit interface:

| Presentation Figure | Description | Key Concept Illustrated |
| :--- | :--- | :--- |
| [`01_side_by_side_super_resolution.png`](outputs/presentation/01_side_by_side_super_resolution.png) | 4-Column Side-by-Side Super Resolution | Reference 10m vs. Bicubic 2x vs. GeoFUSE SR vs. Trust Map overlay with comparative scorecards. |
| [`02_trust_guard_live_warning.png`](outputs/presentation/02_trust_guard_live_warning.png) | Live Demonstration: Trust Guard Warning | Tile #16 flagged below 86.50% threshold with risk advisory and evidence breakdown. |
| [`03_evidence_breakdown_signals.png`](outputs/presentation/03_evidence_breakdown_signals.png) | Multi-Source Evidence Signal Breakdown | 4-panel normalized inspection of Disagreement $\sigma$, Stability $\text{Var}$, $\Delta\text{NDVI}$, and Gradient error. |
| [`04_downstream_footprint_analysis.png`](outputs/presentation/04_downstream_footprint_analysis.png) | Downstream Building Footprint Analysis | Bicubic contours, GeoFUSE SR contours, and Footprint consensus vs. boundary discrepancy. |
| [`05_auditable_trust_receipt_report.png`](outputs/presentation/05_auditable_trust_receipt_report.png) | Auditable Trust Receipt Report | Structured HTML report card, geospatial metadata, evidence metrics, and JSON schema explorer. |

To regenerate these presentation assets at any time:
```bash
python scripts/export_presentation_assets.py
```








