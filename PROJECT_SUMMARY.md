# GeoFUSE SentinelGuard: Comprehensive Project Summary & Trust Engine Architecture

> **Core Mandate:** *"Deep Learning Based Super-Resolution Mapping (SRM) from Medium-Resolution Satellite Imageries — With Empirical Evidence Attached."*

---

## 1. Executive Summary

**GeoFUSE SentinelGuard** is an open-source, evidence-backed deep learning super-resolution and spatial reliability verification system designed for Earth Observation (EO) satellite imagery.

### The Operational Challenge
- **Spatial Resolution Limitations:** The European Space Agency (ESA) Sentinel-2 constellation provides global multispectral coverage with a 5-day revisit cadence for free. However, its highest-resolution bands (B02 Blue, B03 Green, B04 Red, B08 NIR) are captured at a native **10.0-meter Ground Sampling Distance (GSD)**. At 10m per pixel, critical high-value features—such as individual rural homesteads, informal settlements, farm field margins, and narrow roads—are blurred.
- **Cost of Sub-Meter Commercial Constellations:** Procuring tasking or archival data from commercial sub-meter satellites (WorldView, Pleiades) costs thousands of dollars per capture and lacks frequent, open temporal revisit.
- **The "AI Hallucination" Hazard:** Traditional deep learning super-resolution techniques (such as bicubic interpolation or unconstrained generative adversarial networks) often invent plausible-looking high-frequency artifacts that do not exist on the ground, corrupting scientific vegetation indices and falsifying building polygons.
- **The Ground-Truth Void:** In academic research, algorithms evaluate "degraded" images where the original image is treated as ground truth to calculate PSNR and SSIM. In real-world deployment on genuine Sentinel-2 data, **no ground truth exists**. Traditional pixel metrics cannot be computed.

### The GeoFUSE Solution
1. **Direct 2.5× Super-Resolution Mapping:** Expands real ESA Sentinel-2 L2A observations from **10.0m GSD to nominal 4.0m GSD** using trained residual convolutional ensembles. This shrinks the ground footprint of each pixel from $100\,\text{m}^2$ down to $16\,\text{m}^2$ (an **$84\%$ reduction in pixel blur area**).
2. **SentinelGuard Empirical Trust Engine:** Rather than treating model outputs as black-box truths, GeoFUSE attaches an **auditable, pixel-level Trust & Risk Map** and an **auditable SHA-256 cryptographic provenance receipt** to every processed tile.
3. **Dual Operational Modes:** Supports both direct real-world satellite inference (Mode 1) and rigorous synthetic benchmark validation (Mode 2).

---

## 2. Dual Operational Pipeline

```
                                  ┌───────────────────────────────┐
                                  │      SENTINEL-2 L2A INPUT     │
                                  │  (B02, B03, B04, B08 @ 10.0m) │
                                  └───────────────┬───────────────┘
                                                  │
                        ┌─────────────────────────┴─────────────────────────┐
                        ▼                                                   ▼
       ┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
       │             MODE 1              │                 │             MODE 2              │
       │    DIRECT REAL SENTINEL-2 SR    │                 │  CONTROLLED SYNTHETIC BENCHMARK │
       │     (Primary Operational)       │                 │     (Scientific Validation)     │
       ├─────────────────────────────────┤                 ├─────────────────────────────────┤
       │ • Zero Synthetic Degradation    │                 │ • Synthetic 2.5× PSF Blur+Noise │
       │ • Live Pre-Loaded or Uploaded   │                 │ • Ground Truth Reference Target │
       │ • Nominal 4.0m Grid Expansion   │                 │ • PSNR / SSIM / FFT Evaluation  │
       │ • Empirical Multi-Signal Trust  │                 │ • Baseline vs. SR Comparison    │
       └────────────────┬────────────────┘                 └────────────────┬────────────────┘
                        │                                                   │
                        ▼                                                   ▼
       ┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
       │   TRUST ENGINE & RISK OVERLAY   │                 │      QUAD-VIEW BENCHMARK        │
       │  (Approved vs. Advisory Gate)   │                 │  (LR vs Bicubic vs SR vs Target)│
       └─────────────────────────────────┘                 └─────────────────────────────────┘
```

| Operational Dimension | **Mode 1: Direct Real Sentinel-2 (Primary)** | **Mode 2: Controlled Synthetic Benchmark** |
| :--- | :--- | :--- |
| **Input Data Source** | Genuine ESA Sentinel-2 MSI Level-2A surface reflectance | High-resolution reference tiles |
| **Synthetic Degradation** | **OFF (Zero blur, zero downsampling injected)** | **ON (2.5× optical PSF blur + decimation + noise)** |
| **Ground Truth Reference** | **Non-existent (Real satellite observations)** | Present (original unseen 10m target tile) |
| **Evaluation Framework** | **Multi-Criteria Trust Engine & Disagreement** | Quantitative Fidelity: PSNR, SSIM, MAE, LapVar |
| **Operational Goal** | Immediate GIS deployment for urban & agriculture | Scientific benchmark proving learned SR beats Bicubic |

---

## 3. How the Models Decide the Trust Rate (The Trust Engine)

The Trust Rate is **not** a decorative confidence score. It is derived through a mathematically formulated **multi-signal evidence fusion pipeline** implemented in [`src/models/ensemble.py`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/models/ensemble.py), [`src/evaluation/fusion.py`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/evaluation/fusion.py), and [`src/inference/real_inference.py`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/inference/real_inference.py).

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              4 RAW EVIDENCE STREAMS                     │
                        ├────────────────────────────┬────────────────────────────┤
                        │ 1. Ensemble Disagreement   │ 2. Perturbation Stability  │
                        │    (Epistemic Uncertainty) │    (Sensor Sensitivity)    │
                        ├────────────────────────────┼────────────────────────────┤
                        │ 3. Spectral NDVI Error     │ 4. Structural Gradient     │
                        │    (Radiometric Fidelity)  │    (Edge Preservation)     │
                        └────────────────────────────┴────────────────────────────┘
                                                     │
                                                     ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │             MIN-MAX NORMALIZATION TO [0.0, 1.0]         │
                        │        S_norm = (S - min(S)) / (max(S) - min(S))        │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │               WEIGHTED COMPOSITE RISK MAP               │
                        │    Risk = 0.25·E1_norm + 0.25·E2_norm +                 │
                        │           0.25·E3_norm + 0.25·E4_norm                   │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │               COMPOSITE TRUST MAP & SCORE               │
                        │    Trust(x,y) = 1.0 - Risk(x,y)                         │
                        │    Trust Score (%) = Mean(Trust) × 100%                 │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │            OPERATIONAL DECISION GATE (86.5%)            │
                        ├────────────────────────────┬────────────────────────────┤
                        │  Trust Score ≥ 86.5%       │  Trust Score < 86.5%       │
                        │  HIGH_TRUST_APPROVED       │  LOW_TRUST_ADVISORY        │
                        │  (Automated Ingestion OK)  │  (Human-in-the-Loop QA)    │
                        └────────────────────────────┴────────────────────────────┘
```

---

### Step 1: Epistemic Uncertainty via Ensemble Disagreement ($\sigma$)
- **Source Code:** [`src/models/ensemble.py#L183-L196`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/models/ensemble.py#L183-L196)
- **Concept:** Measures model uncertainty by comparing independent hypotheses. GeoFUSE runs an ensemble of $M = 3$ independently trained `ResidualSRNet` networks (trained with different random seeds: `42`, `101`, `2024`) on the input raster $x \in \mathbb{R}^{H \times W \times 4}$.
- **Mathematical Formulation:**
  The ensemble consensus super-resolved prediction is the arithmetic mean across all models:
  $$\bar{y}(i, j, c) = \frac{1}{M} \sum_{m=1}^M y_m(i, j, c)$$
  The per-pixel epistemic disagreement map $\sigma(i, j)$ is the standard deviation across models, averaged across the 4 spectral bands:
  $$\sigma(i, j) = \frac{1}{C} \sum_{c=1}^C \sqrt{\frac{1}{M-1} \sum_{m=1}^M \left( y_m(i, j, c) - \bar{y}(i, j, c) \right)^2}$$
- **Physical Meaning:**
  - In uniform areas (open water, flat agricultural fields), all 3 models converge on identical values ($\sigma \approx 0$).
  - At complex, ambiguous high-frequency boundaries (dense building corners, riverbanks), the models diverge slightly ($\sigma > 0$), signaling higher uncertainty.

---

### Step 2: Input-Perturbation Stability Testing (Sensor Noise Sensitivity)
- **Source Code:** [`src/inference/real_inference.py#L518-L539`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/inference/real_inference.py#L518-L539)
- **Concept:** Satellite sensors experience slight atmospheric fluctuations, sensor thermal noise, and radiometric calibration drift. A dependable model must be robust to minor input noise and not hallucinate radical structural shifts.
- **Mathematical Formulation:**
  Controlled zero-mean Gaussian noise ($\eta \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ with $\sigma_{\text{noise}} \in [0.01, 0.02]$) and brightness jitter are added to the input:
  $$x^{(k)} = \text{clip}(x + \eta_k, 0.0, 1.0)$$
  The variance across $K$ perturbed forward passes is calculated per pixel:
  $$S_{\text{stability}}(i, j) = \frac{1}{C} \sum_{c=1}^C \text{Var}_k \left( \hat{y}^{(k)}(i, j, c) \right)$$
- **Physical Meaning:**
  - Genuine land cover features remain geometrically stable under minor input noise.
  - Hallucinated or unstable artifacts fluctuate wildly under perturbations, producing high variance that flags the region as high-risk.

---

### Step 3: Cross-Scale Spectral NDVI Fidelity (Radiometric Consistency)
- **Source Code:** [`src/evaluation/spectral_check.py`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/evaluation/spectral_check.py) and [`src/inference/real_inference.py#L540-L549`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/inference/real_inference.py#L540-L549)
- **Concept:** Super-resolution should sharpen spatial boundaries without corrupting physical surface reflectance or biological vegetation indices.
- **Mathematical Formulation:**
  The Normalized Difference Vegetation Index (NDVI) is computed for both the 4m super-resolved product and the baseline 10m observation (upsampled via bicubic interpolation to match the 4m grid):
  $$\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04} + 10^{-8}}$$
  The absolute spectral error is computed with a calibrated sensor noise floor deadband ($\tau_{\text{noise}} = 0.03$):
  $$\Delta \text{NDVI}(i, j) = \max\left(0.0,\, |\text{NDVI}_{4\text{m}}(i, j) - \text{NDVI}_{10\text{m}}(i, j)| - 0.03\right)$$
- **Physical Meaning:**
  - Prevents the network from artificially altering crop vigor signals or inventing green vegetation on barren ground.

---

### Step 4: Structural Edge Gradient Diagnostics (High-Frequency Fidelity)
- **Source Code:** [`src/evaluation/edge_check.py`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/evaluation/edge_check.py) and [`src/inference/real_inference.py#L550-L554`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/inference/real_inference.py#L550-L554)
- **Concept:** Verifies that sharpened edges correspond to physical gradient energy rather than high-frequency checkerboard noise or ringing artifacts.
- **Mathematical Formulation:**
  Sobel convolution kernels ($K_x, K_y$) compute the spatial gradient magnitude:
  $$G_x = K_x * I, \quad G_y = K_y * I, \quad |\nabla I|(i, j) = \sqrt{G_x(i, j)^2 + G_y(i, j)^2}$$
  The structural gradient discrepancy between the 4m product and the 10m baseline is:
  $$\Delta \text{Grad}(i, j) = \left| |\nabla I_{4\text{m}}|(i, j) - |\nabla I_{10\text{m}}|(i, j) \right|$$

---

### Step 5: Min-Max Normalization & Weighted Evidence Fusion
- **Source Code:** [`src/evaluation/fusion.py#L24-L150`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/evaluation/fusion.py#L24-L150)
- **The Scale Invariance Requirement:**
  Because $\sigma$, stability variance, $\Delta\text{NDVI}$, and edge gradients have different physical dimensions and magnitudes, each signal is independently normalized to $[0.0, 1.0]$ across the tile:
  $$\tilde{S}(i, j) = \frac{S(i, j) - \min(S)}{\max(S) - \min(S) + \epsilon}$$
- **Weighted Composite Risk Map:**
  Using the configured evidence weights ($w_k = 0.25$ by default, summing to $1.0$):
  $$\text{Risk}(i, j) = w_{\text{disag}} \cdot \tilde{\sigma}(i, j) + w_{\text{stab}} \cdot \tilde{S}_{\text{stab}}(i, j) + w_{\text{spec}} \cdot \widetilde{\Delta\text{NDVI}}(i, j) + w_{\text{struct}} \cdot \widetilde{\Delta\text{Grad}}(i, j)$$
- **The Composite Trust Map & Overall Trust Rate:**
  Trust is the exact mathematical complement of risk:
  $$\text{Trust}(i, j) = 1.0 - \text{Risk}(i, j)$$
  The scalar **Composite Trust Rate (Percentage)** is the spatial mean across all $H \times W$ pixels:
  $$\text{Trust Rate (\%)} = \left( \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W \text{Trust}(i, j) \right) \times 100\%$$

---

## 4. Operational Decision Thresholding & Gatekeeping

In [`src/inference/real_inference.py#L572-L574`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/inference/real_inference.py#L572-L574) and [`config.yaml`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/config.yaml), a strict decision threshold governs automated downstream usage:

$$\text{Operational Cutoff Threshold} = 86.5\%$$

### Case A: `HIGH_TRUST_APPROVED` ($\ge 86.5\%$)
- **Condition:** All 4 evidence streams exhibit low error and low uncertainty across the tile.
- **Action:** Certified for **autonomous downstream GIS pipelines** (e.g., automated building polygon vectorization, automated cadastral parcel updates).

### Case B: `LOW_TRUST_ADVISORY` ($< 86.5\%$)
- **Condition:** High model disagreement, sensor noise sensitivity, or spectral deviation detected (often in mountainous shadows, cloud fringes, or complex water-land boundaries).
- **Action:** Halts autonomous downstream ingestion and triggers a **Human-in-the-Loop Advisory**. The dashboard renders the color-coded Trust/Risk overlay (Green = verified, Yellow/Red = advisory) so human analysts immediately know where caution is required.

---

## 5. Cryptographic Provenance & Trust Receipts

Every inference run generates an auditable, machine-readable JSON Trust Receipt ([`src/evaluation/trust_receipt.py`](file:///c:/Users/suhas/Desktop/SIH%20PROJECT/src/evaluation/trust_receipt.py)) saved alongside the 4m GeoTIFF:

```json
{
  "receipt_version": "2.1.0",
  "provenance": {
    "system_name": "GeoFUSE SentinelGuard",
    "synthetic_degradation": false,
    "input_gsd_meters": 10.0,
    "output_gsd_meters": 4.0,
    "scale_factor": 2.5,
    "device": "NVIDIA GeForce RTX 4060 Laptop GPU (CUDA)"
  },
  "verification_signals": {
    "composite_trust_score_pct": 86.82,
    "disagreement_mean": 0.00762,
    "stability_mean": 0.00018,
    "delta_ndvi_mean": 0.0194,
    "structural_diff_mean": 0.0241,
    "operational_status": "HIGH_TRUST_APPROVED"
  },
  "cryptographic_signatures": {
    "input_hash_sha256": "3a88d742e91b58...",
    "model_checkpoint_hashes": [
      "9c47fa10b9...",
      "41e8c92a14...",
      "a50c82f93d..."
    ]
  }
}
```

### Key Guarantees:
1. **Auditable Proof:** Verifies that `synthetic_degradation: false` was enforced during live inference.
2. **Deterministic Provenance:** Binds output GeoTIFF rasters directly to the exact model checkpoint SHA-256 hashes and input data hashes.

---

## 6. Downstream Geospatial Application Value

| Sector / Application | At Native Sentinel-2 10.0m GSD | At GeoFUSE Super-Resolved 4.0m GSD |
| :--- | :--- | :--- |
| **Urban Infrastructure & Built Environment** | Adjacent residential rooftops merge into indistinct contiguous pixel clusters; narrow unpaved roads disappear. | Morphological contours separate individual rooftop structures; rural roads and building corridors are delineated. |
| **Precision Agriculture & Crop Health** | Sub-field parcel heterogeneity and field boundary furrows are averaged into a single coarse NDVI value. | Internal crop vigor variation, irrigation patterns, and smallholder parcel boundaries are crisply resolvable. |
| **Disaster Response & Flood Boundary Mapping** | Coarse mixed-water pixels at shorelines create ambiguous flood inundation lines. | Crisper water-land gradient boundaries enable more precise flood extent estimation. |

---

## 7. What This System Explicitly Does NOT Claim

In keeping with our scientific honesty mandate:
1. **Not Optical High-Resolution Ground Truth:** GeoFUSE reconstructs a learned mathematical approximation on a nominal 4m grid; it does not turn a 10m sensor into a physical sub-meter spy satellite.
2. **Not a Calibrated Bayesian Posterior Probability:** The Trust Score ($0 - 100\%$) is an empirical heuristic fusion of normalized evidence proxies, not a formal Bayesian credible interval or conformal prediction bound.
3. **No Fabricated Vector Ground Truth:** Downstream building footprint extraction is evaluated as relative agreement between algorithms, never as fabricated "detection accuracy" without certified ground survey vectors.
