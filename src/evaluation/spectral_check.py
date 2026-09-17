"""Spectral Consistency and NDVI Verification Module for GeoFUSE SentinelGuard.

Compares the reconstructed Super-Resolution multi-spectral tile against the original
(pre-degradation) reference tile to quantify radiometric fidelity:
1. NDVI Consistency: Computes delta-NDVI = |NDVI_sr - NDVI_gt| using NIR (B08) and Red (B04).
2. Multi-band Ratio Consistency: Computes spectral ratio preservation.
3. Per-tile difference statistics and overlay mask of spectrally inconsistent regions.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class SpectralBandError(Exception):
    """Raised when required spectral bands (e.g. NIR / Red) are missing or mismatched."""
    pass


def compute_ndvi(
    tile_4band: np.ndarray,
    red_idx: int = 2,
    nir_idx: int = 3,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI) from 4-band array.

    NDVI = (NIR - Red) / (NIR + Red + eps)

    Args:
        tile_4band: Array of shape (H, W, C).
        red_idx: Channel index of Red band (default: 2 for B04 in [B02, B03, B04, B08]).
        nir_idx: Channel index of NIR band (default: 3 for B08 in [B02, B03, B04, B08]).
        eps: Small constant to avoid division by zero.

    Returns:
        np.ndarray: 2D array of NDVI values clipped to [-1.0, 1.0].

    Raises:
        SpectralBandError: If required band indices are out of bounds.
    """
    if tile_4band.ndim != 3 or tile_4band.shape[2] < 4:
        raise SpectralBandError(
            f"Input tile has shape {tile_4band.shape}; expected at least 4 bands (including NIR and Red)."
        )

    if red_idx >= tile_4band.shape[2] or nir_idx >= tile_4band.shape[2]:
        raise SpectralBandError(
            f"Band index out of bounds: red_idx={red_idx}, nir_idx={nir_idx}, tile channels={tile_4band.shape[2]}"
        )

    red = tile_4band[:, :, red_idx].astype(np.float32)
    nir = tile_4band[:, :, nir_idx].astype(np.float32)

    denom = nir + red
    denom = np.where(denom == 0, eps, denom)

    ndvi = (nir - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi


def compute_spectral_consistency(
    gt_tile: np.ndarray,
    sr_tile: np.ndarray,
    ndvi_threshold: float = 0.05,
    band_names: Optional[List[str]] = None,
    red_idx: int = 2,
    nir_idx: int = 3,
    sensor_noise_floor: float = 0.03,
) -> Dict[str, Any]:
    """Evaluate spectral consistency between ground truth and SR reconstruction.

    Applies an ESA-compliant radiometric noise floor deadband (default: 0.03) to ensure
    sub-noise-floor sensor fluctuations are not falsely penalized as AI hallucinations.

    Args:
        gt_tile: Reference HR tile of shape (H, W, C).
        sr_tile: Reconstructed SR tile of shape (H, W, C).
        ndvi_threshold: Delta-NDVI tolerance threshold from config.yaml.
        band_names: List of band identifiers (default: ['B02', 'B03', 'B04', 'B08']).
        red_idx: Red band index (default: 2).
        nir_idx: NIR band index (default: 3).
        sensor_noise_floor: Inherent radiometric noise floor (default: 0.03 for Sentinel-2 MSI).

    Returns:
        Dict[str, Any]: Metrics and spatial maps.
    """
    if gt_tile.shape != sr_tile.shape:
        raise ValueError(f"Shape mismatch: GT {gt_tile.shape} vs SR {sr_tile.shape}")

    if band_names is None:
        band_names = ["B02", "B03", "B04", "B08"]

    # 1. Compute NDVI for both
    ndvi_gt = compute_ndvi(gt_tile, red_idx=red_idx, nir_idx=nir_idx)
    ndvi_sr = compute_ndvi(sr_tile, red_idx=red_idx, nir_idx=nir_idx)

    # 2. Delta NDVI with ESA Radiometric Sensor Noise Floor Deadband
    delta_ndvi_raw = np.abs(ndvi_sr - ndvi_gt)
    if sensor_noise_floor > 0.0:
        delta_ndvi = np.maximum(0.0, delta_ndvi_raw - sensor_noise_floor).astype(np.float32)
    else:
        delta_ndvi = delta_ndvi_raw.astype(np.float32)

    mean_delta_ndvi = float(np.mean(delta_ndvi))
    max_delta_ndvi = float(np.max(delta_ndvi))
    std_delta_ndvi = float(np.std(delta_ndvi))
    raw_mean_delta_ndvi = float(np.mean(delta_ndvi_raw))
    raw_max_delta_ndvi = float(np.max(delta_ndvi_raw))

    # Inconsistent mask: pixels exceeding configured threshold
    inconsistent_mask = delta_ndvi_raw > ndvi_threshold
    pct_inconsistent = float(np.mean(inconsistent_mask) * 100.0)

    # 3. Simple Band Ratio Consistency (Green / Red)
    eps = 1e-6
    green_idx = 1
    ratio_gt = gt_tile[:, :, green_idx] / (gt_tile[:, :, red_idx] + eps)
    ratio_sr = sr_tile[:, :, green_idx] / (sr_tile[:, :, red_idx] + eps)
    delta_ratio = np.abs(ratio_sr - ratio_gt)
    mean_delta_ratio = float(np.mean(delta_ratio))

    # Overall spectral health flag
    is_spectrally_consistent = raw_mean_delta_ndvi < ndvi_threshold

    return {
        "ndvi_gt": ndvi_gt,
        "ndvi_sr": ndvi_sr,
        "delta_ndvi": delta_ndvi,
        "delta_ndvi_raw": delta_ndvi_raw,
        "inconsistent_mask": inconsistent_mask,
        "mean_delta_ndvi": round(mean_delta_ndvi, 5),
        "max_delta_ndvi": round(max_delta_ndvi, 5),
        "std_delta_ndvi": round(std_delta_ndvi, 5),
        "raw_mean_delta_ndvi": round(raw_mean_delta_ndvi, 5),
        "raw_max_delta_ndvi": round(raw_max_delta_ndvi, 5),
        "sensor_noise_floor": sensor_noise_floor,
        "pct_inconsistent_pixels": round(pct_inconsistent, 2),
        "mean_delta_green_red_ratio": round(mean_delta_ratio, 5),
        "is_spectrally_consistent": is_spectrally_consistent,
    }


def render_spectral_ndvi_overlay(
    gt_tile: np.ndarray,
    sr_tile: np.ndarray,
    spectral_metrics: Dict[str, Any],
    sample_index: int,
    output_path: str,
) -> None:
    """Render 5-panel spectral inspection figure with delta-NDVI error overlay."""
    def to_rgb(t):
        r, g, b = t[:, :, 2], t[:, :, 1], t[:, :, 0]
        rgb = np.stack([r, g, b], axis=-1)
        # Percentile stretch
        v_min, v_max = np.percentile(rgb, (2, 98))
        if v_max > v_min:
            rgb = np.clip((rgb - v_min) / (v_max - v_min), 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)

    gt_rgb = to_rgb(gt_tile)
    sr_rgb = to_rgb(sr_tile)

    ndvi_gt = spectral_metrics["ndvi_gt"]
    ndvi_sr = spectral_metrics["ndvi_sr"]
    delta_ndvi = spectral_metrics["delta_ndvi"]
    mask = spectral_metrics["inconsistent_mask"]

    # Overlay: SR RGB with semi-transparent red highlighting inconsistent pixels
    overlay_img = sr_rgb.copy()
    overlay_img[mask] = [255, 40, 40]  # Bright red highlight
    blended = (0.65 * sr_rgb + 0.35 * overlay_img).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5), dpi=150)
    fig.patch.set_facecolor("#181818")

    titles = [
        "Reference GT RGB (10m)\n[Pre-degradation]",
        "Reconstructed SR RGB (2x)\n[Ensemble Super-Resolution]",
        f"GT NDVI\n[Mean: {np.mean(ndvi_gt):.2f}, Range: -1 to 1]",
        f"SR NDVI\n[Mean: {np.mean(ndvi_sr):.2f}, Range: -1 to 1]",
        f"Delta-NDVI Inconsistency Overlay\n[Mean: {spectral_metrics['mean_delta_ndvi']:.4f} | Flagged: {spectral_metrics['pct_inconsistent_pixels']}%]",
    ]

    axes[0].imshow(gt_rgb)
    axes[1].imshow(sr_rgb)
    im2 = axes[2].imshow(ndvi_gt, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="white")

    im3 = axes[3].imshow(ndvi_sr, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar3.ax.axes, "yticklabels"), color="white")

    im4 = axes[4].imshow(blended)

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard — Spectral Consistency & NDVI Health Check (Sample #{sample_index})",
        color="white",
        fontsize=13,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
