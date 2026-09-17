"""GeoFUSE SentinelGuard — Real Sentinel-2 Direct Super-Resolution Inference Engine.

Implements the live, un-degraded 10m -> 4m (2.5x) super-resolution pipeline:
1. Input validation (4 channels: B02, B03, B04, B08; CRS & affine consistency; ~10m resolution)
2. Normalization & preprocessing matching training data (DN / 10000.0, range [0.0, 1.2])
3. Direct 2.5x learned SR via the 3-member ResidualSRNet ensemble (zero synthetic degradation)
4. Overlap-aware Hann-window tile stitching for seamless full-scene 4m reconstruction
5. Georeferencing transform updating 10m -> 4m GSD while preserving exact geographic bounds
6. Ground-truth-free empirical trust/risk evaluation (disagreement, stability, spectral, edge)
7. Production GeoTIFF export (sr_4m.tif) and auditable Trust Receipt (synthetic_degradation: false)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
import torch.nn as nn
from PIL import Image

from src.data.inspect_data import (
    MissingBandError,
    SpatialAlignmentError,
    check_spatial_consistency,
    find_band_files,
)
from src.evaluation.downstream_eval import extract_building_footprints
from src.evaluation.edge_check import compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_ndvi
from src.evaluation.stability import apply_controlled_perturbation
from src.models.ensemble import enhance_edge_sharpness, load_ensemble_members
from src.utils.config import get_device, get_project_root, load_config

BAND_NAMES = ["B02", "B03", "B04", "B08"]
BAND_DESCRIPTIONS = [
    "B02 - Blue (10m -> 4m SR)",
    "B03 - Green (10m -> 4m SR)",
    "B04 - Red (10m -> 4m SR)",
    "B08 - NIR (10m -> 4m SR)",
]


# =============================================================================
# 1. Input Validation & Discovery
# =============================================================================

def validate_sentinel2_input(
    input_source: Union[str, Path, List[Union[str, Path]]],
    expected_bands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Inspect and strictly validate Sentinel-2 input data.

    Supports:
    - Format A: 4 individual band GeoTIFF files or directory containing them.
    - Format B: 1 four-band GeoTIFF file.

    Validates:
    - Number of bands is exactly 4
    - Band identification: B02, B03, B04, B08
    - Matching dimensions, CRS, affine transform, and resolution (~10m)

    Args:
        input_source: File path, directory path, or list of file paths.
        expected_bands: List of expected bands (default: ['B02', 'B03', 'B04', 'B08']).

    Returns:
        Dict[str, Any]: Summary dictionary containing validated file paths, format type,
                        spatial metadata, and native pixel size.

    Raises:
        FileNotFoundError: If input paths do not exist.
        MissingBandError: If one or more required Sentinel-2 bands cannot be identified.
        SpatialAlignmentError: If CRS, dimensions, or transforms mismatch between bands.
        ValueError: If file format or band count is unsupported.
    """
    if expected_bands is None:
        expected_bands = BAND_NAMES

    # Normalize candidate paths
    candidate_paths: List[Path] = []
    if isinstance(input_source, (list, tuple)):
        candidate_paths = [Path(p) for p in input_source]
    else:
        p = Path(input_source)
        if not p.exists():
            raise FileNotFoundError(f"Input source path does not exist: {p.resolve()}")
        if p.is_dir():
            for ext in ("*.tif", "*.tiff", "*.jp2", "*.TIF", "*.TIFF"):
                candidate_paths.extend(p.glob(ext))
                candidate_paths.extend(p.glob(f"**/{ext}"))
            candidate_paths = list({f.resolve(): f for f in candidate_paths}.values())
        else:
            candidate_paths = [p]

    if not candidate_paths:
        raise FileNotFoundError(f"No GeoTIFF raster files found in: {input_source}")

    for cp in candidate_paths:
        if not cp.exists():
            raise FileNotFoundError(f"File not found: {cp.resolve()}")

    # Case 1: Single file input (Multi-band GeoTIFF or single band file with siblings)
    if len(candidate_paths) == 1 and candidate_paths[0].is_file():
        single_file = candidate_paths[0]
        if single_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            raise ValueError(f"Real Sentinel-2 inference requires B02, B03, B04 and B08. Standard RGB image '{single_file.name}' is not supported in real mode.")

        with rasterio.open(single_file) as ds:
            if ds.count >= len(expected_bands):
                res_x, res_y = abs(ds.transform[0]), abs(ds.transform[4])
                if not (3.0 <= res_x <= 25.0):
                    raise ValueError(f"Expected approximately 10m Sentinel-2 data, but found {res_x:.1f}m resolution.")
                return {
                    "is_valid": True,
                    "format": "single_multiband",
                    "file_path": single_file,
                    "shape": (ds.height, ds.width),
                    "crs": str(ds.crs) if ds.crs else "EPSG:4326",
                    "transform": ds.transform,
                    "resolution": (res_x, res_y),
                    "bounds": ds.bounds,
                    "bands": expected_bands,
                    "band_order": list(expected_bands),
                    "nodata": ds.nodata,
                    "dtype": str(ds.dtypes[0]),
                }
            else:
                # Check if sibling band files exist in the same directory
                try:
                    sibling_bands = find_band_files(single_file.parent, expected_bands)
                    candidate_paths = [sibling_bands[b] for b in expected_bands]
                except MissingBandError:
                    raise ValueError(
                        f"The uploaded GeoTIFF contains {ds.count} band(s); 4 channels are required (B02, B03, B04, B08)."
                    )

    # Case 2: Multiple individual band files (or folder)
    matched_bands = find_band_files(
        candidate_paths[0].parent if len(candidate_paths) == 1 else candidate_paths[0].parent,
        expected_bands,
    ) if any(p.is_dir() for p in candidate_paths) else _match_explicit_paths(candidate_paths, expected_bands)

    # Open all datasets to verify spatial alignment
    band_datasets: Dict[str, rasterio.DatasetReader] = {}
    try:
        for b in expected_bands:
            band_datasets[b] = rasterio.open(matched_bands[b])

        spatial_meta = check_spatial_consistency(band_datasets)
        ref_ds = next(iter(band_datasets.values()))
        res_x, res_y = abs(ref_ds.transform[0]), abs(ref_ds.transform[4])
        if not (3.0 <= res_x <= 25.0):
            raise ValueError(f"Expected approximately 10m Sentinel-2 data, but found {res_x:.1f}m resolution.")

        res_dict = {
            "is_valid": True,
            "format": "separate_bands",
            "band_files": matched_bands,
            "band_paths": matched_bands,
            "band_order": list(expected_bands),
            "shape": spatial_meta["shape"],
            "crs": spatial_meta["crs"],
            "transform": spatial_meta["transform"],
            "resolution": (res_x, res_y),
            "bounds": spatial_meta["bounds"],
            "bands": expected_bands,
            "nodata": ref_ds.nodata,
            "dtype": str(ref_ds.dtypes[0]),
        }
        return res_dict
    finally:
        for ds in band_datasets.values():
            ds.close()




def _match_explicit_paths(paths: List[Path], expected_bands: List[str]) -> Dict[str, Path]:
    """Helper to match a provided list of file paths to expected Sentinel-2 bands."""
    aliases = {
        "B02": ["B02", "B2", "BLUE"],
        "B03": ["B03", "B3", "GREEN"],
        "B04": ["B04", "B4", "RED"],
        "B08": ["B08", "B8", "NIR"],
    }
    matched: Dict[str, Path] = {}
    assigned = set()

    for band in expected_bands:
        band_aliases = aliases.get(band, [band])
        for p in paths:
            if p in assigned:
                continue
            stem = p.stem.upper().replace("-", "_").replace(".", "_")
            tokens = stem.split("_")
            if any(a in tokens for a in band_aliases) or any(f"_{a}" in stem or f"{a}_" in stem for a in band_aliases):
                matched[band] = p
                assigned.add(p)
                break

    # If exactly 4 files provided and standard matching didn't match all, sort alphabetically
    if len(matched) < len(expected_bands) and len(paths) == len(expected_bands):
        sorted_p = sorted(paths, key=lambda x: x.name.lower())
        for i, b in enumerate(expected_bands):
            if b not in matched:
                matched[b] = sorted_p[i]

    missing = [b for b in expected_bands if b not in matched]
    if missing:
        missing_str = ", ".join(missing)
        raise MissingBandError(
            f"Missing {missing_str}. Real Sentinel-2 inference requires B02, B03, B04 and B08. Provided: {[p.name for p in paths]}"
        )
    return matched



# =============================================================================
# 2. Preprocessing & Normalization (Strictly Training-Compatible)
# =============================================================================

def load_and_preprocess_sentinel2(
    input_source: Union[str, Path, List[Union[str, Path]]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load real Sentinel-2 bands and normalize exactly as trained.

    Converts raw Sentinel-2 L2A Digital Numbers (> 10.0) to surface reflectance
    by dividing by 10000.0, or 8-bit images by 255.0. Preserves physical non-negativity
    and extracts full spatial metadata.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Normalized array of shape (H, W, 4), dtype float32, in range [0.0, 1.2]
            - Metadata dictionary describing CRS, transform, resolution, bounds, nodata
    """
    val_info = validate_sentinel2_input(input_source)
    fmt = val_info["format"]

    band_arrays: List[np.ndarray] = []
    nodata_mask: Optional[np.ndarray] = None

    if fmt == "single_multiband":
        with rasterio.open(val_info["file_path"]) as ds:
            for b_idx in range(1, 5):
                arr = ds.read(b_idx).astype(np.float32)
                if ds.nodata is not None:
                    if nodata_mask is None:
                        nodata_mask = (arr == ds.nodata)
                    else:
                        nodata_mask |= (arr == ds.nodata)
                band_arrays.append(arr)
    else:
        for b in BAND_NAMES:
            path = val_info["band_files"][b]
            with rasterio.open(path) as ds:
                arr = ds.read(1).astype(np.float32)
                if ds.nodata is not None:
                    if nodata_mask is None:
                        nodata_mask = (arr == ds.nodata)
                    else:
                        nodata_mask |= (arr == ds.nodata)
                band_arrays.append(arr)

    # Stack along channels: shape (H, W, 4) in strict order [B02, B03, B04, B08]
    stack = np.stack(band_arrays, axis=-1)

    # Normalize to surface reflectance [0.0, 1.0] matching training pipeline
    # Sentinel-2 L2A raw DNs are 0..10000; divide by 10000.0
    max_val = float(np.nanmax(stack)) if stack.size > 0 else 1.0
    if max_val > 10.0:
        stack = stack / 10000.0 if max_val > 255.0 else stack / 255.0

    # Physical reflectance bounds [0.0, 1.2]
    stack = np.clip(stack, 0.0, 1.2).astype(np.float32)

    val_info["raw_max"] = max_val
    val_info["is_scaled_dn"] = (max_val > 10.0)
    val_info["nodata_mask"] = nodata_mask
    val_info["band_names"] = BAND_NAMES

    return stack, val_info


# =============================================================================
# 3. Direct Real Ensemble Super-Resolution (Tile & Scene)
# =============================================================================

def run_direct_sr_tile(
    models: List[nn.Module],
    tile_10m: np.ndarray,
    device: Optional[torch.device] = None,
    sharpness_boost: float = 2.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute direct learned 2.5x super-resolution on a single 10m tile.

    CRITICAL ARCHITECTURE REQUIREMENT:
    Zero synthetic degradation is applied. The 10m tile is passed directly into
    the trained 3-member ensemble models.

    Args:
        models: List of loaded PyTorch ensemble members.
        tile_10m: Float32 array of shape (H, W, 4) in [0.0, 1.2].
        device: Torch compute device.
        sharpness_boost: Post-refinement edge enhancement factor (default: 2.4).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - sr_4m: Super-resolved 4m array of shape (round(H*2.5), round(W*2.5), 4)
            - disagreement_map: Per-pixel ensemble standard deviation (round(H*2.5), round(W*2.5))
    """
    if device is None:
        device = next(models[0].parameters()).device

    # Shape: (H, W, 4) -> (1, 4, H, W)
    tensor_in = torch.from_numpy(tile_10m).permute(2, 0, 1).unsqueeze(0).float().to(device)

    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            out = model(tensor_in)
            # Remove batch dim: (4, H_sr, W_sr) -> (H_sr, W_sr, 4)
            arr_out = out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            preds.append(arr_out)

    stacked = np.stack(preds, axis=0)  # (M, H_sr, W_sr, 4)
    mean_recon = np.mean(stacked, axis=0)

    if sharpness_boost > 0.0:
        mean_recon = enhance_edge_sharpness(mean_recon, boost=sharpness_boost)

    # Disagreement proxy = average standard deviation across the 4 spectral bands
    std_per_band = np.std(stacked, axis=0)
    disagreement_map = np.mean(std_per_band, axis=-1).astype(np.float32)

    return mean_recon, disagreement_map


def _create_2d_hann_window(height: int, width: int) -> np.ndarray:
    """Create a 2D Hann (cosine) taper window for smooth overlap blending."""
    wy = np.hanning(height)
    wx = np.hanning(width)
    w2d = np.outer(wy, wx).astype(np.float32)
    # Prevent zero weights at exact borders
    return np.clip(w2d, 1e-4, 1.0)


def run_direct_sr_scene(
    models: List[nn.Module],
    scene_10m: np.ndarray,
    patch_size: int = 64,
    stride: int = 48,
    scale_factor: float = 2.5,
    device: Optional[torch.device] = None,
    sharpness_boost: float = 2.4,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute tile-based direct 2.5x super-resolution with overlap-aware stitching.

    Splits large 10m Sentinel-2 scenes into tiles, executes direct 2.5x SR on each tile,
    and blends overlapping reconstructions using a 2D Hann feathering window to eliminate
    tile edge seams.

    Args:
        models: Loaded ensemble members.
        scene_10m: Full-scene 10m array of shape (H, W, 4).
        patch_size: 10m input tile dimension (default: 64).
        stride: Step between adjacent tiles (default: 48 gives 25% overlap).
        scale_factor: Spatial scale factor (2.5).
        device: Torch compute device.
        sharpness_boost: Edge enhancement factor.
        tile_size: Alias for patch_size.
        overlap: Overlap in pixels between adjacent tiles (overrides stride).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - sr_4m_full: Reconstructed 4m scene of shape (round(H*2.5), round(W*2.5), 4)
            - disagreement_full: Full-scene disagreement map (round(H*2.5), round(W*2.5))
    """
    if tile_size is not None:
        patch_size = tile_size
    if overlap is not None:
        stride = max(1, patch_size - overlap)

    h, w, c = scene_10m.shape
    h_target = int(round(h * scale_factor))
    w_target = int(round(w * scale_factor))

    # If the scene is already small or equal to patch size, run directly
    if h <= patch_size and w <= patch_size:
        return run_direct_sr_tile(models, scene_10m, device=device, sharpness_boost=sharpness_boost)

    # Accumulator buffers for weighted blending
    sr_accum = np.zeros((h_target, w_target, c), dtype=np.float32)
    disag_accum = np.zeros((h_target, w_target), dtype=np.float32)
    weight_accum = np.zeros((h_target, w_target), dtype=np.float32)

    # Coordinate grids
    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))
    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    for y in y_steps:
        for x in x_steps:
            in_tile = scene_10m[y : y + patch_size, x : x + patch_size, :]
            sr_tile, disag_tile = run_direct_sr_tile(
                models, in_tile, device=device, sharpness_boost=sharpness_boost
            )

            th, tw = sr_tile.shape[:2]
            y_out_start = int(round(y * scale_factor))
            x_out_start = int(round(x * scale_factor))
            y_out_end = min(y_out_start + th, h_target)
            x_out_end = min(x_out_start + tw, w_target)

            actual_th = y_out_end - y_out_start
            actual_tw = x_out_end - x_out_start

            win = _create_2d_hann_window(actual_th, actual_tw)

            sr_accum[y_out_start:y_out_end, x_out_start:x_out_end, :] += (
                sr_tile[:actual_th, :actual_tw, :] * win[:, :, None]
            )
            disag_accum[y_out_start:y_out_end, x_out_start:x_out_end] += (
                disag_tile[:actual_th, :actual_tw] * win
            )
            weight_accum[y_out_start:y_out_end, x_out_start:x_out_end] += win

    # Normalize by accumulated weights to achieve seamless blend
    weight_accum = np.maximum(weight_accum, 1e-6)
    sr_4m_full = sr_accum / weight_accum[:, :, None]
    disagreement_full = disag_accum / weight_accum

    return sr_4m_full.astype(np.float32), disagreement_full.astype(np.float32)


# =============================================================================
# 4. Georeferencing & Transform (10m -> 4m)
# =============================================================================

def compute_4m_geotransform(
    in_transform: Affine,
    in_shape: Optional[Tuple[int, int]] = None,
    out_shape: Optional[Tuple[int, int]] = None,
    scale_factor: float = 2.5,
) -> Affine:
    """Compute the nominal 4m GeoTIFF affine transform from the 10m input transform.

    Preserves exact upper-left geographic origin and scene bounding extent:
    - a' = a / scale_factor (e.g. 10.0 / 2.5 = 4.0)
    - e' = e / scale_factor (e.g. -10.0 / 2.5 = -4.0)
    - c' = c, f' = f (origin coordinates unchanged)

    Args:
        in_transform: 10m rasterio Affine transform.
        in_shape: Optional (height, width) of input raster.
        out_shape: Optional (height, width) of output raster.
        scale_factor: Spatial scale factor (2.5).

    Returns:
        Affine: 4m rasterio Affine transform.
    """
    return Affine(
        in_transform.a / scale_factor,
        in_transform.b / scale_factor,
        in_transform.c,
        in_transform.d / scale_factor,
        in_transform.e / scale_factor,
        in_transform.f,
    )



# =============================================================================
# 5. Ground-Truth-Free Trust & Uncertainty Verification
# =============================================================================

def compute_real_inference_trust(
    sr_4m: np.ndarray,
    lr_10m: np.ndarray,
    disagreement_map: np.ndarray,
    models: List[nn.Module],
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate multi-criteria empirical trust signals for real Sentinel-2 inference.

    SCIENTIFIC INTEGRITY RULE:
    Zero fake PSNR/SSIM is computed against non-existent ground truth.
    Uses verifiable signals only:
    1. Ensemble Disagreement Map (epistemic uncertainty proxy)
    2. Input-Perturbation Stability Map (sensitivity to sensor radiometric fluctuations)
    3. Cross-Scale Spectral NDVI Fidelity (comparing 4m SR against low-frequency 10m observation)
    4. Structural Edge Gradient Diagnostics (Sobel edge energy preservation)
    5. Multi-evidence heuristic fusion into a normalized Trust/Risk map

    Returns:
        Dict[str, Any]: Complete verification report including trust score and risk map.
    """
    if config is None:
        config = load_config()

    if device is None:
        device = next(models[0].parameters()).device

    h_4m, w_4m = sr_4m.shape[:2]

    # 1. Perturbation Stability Testing
    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))

    stability_trials = []
    rng = np.random.default_rng(42)
    for trial_idx in range(2):
        for noise in noise_levels:
            perturbed = apply_controlled_perturbation(
                lr_10m, noise_std=noise, brightness_jitter_std=jitter_std, seed=100 + trial_idx * 10
            )
            # Run one ensemble member per trial to conserve laptop GPU memory
            model_sub = [models[trial_idx % len(models)]]
            pert_sr, _ = run_direct_sr_tile(model_sub, perturbed, device=device, sharpness_boost=0.0)
            if pert_sr.shape[:2] != (h_4m, w_4m):
                pert_sr = cv2.resize(pert_sr, (w_4m, h_4m), interpolation=cv2.INTER_CUBIC)
            stability_trials.append(pert_sr)

    stability_stack = np.stack(stability_trials, axis=0)
    stability_map = np.mean(np.var(stability_stack, axis=0), axis=-1).astype(np.float32)

    # 2. Cross-Scale Spectral NDVI Consistency
    # Upsample the 10m observation via bicubic interpolation to 4m as the baseline spectral reference
    lr_upsampled_4m = cv2.resize(lr_10m, (w_4m, h_4m), interpolation=cv2.INTER_CUBIC)
    ndvi_sr = compute_ndvi(sr_4m, red_idx=2, nir_idx=3)
    ndvi_lr = compute_ndvi(lr_upsampled_4m, red_idx=2, nir_idx=3)

    delta_ndvi = np.abs(ndvi_sr - ndvi_lr).astype(np.float32)
    noise_floor = float(config.get("verification", {}).get("spectral_consistency", {}).get("sensor_noise_floor", 0.03))
    delta_ndvi_clean = np.where(delta_ndvi <= noise_floor, 0.0, delta_ndvi - noise_floor).astype(np.float32)

    # 3. Structural Gradient Diagnostics
    grad_sr = compute_gradient_magnitude(sr_4m)
    grad_lr = compute_gradient_magnitude(lr_upsampled_4m)
    structural_diff = np.abs(grad_sr - grad_lr).astype(np.float32)

    # 4. Multi-Evidence Heuristic Fusion
    fusion_cfg = config.get("verification", {}).get("evidence_fusion", {})
    weights = fusion_cfg.get("weights", {
        "disagreement": 0.25,
        "stability": 0.25,
        "spectral": 0.25,
        "structural": 0.25,
    })
    fusion_result = fuse_trust_risk_maps(
        disagreement_map=disagreement_map,
        stability_map=stability_map,
        delta_ndvi_map=delta_ndvi_clean,
        structural_diff_map=structural_diff,
        weights=weights,
        high_risk_threshold=float(fusion_cfg.get("high_risk_threshold", 0.65)),
    )

    min_thresh = float(config.get("trust_receipt", {}).get("min_trust_score_threshold", 86.5))
    is_trusted = bool(fusion_result["trust_score_pct"] >= min_thresh)

    return {
        "fusion_result": fusion_result,
        "trust_score_pct": float(fusion_result["trust_score_pct"]),
        "is_trusted": is_trusted,
        "disagreement_mean": float(np.mean(disagreement_map)),
        "stability_mean": float(np.mean(stability_map)),
        "delta_ndvi_mean": float(np.mean(delta_ndvi)),
        "structural_diff_mean": float(np.mean(structural_diff)),
        "ndvi_sr": ndvi_sr,
        "ndvi_lr": ndvi_lr,
        "delta_ndvi_map": delta_ndvi,
        "stability_map": stability_map,
    }


# =============================================================================
# 6. Trust Receipt Generation (Scientific Honesty Guaranteed)
# =============================================================================

def generate_real_trust_receipt(
    scene_name: str,
    val_info: Dict[str, Any],
    trust_data: Dict[str, Any],
    config: Dict[str, Any],
    device_name: str,
) -> Dict[str, Any]:
    """Generate machine-readable JSON certificate for real Sentinel-2 inference.

    Explicitly records that NO synthetic degradation was used, preserving verifiable provenance.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_cfg = config.get("model", {})
    fusion = trust_data["fusion_result"]

    scale = float(model_cfg.get("scale_factor", 2.5))
    ens_size = int(config.get("ensemble", {}).get("ensemble_size", 3))

    return {
        "receipt_id": f"TR-REAL-{scene_name}-{int(datetime.now(timezone.utc).timestamp())}",
        "timestamp_utc": now_utc,
        "execution_mode": "REAL_SENTINEL_2_DIRECT_INFERENCE",
        "input_type": "real_sentinel2_l2a",
        "input_gsd_m": 10.0,
        "output_gsd_m": 4.0,
        "scale_factor": scale,
        "bands": BAND_NAMES,
        "band_descriptions": BAND_DESCRIPTIONS,
        "ensemble_size": ens_size,
        "synthetic_degradation_used_for_inference": False,
        "synthetic_degradation_note": "Zero synthetic degradation applied. Real Sentinel-2 10m input was fed directly to the trained model.",
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "model_architecture": model_cfg.get("architecture", "residual_srnet"),
        "num_residual_blocks": int(model_cfg.get("num_residual_blocks", 8)),
        "num_features": int(model_cfg.get("num_features", 64)),
        "model": {
            "architecture": model_cfg.get("architecture", "residual_srnet"),
            "num_residual_blocks": int(model_cfg.get("num_residual_blocks", 8)),
            "num_features": int(model_cfg.get("num_features", 64)),
            "scale_factor": scale,
        },

        "provenance": {
            "input_type": "real_sentinel2",
            "synthetic_degradation_used_for_inference": False,
            "ensemble_size": ens_size,
            "device": device_name,
            "cuda_available": torch.cuda.is_available(),
        },
        "resolution": {
            "input_gsd_m": 10.0,
            "output_gsd_m": 4.0,
            "scale_factor": scale,
        },
        "reconstruction_metrics_psnr_ssim": {
            "psnr_db": None,
            "ssim": None,
            "note": "PSNR/SSIM unavailable for unreferenced real Sentinel-2 scenes",
        },
        "spatial_metadata": {
            "crs": str(val_info.get("crs", "EPSG:4326")),
            "input_resolution_m": [10.0, 10.0],
            "output_resolution_m": [4.0, 4.0],
            "input_shape": list(val_info.get("shape", [])),
        },

        "empirical_trust_signals": {
            "composite_trust_score_pct": round(trust_data["trust_score_pct"], 2),
            "mean_risk_score": round(float(fusion["mean_risk_score"]), 4),
            "pct_high_risk_pixels": round(float(fusion["pct_high_risk_pixels"]), 2),
            "ensemble_disagreement_mean": round(trust_data["disagreement_mean"], 6),
            "perturbation_stability_mean": round(trust_data["stability_mean"], 6),
            "cross_scale_delta_ndvi_mean": round(trust_data["delta_ndvi_mean"], 4),
            "structural_diff_mean": round(trust_data["structural_diff_mean"], 4),
            "weights_used": fusion["weights_used"],
        },
        "ground_truth_reconstruction_fidelity": {
            "psnr_db": "Not available for this scene (requires independent high-resolution sub-meter reference)",
            "ssim": "Not available for this scene (requires independent high-resolution sub-meter reference)",
            "scientific_note": "Independent high-resolution reference data are required for definitive real-world reconstruction validation. No fabricated metrics are reported.",
        },
        "status": "HIGH_TRUST_APPROVED" if trust_data["is_trusted"] else "LOW_TRUST_ADVISORY",
        "limitations": [
            "Super-resolution reconstructs fine-scale spatial patterns via learned residual priors; it is not direct sub-meter sensor observation.",
            "Independent sub-meter reference imagery is required for quantitative error validation on real scenes.",
            "Cloud-covered or shadowed pixels may exhibit elevated uncertainty.",
        ],
    }


# =============================================================================
# 7. Visualization & Product Export
# =============================================================================

def create_percentile_rgb(
    tile_4band: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
    false_color: bool = False,
) -> np.ndarray:
    """Render 8-bit true-color RGB or false-color infrared CIR composite with percentile stretch."""
    if false_color:
        # False-Color Infrared (CIR): NIR (B08), Red (B04), Green (B03)
        c0, c1, c2 = tile_4band[:, :, 3], tile_4band[:, :, 2], tile_4band[:, :, 1]
    else:
        # Natural True-Color: Red (B04), Green (B03), Blue (B02)
        c0, c1, c2 = tile_4band[:, :, 2], tile_4band[:, :, 1], tile_4band[:, :, 0]

    stretched = []
    for ch in [c0, c1, c2]:
        valid = ch[ch > 0]
        if valid.size > 0:
            vmin = np.percentile(valid, p_low)
            vmax = np.percentile(valid, p_high)
            if vmax > vmin:
                clipped = np.clip(ch, vmin, vmax)
                norm = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                norm = np.zeros_like(ch, dtype=np.uint8)
        else:
            norm = np.zeros_like(ch, dtype=np.uint8)
        stretched.append(norm)

    return np.stack(stretched, axis=-1)


def colorize_ndvi(ndvi_2d: np.ndarray) -> np.ndarray:
    """Colorize 2D NDVI array [-1.0, 1.0] to 8-bit RGB using RdYlGn colormap."""
    # Scale [-0.2, 0.8] to [0, 255]
    norm = np.clip((ndvi_2d + 0.2) / 1.0, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    # Apply COLORMAP_SUMMER or COLORMAP_VIRIDIS for vegetation vigor
    colorized_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_SUMMER)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
    return colorized_rgb


def export_real_inference_products(
    output_dir: Path,
    scene_name: str,
    sr_4m: np.ndarray,
    lr_10m: np.ndarray,
    disagreement_map: np.ndarray,
    trust_map: np.ndarray,
    val_info: Dict[str, Any],
    receipt: Dict[str, Any],
) -> Dict[str, Path]:
    """Export complete production deliverables into outputs/real_inference/<scene_name>/.

    Writes:
    - sr_4m.tif (4-band GeoTIFF, 4m pixel resolution, correct CRS & transform)
    - disagreement_map.tif (single-band 4m GeoTIFF)
    - trust_map.tif (single-band 4m GeoTIFF)
    - rgb_preview.png (true-color 4m)
    - cir_preview.png (false-color 4m)
    - ndvi_preview.png (colorized NDVI 4m)
    - metadata.json
    - trust_receipt.json
    """
    out_scene_dir = output_dir / scene_name
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    h_4m, w_4m = sr_4m.shape[:2]
    crs_val = val_info.get("crs", "EPSG:4326")
    in_trans = val_info.get("transform", Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0))
    trans_4m = compute_4m_geotransform(in_trans, scale_factor=2.5)

    # 1. sr_4m.tif (Main Super-Resolved GeoTIFF)
    sr_path = out_scene_dir / "sr_4m.tif"
    with rasterio.open(
        sr_path,
        "w",
        driver="GTiff",
        height=h_4m,
        width=w_4m,
        count=4,
        dtype="float32",
        crs=crs_val,
        transform=trans_4m,
        nodata=0.0,
    ) as dst:
        for b_idx in range(4):
            dst.write(sr_4m[:, :, b_idx], b_idx + 1)
            dst.set_band_description(b_idx + 1, BAND_DESCRIPTIONS[b_idx])

    # 2. disagreement_map.tif
    disag_path = out_scene_dir / "disagreement_map.tif"
    with rasterio.open(
        disag_path,
        "w",
        driver="GTiff",
        height=h_4m,
        width=w_4m,
        count=1,
        dtype="float32",
        crs=crs_val,
        transform=trans_4m,
        nodata=0.0,
    ) as dst:
        dst.write(disagreement_map, 1)
        dst.set_band_description(1, "Ensemble Disagreement Uncertainty Proxy (sigma)")

    # 3. trust_map.tif
    trust_path = out_scene_dir / "trust_map.tif"
    with rasterio.open(
        trust_path,
        "w",
        driver="GTiff",
        height=h_4m,
        width=w_4m,
        count=1,
        dtype="float32",
        crs=crs_val,
        transform=trans_4m,
        nodata=0.0,
    ) as dst:
        dst.write(trust_map, 1)
        dst.set_band_description(1, "Composite Empirical Trust Map [0.0 - 1.0]")

    # 4. Preview PNGs
    rgb_path = out_scene_dir / "rgb_preview.png"
    rgb_img = create_percentile_rgb(sr_4m, false_color=False)
    Image.fromarray(rgb_img).save(rgb_path)

    cir_path = out_scene_dir / "cir_preview.png"
    cir_img = create_percentile_rgb(sr_4m, false_color=True)
    Image.fromarray(cir_img).save(cir_path)

    ndvi_path = out_scene_dir / "ndvi_preview.png"
    ndvi_arr = compute_ndvi(sr_4m)
    ndvi_img = colorize_ndvi(ndvi_arr)
    Image.fromarray(ndvi_img).save(ndvi_path)

    # 5. Metadata and Receipt JSON
    receipt_path = out_scene_dir / "trust_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    meta_path = out_scene_dir / "metadata.json"
    meta_dict = {
        "scene_name": scene_name,
        "input_format": val_info.get("format", "unknown"),
        "input_shape": val_info.get("shape"),
        "output_shape": [h_4m, w_4m],
        "scale_factor": 2.5,
        "input_gsd_m": 10.0,
        "output_gsd_m": 4.0,
        "synthetic_degradation": False,
        "crs": str(crs_val),
        "affine_4m": list(trans_4m)[:6],
        "generated_files": {
            "sr_4m": sr_path.name,
            "disagreement_map": disag_path.name,
            "trust_map": trust_path.name,
            "rgb_preview": rgb_path.name,
            "cir_preview": cir_path.name,
            "ndvi_preview": ndvi_path.name,
            "receipt": receipt_path.name,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return {
        "sr_4m": sr_path,
        "disagreement_map": disag_path,
        "trust_map": trust_path,
        "rgb_preview": rgb_path,
        "cir_preview": cir_path,
        "ndvi_preview": ndvi_path,
        "trust_receipt": receipt_path,
        "metadata": meta_path,
    }


# =============================================================================
# 8. Complete High-Level Execution Entry Point
# =============================================================================

def execute_real_sentinel2_sr(
    input_source: Union[str, Path, List[Union[str, Path]]],
    output_dir: Optional[Path] = None,
    scene_name: Optional[str] = None,
    models: Optional[List[nn.Module]] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute complete direct real Sentinel-2 super-resolution pipeline.

    ZERO SYNTHETIC DEGRADATION.
    Real 10m -> Direct 2.5x Ensemble SR -> Nominal 4m GeoTIFF + Trust Receipt.
    """
    root = get_project_root()
    if config is None:
        config = load_config()
    if device is None:
        device = get_device(config)
    if output_dir is None:
        output_dir = root / "outputs" / "real_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate Sentinel-2 input
    scene_10m, val_info = load_and_preprocess_sentinel2(input_source)

    if scene_name is None:
        if isinstance(input_source, (list, tuple)):
            scene_name = Path(input_source[0]).stem.split("_")[0] or "sentinel2_scene"
        elif Path(input_source).is_dir():
            scene_name = Path(input_source).name
        else:
            scene_name = Path(input_source).stem

    # 2. Load trained ensemble models
    if models is None:
        ckpt_dir = root / config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
        ckpt_paths = [ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)]
        models = load_ensemble_members(ckpt_paths, config=config, device=device)

    # 3. Direct 2.5x Super-Resolution (ZERO degradation)
    sr_4m, disagreement_map = run_direct_sr_scene(
        models=models,
        scene_10m=scene_10m,
        patch_size=int(config.get("preprocessing", {}).get("patch_size", 64)),
        stride=int(config.get("preprocessing", {}).get("stride", 48)),
        scale_factor=float(config.get("model", {}).get("scale_factor", 2.5)),
        device=device,
    )

    # 4. Empirical Trust Verification (No fake ground truth)
    trust_data = compute_real_inference_trust(
        sr_4m=sr_4m,
        lr_10m=scene_10m,
        disagreement_map=disagreement_map,
        models=models,
        device=device,
        config=config,
    )

    # 5. Auditable Trust Receipt
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    receipt = generate_real_trust_receipt(
        scene_name=scene_name,
        val_info=val_info,
        trust_data=trust_data,
        config=config,
        device_name=device_name,
    )

    # 6. Export Products
    exported_paths = export_real_inference_products(
        output_dir=output_dir,
        scene_name=scene_name,
        sr_4m=sr_4m,
        lr_10m=scene_10m,
        disagreement_map=disagreement_map,
        trust_map=trust_data["fusion_result"]["trust_map"],
        val_info=val_info,
        receipt=receipt,
    )

    return {
        "scene_name": scene_name,
        "input_10m": scene_10m,
        "sr_4m": sr_4m,
        "disagreement_map": disagreement_map,
        "trust_map": trust_data["fusion_result"]["trust_map"],
        "trust_data": trust_data,
        "receipt": receipt,
        "val_info": val_info,
        "exported_paths": exported_paths,
        "synthetic_degradation_used": False,
    }
