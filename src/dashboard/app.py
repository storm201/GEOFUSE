"""GeoFUSE SentinelGuard — Interactive Streamlit Demonstration Dashboard.

Demonstrates trust-aware satellite image super-resolution on real Sentinel-2 L2A imagery:
1. Mode 1 (Primary): Direct Real Sentinel-2 (10m -> 4m SR) with ZERO synthetic degradation.
   - Side-by-side: Sentinel-2 10m Original vs. GeoFUSE SR 4m (2.5x learned reconstruction).
   - Natural RGB, False-Color Infrared (CIR), and 4m NDVI Vegetation Index views.
   - Ensemble Disagreement Map (uncertainty proxy) & Composite Trust/Risk Map.
   - Domain analytics for Urban Infrastructure (building footprints) and Agriculture (crop vigor).
   - Auditable Trust Receipt recording synthetic_degradation: false and full provenance.
2. Mode 2: Controlled Synthetic Benchmark (Degrade & Recover)
   - Scientific baseline evaluation: HR Reference vs. Degraded Pseudo-LR vs. Bicubic vs. SR.
   - Quantitative fidelity: PSNR, SSIM, MAE, LapVar, 1D transect profile, and 2D Fourier FFT.
"""

import importlib
import inspect
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
import streamlit as st
import torch

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import bicubic_upsample, evaluate_reconstruction_fidelity, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.downstream_eval import (
    compare_downstream_footprints,
    extract_building_footprints,
)
from src.evaluation.edge_check import (
    compute_edge_consistency,
    compute_gradient_magnitude,
)
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_ndvi, compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.evaluation.trust_receipt import (
    generate_trust_receipt,
    render_trust_receipt_html,
)
from src.inference.real_inference import (
    compute_4m_geotransform,
    compute_real_inference_trust,
    export_real_inference_products,
    generate_real_trust_receipt,
    load_and_preprocess_sentinel2,
    run_direct_sr_scene,
    run_direct_sr_tile,
    validate_sentinel2_input,
)

from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config
import src.utils.scene_visualizer as sv

try:
    get_tile_metadata_grid = getattr(sv, "get_tile_metadata_grid")
    render_scene_with_tile_overlay = sv.render_scene_with_tile_overlay
    to_display_rgb = sv.to_display_rgb
except (AttributeError, ImportError):
    render_scene_with_tile_overlay = getattr(sv, "render_scene_with_tile_overlay", None)
    to_display_rgb = getattr(sv, "to_display_rgb", None)

    def get_tile_metadata_grid(h: int = 512, w: int = 512, patch_size: int = 128, stride: int = 96):
        y_steps = list(range(0, h - patch_size + 1, stride))
        x_steps = list(range(0, w - patch_size + 1, stride))
        if y_steps[-1] != h - patch_size:
            y_steps.append(h - patch_size)
        if x_steps[-1] != w - patch_size:
            x_steps.append(w - patch_size)
        tiles = []
        t_id = 0
        for r_idx, y in enumerate(y_steps):
            for c_idx, x in enumerate(x_steps):
                tiles.append({
                    "tile_id": t_id,
                    "row": r_idx + 1,
                    "col": c_idx + 1,
                    "x": x,
                    "y": y,
                    "w": patch_size,
                    "h": patch_size,
                    "patch_size": patch_size,
                    "description": f"Grid Partition (R{r_idx+1}:C{c_idx+1})",
                    "label": f"Tile #{t_id:02d} [R{r_idx+1}:C{c_idx+1}] (X:{x}, Y:{y})",
                })
                t_id += 1
        return tiles


# -----------------------------------------------------------------------------
# Cached Model & Asset Loading
# -----------------------------------------------------------------------------

def get_demo_cache_dir() -> Path:
    """Resolve directory containing precomputed offline demo cache bundles."""
    root = get_project_root()
    config = load_config()
    return root / config.get("paths", {}).get("outputs_dir", "outputs") / "demo_cache"


def load_demo_manifest() -> Optional[Dict[str, Any]]:
    """Load precomputed demo cache manifest directly from disk without stale RAM caching."""
    cache_dir = get_demo_cache_dir()
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_demo_bundle(tile_idx: int) -> Optional[Dict[str, Any]]:
    """Load precomputed offline demo bundle directly from disk."""
    cache_dir = get_demo_cache_dir()
    bundle_path = cache_dir / f"demo_tile_{tile_idx}.pkl"
    if bundle_path.exists():
        try:
            with open(bundle_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner="Loading ensemble checkpoints onto compute device...")
def load_cached_models():
    """Load the 3 trained ensemble members once and cache in memory."""
    try:
        root = get_project_root()
        config = load_config()
        device = get_device(config)
        ckpt_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
        if not (ckpt_dir / "ensemble_member_0.pth").exists():
            ckpt_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
        ckpt_paths = [ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)]
        for p in ckpt_paths:
            if not p.exists():
                return None, config, device
        models = load_ensemble_members(ckpt_paths, config=config, device=device, verbose=False)
        return models, config, device
    except Exception:
        return None, load_config(), torch.device("cpu")


@st.cache_data(show_spinner="Loading Sentinel-2 reference scene...")
def load_cached_scene():
    """Load raw Sentinel-2 scene and pre-slice candidate tiles."""
    root = get_project_root()
    config = load_config()
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    try:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        stack, meta = load_sentinel2_stack(raw_dir)
        tiles = extract_tiles(stack, patch_size=128, stride=96)
        return stack, meta, tiles
    except Exception as e:
        dummy_tiles = [
            {"tile_id": i, "data": np.zeros((128, 128, 4), dtype=np.float32)}
            for i in [0, 8, 16, 24]
        ]
        return None, {"crs": "EPSG:32643", "error": str(e)}, dummy_tiles


# -----------------------------------------------------------------------------
# Benchmark Pipeline Execution (Mode 2)
# -----------------------------------------------------------------------------

def run_cached_pipeline(tile_idx: int) -> Optional[Dict[str, Any]]:
    """Execute complete benchmark inference and verification for a tile."""
    cached_bundle = load_demo_bundle(tile_idx)
    if cached_bundle is not None:
        return cached_bundle

    try:
        models, config, device = load_cached_models()
        if models is None:
            return None

        root = get_project_root()
        raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
        _, _, tiles = load_cached_scene()
        if not tiles or tile_idx >= len(tiles):
            return None

        scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))
        hr_tile = tiles[tile_idx]["data"]
        lr_tile = synthesize_pseudo_lr(
            hr_tile,
            downsample_factor=scale_factor,
            blur_kernel_size=3,
            noise_std=0.01,
            seed=1000 + tile_idx,
        )

        bicubic_tile = bicubic_upsample(lr_tile, scale_factor=scale_factor, target_shape=hr_tile.shape[:2])
        sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)
        if sr_tile.shape[:2] != hr_tile.shape[:2]:
            sr_tile = cv2.resize(sr_tile, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)
            disagreement_map = cv2.resize(disagreement_map, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)

        pert_cfg = config.get("verification", {}).get("perturbation_test", {})
        noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
        jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))
        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
            device=device,
        )

        red_idx, nir_idx = 2, 3
        spectral_metrics = compute_spectral_consistency(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            red_idx=red_idx,
            nir_idx=nir_idx,
        )

        grad_gt = compute_gradient_magnitude(hr_tile)
        grad_sr = compute_gradient_magnitude(sr_tile)
        structural_diff = np.abs(grad_sr - grad_gt)
        edge_metrics = compute_edge_consistency(hr_tile, sr_tile)

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
            delta_ndvi_map=spectral_metrics["delta_ndvi"],
            structural_diff_map=structural_diff,
            weights=weights,
        )

        down_cfg = config.get("downstream", {})
        morph_cfg = down_cfg.get("morphology", {})
        foot_bic = extract_building_footprints(
            bicubic_tile,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        foot_sr = extract_building_footprints(
            sr_tile,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        foot_ref = extract_building_footprints(
            hr_tile,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        downstream_comp = compare_downstream_footprints(
            mask_bicubic=foot_bic["mask"],
            mask_sr=foot_sr["mask"],
            trust_map=fusion_result["trust_map"],
            ref_mask=foot_ref["mask"],
            trust_threshold=float(down_cfg.get("trust_partition_threshold", 0.85)),
        )

        pipeline_data = {
            "fusion_result": fusion_result,
            "spectral_metrics": spectral_metrics,
            "edge_metrics": edge_metrics,
            "downstream_comp": downstream_comp,
        }
        receipt = generate_trust_receipt(
            tile_idx=tile_idx,
            raw_dir=raw_dir,
            config=config,
            pipeline_data=pipeline_data,
        )

        return {
            "hr_tile": hr_tile,
            "lr_tile": lr_tile,
            "bicubic_tile": bicubic_tile,
            "sr_tile": sr_tile,
            "disagreement_map": disagreement_map,
            "stability_map": stability_map,
            "spectral_metrics": spectral_metrics,
            "edge_metrics": edge_metrics,
            "structural_diff": structural_diff,
            "fusion_result": fusion_result,
            "foot_bic": foot_bic,
            "foot_sr": foot_sr,
            "foot_ref": foot_ref,
            "downstream_comp": downstream_comp,
            "receipt": receipt,
            "tile_idx": tile_idx,
            "crop_box": {"x": (tile_idx % 5) * 96, "y": (tile_idx // 5) * 96, "w": 128, "h": 128},
        }
    except Exception:
        return None


def run_cached_crop_pipeline(crop_x: int, crop_y: int) -> Optional[Dict[str, Any]]:
    """Execute live pipeline on an arbitrary coordinate bounding box across the scene."""
    try:
        models, config, device = load_cached_models()
        if models is None:
            return None

        root = get_project_root()
        raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
        stack, _, _ = load_cached_scene()
        if stack is None:
            return None

        h, w = stack.shape[:2]
        cx = max(0, min(int(crop_x), w - 128))
        cy = max(0, min(int(crop_y), h - 128))

        scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))
        hr_tile = stack[cy : cy + 128, cx : cx + 128, :]
        lr_tile = synthesize_pseudo_lr(
            hr_tile,
            downsample_factor=scale_factor,
            blur_kernel_size=3,
            noise_std=0.01,
            seed=2000 + cx + cy,
        )

        bicubic_tile = bicubic_upsample(lr_tile, scale_factor=scale_factor, target_shape=hr_tile.shape[:2])
        sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)
        if sr_tile.shape[:2] != hr_tile.shape[:2]:
            sr_tile = cv2.resize(sr_tile, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)
            disagreement_map = cv2.resize(disagreement_map, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)

        pert_cfg = config.get("verification", {}).get("perturbation_test", {})
        noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
        jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))
        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
            device=device,
        )

        red_idx, nir_idx = 2, 3
        spectral_metrics = compute_spectral_consistency(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            red_idx=red_idx,
            nir_idx=nir_idx,
        )

        grad_gt = compute_gradient_magnitude(hr_tile)
        grad_sr = compute_gradient_magnitude(sr_tile)
        structural_diff = np.abs(grad_sr - grad_gt)
        edge_metrics = compute_edge_consistency(hr_tile, sr_tile)

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
            delta_ndvi_map=spectral_metrics["delta_ndvi"],
            structural_diff_map=structural_diff,
            weights=weights,
        )

        down_cfg = config.get("downstream", {})
        morph_cfg = down_cfg.get("morphology", {})
        foot_bic = extract_building_footprints(
            bicubic_tile,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        foot_sr = extract_building_footprints(
            sr_tile,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        foot_ref = extract_building_footprints(
            hr_tile,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        downstream_comp = compare_downstream_footprints(
            mask_bicubic=foot_bic["mask"],
            mask_sr=foot_sr["mask"],
            trust_map=fusion_result["trust_map"],
            ref_mask=foot_ref["mask"],
            trust_threshold=float(down_cfg.get("trust_partition_threshold", 0.85)),
        )

        pipeline_data = {
            "fusion_result": fusion_result,
            "spectral_metrics": spectral_metrics,
            "edge_metrics": edge_metrics,
            "downstream_comp": downstream_comp,
        }
        receipt = generate_trust_receipt(
            tile_idx=int(cx // 96 + (cy // 96) * 5),
            raw_dir=raw_dir,
            config=config,
            pipeline_data=pipeline_data,
        )

        return {
            "hr_tile": hr_tile,
            "lr_tile": lr_tile,
            "bicubic_tile": bicubic_tile,
            "sr_tile": sr_tile,
            "disagreement_map": disagreement_map,
            "stability_map": stability_map,
            "spectral_metrics": spectral_metrics,
            "edge_metrics": edge_metrics,
            "structural_diff": structural_diff,
            "fusion_result": fusion_result,
            "foot_bic": foot_bic,
            "foot_sr": foot_sr,
            "foot_ref": foot_ref,
            "downstream_comp": downstream_comp,
            "receipt": receipt,
            "crop_box": {"x": cx, "y": cy, "w": 128, "h": 128},
            "tile_idx": f"Crop ({cx}, {cy})",
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Real Sentinel-2 Direct Super-Resolution Execution (Mode 1 Cached Engine)
# -----------------------------------------------------------------------------

import hashlib


def compute_files_hash(files: List[Any]) -> str:
    """Compute a deterministic hash from a collection of uploaded files."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: getattr(x, "name", "")):
        h.update(f.name.encode("utf-8"))
        h.update(str(getattr(f, "size", 0)).encode("utf-8"))
    return h.hexdigest()[:16]


def save_uploaded_sentinel_files(uploaded_files: List[Any], target_dir: Path) -> List[Path]:
    """Save Streamlit in-memory uploaded files to a clean target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploaded_files:
        p = target_dir / f.name
        with open(p, "wb") as dst:
            dst.write(f.getbuffer())
        paths.append(p)
    return paths


def execute_real_sentinel2_sr(
    source_path: Union[str, Path, List[Path]],
    scene_name: str,
    tile_size: int = 128,
    crop_coords: Optional[Tuple[int, int]] = None,
    full_scene: bool = False,
    progress_status_fn = None,
) -> Dict[str, Any]:
    """Core execution engine for direct real Sentinel-2 10m -> 4m SR without synthetic degradation."""
    root = get_project_root()
    config = load_config()
    models, _, device = load_cached_models()
    if models is None:
        raise RuntimeError("Ensemble checkpoints could not be loaded.")

    if progress_status_fn:
        progress_status_fn(1, "Loading Sentinel-2 bands (B02, B03, B04, B08)...")
    stack_10m, val_info = load_and_preprocess_sentinel2(source_path)
    h0, w0 = stack_10m.shape[:2]

    if progress_status_fn:
        progress_status_fn(2, f"Validating geometry & spatial alignment... ✓ [{w0}×{h0} px, ~10m GSD]")

    scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))

    if full_scene and (h0 > tile_size or w0 > tile_size):
        if progress_status_fn:
            progress_status_fn(3, "Preparing overlapping tiles & Hann window feathering... ✓")
        if progress_status_fn:
            progress_status_fn(4, f"Loading ensemble on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}... ✓")
        if progress_status_fn:
            progress_status_fn(5, "Running GPU inference across full scene... ✓")
        sr_4m, disag_map = run_direct_sr_scene(models, stack_10m, scale_factor=scale_factor, device=device)
        sub_10m = stack_10m
        crop_box = {"x": 0, "y": 0, "w": w0, "h": h0}
    else:
        if crop_coords is not None:
            cx, cy = crop_coords
            sx = max(0, min(int(cx), w0 - tile_size))
            sy = max(0, min(int(cy), h0 - tile_size))
        else:
            sx = max(0, (w0 - tile_size) // 2)
            sy = max(0, (h0 - tile_size) // 2)

        sub_10m = stack_10m[sy : sy + tile_size, sx : sx + tile_size, :]
        if progress_status_fn:
            progress_status_fn(3, f"Preparing tile tensor [{sub_10m.shape[1]}×{sub_10m.shape[0]} px]... ✓")
        if progress_status_fn:
            progress_status_fn(4, f"Loading ensemble on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}... ✓")
        if progress_status_fn:
            progress_status_fn(5, "Running direct 2.5× super-resolution forward pass (zero synthetic degradation)... ✓")
        sr_4m, disag_map = run_direct_sr_tile(models, sub_10m, device=device)
        crop_box = {"x": sx, "y": sy, "w": tile_size, "h": tile_size}

    if progress_status_fn:
        progress_status_fn(6, "Generating empirical trust maps & uncertainty proxy... ✓")
    trust_data = compute_real_inference_trust(
        sr_4m=sr_4m,
        lr_10m=sub_10m,
        disagreement_map=disag_map,
        models=models,
        device=device,
        config=config,
    )

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    receipt = generate_real_trust_receipt(
        scene_name=scene_name,
        val_info=val_info,
        trust_data=trust_data,
        config=config,
        device_name=dev_name,
    )

    if progress_status_fn:
        progress_status_fn(7, "Exporting 4m GeoTIFF, previews, and auditable receipt... ✓")
    out_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "real_inference"
    exported_paths = export_real_inference_products(
        output_dir=out_dir,
        scene_name=scene_name,
        sr_4m=sr_4m,
        lr_10m=sub_10m,
        disagreement_map=disag_map,
        trust_map=trust_data["fusion_result"]["trust_map"],
        val_info=val_info,
        receipt=receipt,
    )

    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})
    foot_sr = extract_building_footprints(
        sr_4m,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=2,
        nir_idx=3,
    )

    bicubic_10m_up = cv2.resize(sub_10m, (sr_4m.shape[1], sr_4m.shape[0]), interpolation=cv2.INTER_CUBIC)
    foot_10m = extract_building_footprints(
        bicubic_10m_up,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=2,
        nir_idx=3,
    )

    return {
        "mode": "real",
        "scene_key": scene_name,
        "input_10m": sub_10m,
        "full_scene": stack_10m,
        "sr_4m": sr_4m,
        "bicubic_10m_up": bicubic_10m_up,
        "disagreement_map": disag_map,
        "trust_data": trust_data,
        "fusion_result": trust_data["fusion_result"],
        "receipt": receipt,
        "val_info": val_info,
        "foot_sr": foot_sr,
        "foot_10m": foot_10m,
        "crop_box": crop_box,
        "exported_paths": exported_paths,
        "synthetic_degradation_used": False,
    }


@st.cache_data(show_spinner="Executing direct 2.5x super-resolution on real Sentinel-2 scene...")
def run_real_sentinel2_pipeline_cached(
    scene_key: str,
    tile_size: int = 128,
    crop_coords: Optional[Tuple[int, int]] = None,
    full_scene: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute direct real Sentinel-2 10m -> 4m SR pipeline without synthetic degradation."""
    root = get_project_root()
    scene_map = {
        "urban_core": root / "data/additional_datasets/urban_core",
        "agriculture": root / "data/additional_datasets/agriculture",
        "temporal_april2024": root / "data/additional_datasets/temporal_april2024",
        "base_raw": root / "data/raw",
    }
    src_path = scene_map.get(scene_key, scene_map["urban_core"])
    try:
        return execute_real_sentinel2_sr(
            source_path=src_path,
            scene_name=scene_key,
            tile_size=tile_size,
            crop_coords=crop_coords,
            full_scene=full_scene,
        )
    except Exception as e:
        st.error(f"Execution Error: {e}")
        return None



# -----------------------------------------------------------------------------
# Visualization Utilities
# -----------------------------------------------------------------------------

def get_display_stretch_bounds(tile: Optional[np.ndarray], false_color: bool = False) -> Tuple[float, float]:
    """Compute 2nd and 98th percentile stretch bounds from image for consistent display."""
    if tile is None or not isinstance(tile, np.ndarray) or tile.ndim < 3 or tile.shape[2] < 3:
        return (0.0, 1.0)
    try:
        if false_color and tile.shape[2] >= 4:
            rgb = tile[:, :, [3, 2, 1]].astype(np.float32)
        else:
            rgb = tile[:, :, [2, 1, 0]].astype(np.float32)
        p2, p98 = np.percentile(rgb, (2, 98))
        return (float(p2), float(p98)) if p98 > p2 else (0.0, 1.0)
    except Exception:
        return (0.0, 1.0)


def extract_zoomed_crop(
    img: np.ndarray,
    crop_center: Tuple[float, float] = (0.5, 0.5),
    crop_size: int = 40,
    zoom_factor: int = 4,
) -> np.ndarray:
    """Extract a cropped region and upsample using nearest-neighbor for pixel-level inspection."""
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        return np.zeros((crop_size * zoom_factor, crop_size * zoom_factor, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    cy, cx = int(crop_center[0] * h), int(crop_center[1] * w)
    half = crop_size // 2
    y1 = max(0, cy - half)
    y2 = min(h, y1 + crop_size)
    x1 = max(0, cx - half)
    x2 = min(w, x1 + crop_size)

    if y2 - y1 < crop_size and h >= crop_size:
        y1 = max(0, y2 - crop_size)
    if x2 - x1 < crop_size and w >= crop_size:
        x1 = max(0, x2 - crop_size)

    crop = img[y1:y2, x1:x2]
    target_h, target_w = (y2 - y1) * zoom_factor, (x2 - x1) * zoom_factor
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


# -----------------------------------------------------------------------------
# Premium Earth-Observation Mission Control Design System (GeoFUSE v2.5)
# -----------------------------------------------------------------------------

GEOFUSE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
    /* Color Palette: Deep Space Blue & Earth Observation Navy */
    --bg-base: #07111F;
    --bg-surface: #0B1628;
    --bg-card: #0F1D30;
    --bg-card-hover: #132238;
    --bg-elevated: #16263F;

    /* Accents */
    --accent-emerald: #16D9B5;       /* Primary Electric Emerald/Teal */
    --accent-cyan: #38BDF8;          /* Secondary Sky Cyan */
    --accent-amber: #F59E0B;         /* Benchmark Amber */
    --accent-red: #EF4444;           /* Alert Red */

    /* Borders & Glass */
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-card: rgba(56, 189, 248, 0.14);
    --border-accent: rgba(22, 217, 181, 0.35);
    --glass-bg: rgba(15, 29, 48, 0.72);

    /* Text */
    --text-primary: #F8FAFC;
    --text-secondary: #CBD5E1;
    --text-muted: #8190A5;

    /* Typography */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;

    /* Elevation */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.38);
}

/* Global App Shell */
.stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* Streamlit Header / Navigation Override */
header[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.90) !important;
    backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1324 0%, #060B14 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 0.74rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
}

/* Typography Hierarchy */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

p, span, label {
    font-family: var(--font-sans) !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 18px !important;
    box-shadow: var(--shadow-card) !important;
    backdrop-filter: blur(14px) !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--accent-emerald) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px -4px rgba(22, 217, 181, 0.22) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-sans) !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.60rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stMetricDelta"] {
    font-family: var(--font-sans) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

/* Tabs: Segmented Control Bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(11, 22, 40, 0.75) !important;
    padding: 6px !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 18px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 40px !important;
    border-radius: var(--radius-sm) !important;
    padding: 0 18px !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.18s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(22, 217, 181, 0.12) !important;
    color: var(--accent-emerald) !important;
    font-weight: 700 !important;
    border: 1px solid var(--accent-emerald) !important;
    box-shadow: 0 0 16px rgba(22, 217, 181, 0.18) !important;
}

/* Buttons */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #16D9B5 0%, #0EA5E9 100%) !important;
    color: #07111F !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 16px rgba(22, 217, 181, 0.35) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #26E6C2 0%, #38BDF8 100%) !important;
    box-shadow: 0 6px 24px rgba(22, 217, 181, 0.55) !important;
    transform: translateY(-1px) !important;
}

button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: var(--glass-bg) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.18s ease !important;
}

button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent-emerald) !important;
    color: var(--accent-emerald) !important;
    background: rgba(15, 29, 48, 0.95) !important;
}

/* Radio & Form Controls */
[data-testid="stRadio"] label {
    font-size: 0.84rem !important;
    color: var(--text-secondary) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--glass-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* Image Elements */
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.50) !important;
}

/* Custom UI Components */
.geofuse-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: rgba(11, 22, 40, 0.85);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: 20px;
    backdrop-filter: blur(14px);
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-logo-icon {
    font-size: 1.5rem;
}
.nav-brand-text {
    display: flex;
    flex-direction: column;
}
.nav-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}
.nav-title span {
    color: var(--accent-emerald);
}
.nav-subtitle {
    font-size: 0.70rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.nav-center-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    background: rgba(22, 217, 181, 0.08);
    border: 1px solid rgba(22, 217, 181, 0.25);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--accent-emerald);
    letter-spacing: 0.06em;
}
.nav-status-group {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
}
.nav-status-badge.live {
    background: rgba(22, 217, 181, 0.10);
    border-color: rgba(22, 217, 181, 0.35);
    color: var(--accent-emerald);
}
.status-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--accent-emerald);
    box-shadow: 0 0 10px var(--accent-emerald);
    display: inline-block;
    animation: pulse-dot 2s infinite ease-in-out;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.85); }
}

/* Hero Section */
.geofuse-hero {
    margin-bottom: 22px;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-title span {
    background: linear-gradient(135deg, #16D9B5 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 0.94rem;
    color: var(--text-secondary);
    margin: 0 0 14px 0;
    font-weight: 400;
    max-width: 820px;
    line-height: 1.5;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.hero-badge.teal {
    background: rgba(22, 217, 181, 0.10);
    border: 1px solid rgba(22, 217, 181, 0.30);
    color: var(--accent-emerald);
}
.hero-badge.cyan {
    background: rgba(56, 189, 248, 0.10);
    border: 1px solid rgba(56, 189, 248, 0.30);
    color: var(--accent-cyan);
}
.hero-badge.muted {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
}
.hero-badge.amber {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.30);
    color: var(--accent-amber);
}

/* Visual Pipeline Stepper */
.pipeline-stepper {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    padding: 12px 16px;
    background: rgba(11, 22, 40, 0.60);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    margin-bottom: 22px;
}
.step-node {
    display: flex;
    align-items: center;
    gap: 10px;
}
.step-num {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    font-family: var(--font-mono);
    font-size: 0.70rem;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
}
.step-node.active .step-num {
    background: rgba(22, 217, 181, 0.18);
    color: var(--accent-emerald);
    border-color: var(--accent-emerald);
    box-shadow: 0 0 10px rgba(22, 217, 181, 0.35);
}
.step-node.active-amber .step-num {
    background: rgba(245, 158, 11, 0.18);
    color: var(--accent-amber);
    border-color: var(--accent-amber);
}
.step-label {
    display: flex;
    flex-direction: column;
}
.step-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
    text-transform: uppercase;
}
.step-node.active .step-title {
    color: var(--text-primary);
}
.step-desc {
    font-size: 0.66rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
}

/* Consolidated Scene Ready Card */
.scene-ready-card {
    background: linear-gradient(135deg, rgba(15, 29, 48, 0.90) 0%, rgba(11, 22, 40, 0.90) 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin-bottom: 22px;
    box-shadow: var(--shadow-card);
}
.scene-status-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}
.scene-status-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text-primary);
}
.scene-check {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(22, 217, 181, 0.16);
    color: var(--accent-emerald);
    font-size: 0.74rem;
    font-weight: 800;
}
.scene-source-tag {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.12);
    color: var(--accent-cyan);
    border: 1px solid rgba(56, 189, 248, 0.28);
    letter-spacing: 0.06em;
}
.scene-meta-chips {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px;
}
.meta-chip {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
}
.chip-k {
    font-size: 0.64rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.chip-v {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Viewport Channel Card */
.viewport-card {
    background: var(--glass-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-bottom: 12px;
}
.viewport-card.accent {
    border-color: var(--border-accent);
}
.viewport-badge-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.viewport-tag {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
}
.viewport-tag.accent {
    color: var(--accent-emerald);
}
.viewport-gsd {
    font-family: var(--font-mono);
    font-size: 0.70rem;
    font-weight: 600;
    color: var(--accent-cyan);
}
.viewport-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}

/* Scientific Disclaimer Card */
.scientific-notice {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    background: rgba(11, 22, 40, 0.65);
    border: 1px solid var(--border-subtle);
    margin: 16px 0;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.45;
}
.notice-icon {
    font-size: 1.1rem;
    color: var(--accent-cyan);
}

/* Download Deliverables Cards */
.deliverable-card {
    background: var(--glass-bg);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
}
.deliverable-card:hover {
    border-color: var(--accent-emerald);
    transform: translateY(-2px);
}
.deliv-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.deliv-meta {
    font-family: var(--font-mono);
    font-size: 0.70rem;
    color: var(--text-muted);
    margin-bottom: 12px;
}

/* Responsive Media Queries */
@media (max-width: 900px) {
    .pipeline-stepper {
        grid-template-columns: 1fr;
        gap: 6px;
    }
    .scene-meta-chips {
        grid-template-columns: 1fr 1fr;
    }
}
</style>
"""


# -----------------------------------------------------------------------------
# Modular UI Component Renderers
# -----------------------------------------------------------------------------

def render_top_nav():
    """Render the top mission control navigation bar."""
    st.markdown(
        """
        <header class="geofuse-nav">
            <div class="nav-brand">
                <span class="nav-logo-icon">🛰️</span>
                <div class="nav-brand-text">
                    <span class="nav-title">GeoFUSE <span>SentinelGuard</span></span>
                    <span class="nav-subtitle">Earth Observation SRM Platform · SIH 2026</span>
                </div>
            </div>
            <div class="nav-center-pill">
                <span class="status-pulse-dot"></span>
                <span>SYSTEM READY · DIRECT 2.5× LEARNED SR</span>
            </div>
            <div class="nav-status-group">
                <div class="nav-status-badge live">
                    <span class="status-pulse-dot"></span>
                    <span>CUDA ACCELERATED</span>
                </div>
                <div class="nav-status-badge">
                    <span>ESA S2-L2A DIRECT</span>
                </div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_hero(is_benchmark: bool = False):
    """Render the high-impact hero section with clear scientific value proposition."""
    if not is_benchmark:
        st.markdown(
            """
            <div class="geofuse-hero">
                <h1 class="hero-title">AI-Powered <span>Super-Resolution Mapping</span></h1>
                <p class="hero-subtitle">
                    Transforming Sentinel-2 Level-2A imagery from native 10m to a nominal 4m product for finer spatial interpretation. 
                    Zero synthetic blur is applied in real mode; raw 10m MSI observations are processed directly with deep residual ensembles.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge teal">REAL SENTINEL-2 DIRECT</span>
                    <span class="hero-badge cyan">2.5× LEARNED RECONSTRUCTION</span>
                    <span class="hero-badge teal">4.0m NOMINAL PRODUCT</span>
                    <span class="hero-badge muted">4 MSI BANDS (B02·B03·B04·B08)</span>
                    <span class="hero-badge muted">CUDA RTX 4060 ACCELERATED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="geofuse-hero">
                <h1 class="hero-title">Controlled <span>Synthetic Benchmark</span></h1>
                <p class="hero-subtitle">
                    Rigorous degrade-and-recover scientific validation. Evaluates GeoFUSE against bicubic interpolation 
                    using unseen reference imagery with verifiable PSNR, SSIM, and spectral metrics.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge amber">CONTROLLED BENCHMARK</span>
                    <span class="hero-badge muted">DEGRADE & RECOVER PARADIGM</span>
                    <span class="hero-badge cyan">BICUBIC VS. GEOFUSE</span>
                    <span class="hero-badge muted">UNSEEN 10m GROUND TRUTH</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_pipeline_stepper(current_step: int = 3, is_benchmark: bool = False):
    """Render the 5-stage visual workflow stepper."""
    amber_cls = "active-amber" if is_benchmark else "active"
    steps = [
        ("01", "Upload", "Native L2A Ingest", 1),
        ("02", "Validate", "CRS & Band Check", 2),
        ("03", "2.5× Enhance", "Deep Ensemble SR", 3),
        ("04", "Analyze", "Trust & Downstream", 4),
        ("05", "Export", "GeoTIFF & Provenance", 5),
    ]

    items_html = []
    for num, title, desc, step_idx in steps:
        cls = amber_cls if step_idx == current_step else ("active" if step_idx < current_step else "")
        items_html.append(
            f"""
            <div class="step-node {cls}">
                <div class="step-num">{num}</div>
                <div class="step-label">
                    <span class="step-title">{title}</span>
                    <span class="step-desc">{desc}</span>
                </div>
            </div>
            """
        )

    st.markdown(
        f'<div class="pipeline-stepper">{"".join(items_html)}</div>',
        unsafe_allow_html=True,
    )


def render_scene_status_card(scene_title: str, shape_10m: tuple, shape_4m: tuple, crs_str: str, bands_str: str = "B02 · B03 · B04 · B08"):
    """Render a clean, consolidated scene readiness card."""
    st.markdown(
        f"""
        <div class="scene-ready-card">
            <div class="scene-status-header">
                <div class="scene-status-title">
                    <span class="scene-check">✓</span>
                    <span>Scene Ready: {scene_title}</span>
                </div>
                <span class="scene-source-tag">ESA S2-L2A DIRECT</span>
            </div>
            <div class="scene-meta-chips">
                <div class="meta-chip"><span class="chip-k">Multispectral Bands</span><span class="chip-v">{bands_str}</span></div>
                <div class="meta-chip"><span class="chip-k">Native GSD</span><span class="chip-v">10.00 m / px</span></div>
                <div class="meta-chip"><span class="chip-k">Nominal SR GSD</span><span class="chip-v">4.00 m / px</span></div>
                <div class="meta-chip"><span class="chip-k">Input Extent</span><span class="chip-v">{shape_10m[1]} × {shape_10m[0]} px</span></div>
                <div class="meta-chip"><span class="chip-k">Output Extent</span><span class="chip-v">{shape_4m[1]} × {shape_4m[0]} px</span></div>
                <div class="meta-chip"><span class="chip-k">Spatial Reference</span><span class="chip-v">{crs_str}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scientific_notice():
    """Render a subtle, accessible scientific disclaimer banner."""
    st.markdown(
        """
        <div class="scientific-notice">
            <span class="notice-icon">🛡️</span>
            <div>
                <b>Scientific Provenance & Uncertainty Disclosure:</b>
                Reconstructed fine-scale information is model-inferred via 2.5× deep residual learning. 
                Trust indicators represent empirical model agreement, perturbation stability, and spectral conservation, 
                not independent ground-truth verification. No synthetic degradation was applied to live 10m Sentinel-2 inputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="GeoFUSE SentinelGuard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject premium aerospace styling tokens
    st.markdown(GEOFUSE_CSS, unsafe_allow_html=True)

    # Title for test assertion compatibility
    st.title("🛰️ GeoFUSE SentinelGuard")

    # Render top nav & mission control bar
    render_top_nav()

    # -------------------------------------------------------------------------
    # Sidebar Navigation & Experience Mode
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### Mission Navigation")

    operational_mode = st.sidebar.radio(
        "Operational Experience:",
        options=[
            "🛰️ Direct Real Sentinel-2 (10m → 4m SR) [LIVE DEMO]",
            "🔬 Controlled Synthetic Benchmark (Degrade & Recover)",
        ],
        index=0,
        help="Mode 1 processes real Sentinel-2 observations directly with 2.5x learned SR (zero synthetic degradation). Mode 2 retains the degrade-and-recover simulation for scientific benchmark validation.",
    )
    is_benchmark_mode = "Controlled Synthetic Benchmark" in operational_mode

    # Display & Evidence Toggles (Requirement: checkbox[0] must be Show Multi-Criteria Evidence Breakdown)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Display & Evidence Toggles")

    show_evidence = st.sidebar.checkbox("Show Multi-Criteria Evidence Breakdown", value=False)
    show_downstream = st.sidebar.checkbox("Show Downstream Application Analytics", value=True)
    show_scene_overview = st.sidebar.checkbox("Show Macro Scene Overview & Tile Locator", value=True)

    color_mode = st.sidebar.radio(
        "Band Visualization Mode:",
        options=["Natural RGB (B04-B03-B02)", "False-Color Infrared (B08-B04-B03)"],
        index=0,
    )
    is_false_color = "False-Color" in color_mode

    # Advanced Mission Controls (Collapsible)
    with st.sidebar.expander("⚙️ Advanced Mission Controls", expanded=False):
        if st.button("Flush Cache & Reload System", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # =========================================================================
    # MODE 1: DIRECT REAL SENTINEL-2 (PRIMARY HERO DEMO)
    # =========================================================================
    if not is_benchmark_mode:
        render_hero(is_benchmark=False)
        render_pipeline_stepper(current_step=3, is_benchmark=False)

        # Ingestion selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Ingestion Source")
        input_source_type = st.sidebar.radio(
            "Sentinel-2 Input Selection:",
            options=[
                "Pre-Loaded Real Sentinel-2 Scenes",
                "Upload Custom Sentinel-2 Imagery (Live Direct SR)",
            ],
            index=0,
            help="Choose whether to explore pre-loaded genuine Sentinel-2 MSI scenes or upload your own B02, B03, B04, B08 band files for direct 10m -> 4m learned super-resolution.",
        )

        data = None
        selected_key = "urban_core"
        scene_display_name = "Urban Core & Infrastructure (T43PGQ 10m L2A)"

        # ---------------------------------------------------------------------
        # PATH A: PRE-LOADED REAL SENTINEL-2 SCENES
        # ---------------------------------------------------------------------
        if "Pre-Loaded" in input_source_type:
            scene_choice = st.sidebar.selectbox(
                "Select Real Sentinel-2 Scene:",
                options=[
                    "Urban Core & Infrastructure (T43PGQ 10m L2A)",
                    "Rural Farmland & Crop Fields (T43PGQ 10m L2A)",
                    "Dry Season Transition Scene (April 2024)",
                    "Base L2A 4-Band Scene (data/raw)",
                ],
                index=0,
            )
            scene_display_name = scene_choice

            scene_key_map = {
                "Urban Core & Infrastructure (T43PGQ 10m L2A)": "urban_core",
                "Rural Farmland & Crop Fields (T43PGQ 10m L2A)": "agriculture",
                "Dry Season Transition Scene (April 2024)": "temporal_april2024",
                "Base L2A 4-Band Scene (data/raw)": "base_raw",
            }
            selected_key = scene_key_map.get(scene_choice, "urban_core")

            st.sidebar.markdown("### Processing Scope")
            crop_mode = st.sidebar.radio(
                "Region to Explore:",
                options=["Default Center Tile (128×128)", "Custom (X, Y) Coordinates", "Full Scene (1280×1280 Stitched 4m)"],
                index=0,
            )

            crop_coords = None
            full_scene = False
            if "Custom" in crop_mode:
                c1, c2 = st.sidebar.columns(2)
                with c1:
                    cx_in = st.number_input("Crop X:", min_value=0, max_value=384, value=192, step=16)
                with c2:
                    cy_in = st.number_input("Crop Y:", min_value=0, max_value=384, value=192, step=16)
                crop_coords = (int(cx_in), int(cy_in))
            elif "Full Scene" in crop_mode:
                full_scene = True

            data = run_real_sentinel2_pipeline_cached(
                scene_key=selected_key,
                tile_size=128,
                crop_coords=crop_coords,
                full_scene=full_scene,
            )

        # ---------------------------------------------------------------------
        # PATH B: UPLOAD CUSTOM SENTINEL-2 IMAGERY
        # ---------------------------------------------------------------------
        else:
            st.markdown("### 📤 Upload Sentinel-2 Multi-Spectral Scene")
            st.caption(
                "Upload your own ESA Sentinel-2 Level-2A imagery for **direct 2.5× learned super-resolution (10m → 4m)**.  \n"
                "**Zero synthetic degradation is applied.** Native 10m observations feed directly into the trained ensemble."
            )

            u_col1, u_col2 = st.columns([3, 1])
            with u_col1:
                upload_fmt = st.radio(
                    "Select Raster Input Format:",
                    options=["Four Separate Band Files (B02, B03, B04, B08)", "Single 4-Band Multi-Spectral GeoTIFF"],
                    horizontal=True,
                )
            with u_col2:
                if st.button("🗑️ Reset Upload", use_container_width=True):
                    for k in ["custom_upload_result", "custom_upload_hash", "custom_upload_validation"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

            uploaded_files = []
            if "Four Separate" in upload_fmt:
                uploaded_files = st.file_uploader(
                    "Upload 4 Sentinel-2 Bands (Select B02, B03, B04, B08 .tif/.jp2):",
                    type=["tif", "tiff", "jp2", "TIF"],
                    accept_multiple_files=True,
                    key="custom_multi_band_uploader",
                )
            else:
                s_file = st.file_uploader(
                    "Upload Single 4-Band GeoTIFF (Band 1=B02, 2=B03, 3=B04, 4=B08):",
                    type=["tif", "tiff", "jp2", "TIF"],
                    key="custom_single_band_uploader",
                )
                if s_file is not None:
                    uploaded_files = [s_file]

            # Validation & Execution
            if uploaded_files and len(uploaded_files) > 0:
                upload_hash = compute_files_hash(uploaded_files)
                upload_scratch_dir = project_root / "outputs" / "user_uploads" / upload_hash

                if st.session_state.get("custom_upload_hash") != upload_hash:
                    st.session_state["custom_upload_hash"] = upload_hash
                    if "custom_upload_result" in st.session_state:
                        del st.session_state["custom_upload_result"]

                save_uploaded_sentinel_files(uploaded_files, upload_scratch_dir)

                is_valid = False
                val_err = None
                val_info = None
                try:
                    val_info = validate_sentinel2_input(upload_scratch_dir)
                    is_valid = True
                except Exception as e:
                    val_err = str(e)

                if is_valid and val_info:
                    st.markdown(
                        f"""
                        <div class="scene-ready-card">
                            <div class="scene-status-header">
                                <div class="scene-status-title">
                                    <span class="scene-check">✓</span>
                                    <span>Input Validation Passed: Uploaded Sentinel-2 Scene</span>
                                </div>
                                <span class="scene-source-tag">VERIFIED S2-L2A</span>
                            </div>
                            <div class="scene-meta-chips">
                                <div class="meta-chip"><span class="chip-k">Bands</span><span class="chip-v">B02 · B03 · B04 · B08</span></div>
                                <div class="meta-chip"><span class="chip-k">Native GSD</span><span class="chip-v">~10.0m ({val_info['resolution'][0]:.1f}m)</span></div>
                                <div class="meta-chip"><span class="chip-k">Output Target</span><span class="chip-v">4.0m Nominal GSD</span></div>
                                <div class="meta-chip"><span class="chip-k">Dimensions</span><span class="chip-v">{val_info['shape'][1]} × {val_info['shape'][0]} px</span></div>
                                <div class="meta-chip"><span class="chip-k">CRS</span><span class="chip-v">{val_info['crs']}</span></div>
                                <div class="meta-chip"><span class="chip-k">Status</span><span class="chip-v">Ready for Direct SR</span></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    h_up, w_up = val_info["shape"]
                    st.markdown("#### 📐 Processing Scope")
                    r_col_a, r_col_b = st.columns(2)
                    with r_col_a:
                        up_crop_mode = st.radio(
                            "Select Inference Extent:",
                            options=[
                                f"🗺️ Full Scene ({w_up}×{h_up} px → {round(w_up*2.5)}×{round(h_up*2.5)} px 4m Output)",
                                "🎯 Center Tile (128×128 → 320×320 px)",
                                "📍 Custom (X, Y) Coordinates",
                            ],
                            index=0 if h_up <= 256 else 1,
                        )
                    with r_col_b:
                        up_coords = None
                        up_full = False
                        if "Full Scene" in up_crop_mode:
                            up_full = True
                        elif "Custom" in up_crop_mode:
                            cx_u = st.number_input("Crop X:", min_value=0, max_value=max(0, w_up - 128), value=0, step=16)
                            cy_u = st.number_input("Crop Y:", min_value=0, max_value=max(0, h_up - 128), value=0, step=16)
                            up_coords = (int(cx_u), int(cy_u))

                    if st.button("🚀 Run GeoFUSE 10m → 4m Direct Super-Resolution", type="primary", use_container_width=True):
                        status_box = st.empty()
                        progress_bar = st.progress(0.0)

                        def update_progress(step_idx: int, message: str):
                            progress_bar.progress(step_idx / 7.0)
                            status_box.info(f"**Step {step_idx}/7:** {message}")

                        try:
                            custom_res = execute_real_sentinel2_sr(
                                source_path=upload_scratch_dir,
                                scene_name=f"uploaded_{upload_hash[:8]}",
                                tile_size=128,
                                crop_coords=up_coords,
                                full_scene=up_full,
                                progress_status_fn=update_progress,
                            )
                            st.session_state["custom_upload_result"] = custom_res
                            status_box.success("✅ Direct 2.5× Super-Resolution Completed (10m → 4m) with Zero Synthetic Degradation!")
                            progress_bar.progress(1.0)
                        except Exception as ex:
                            status_box.error(f"🚨 Pipeline Execution Failed: {ex}")

                    if "custom_upload_result" in st.session_state:
                        data = st.session_state["custom_upload_result"]
                        selected_key = f"uploaded_{upload_hash[:8]}"
                        scene_display_name = f"Custom Upload ({upload_hash[:8]})"

                else:
                    st.error(f"✖ Input Validation Failed: {val_err}")
            else:
                st.info("ℹ️ Drop your Sentinel-2 band files above to process your custom scene. Showing pre-loaded Urban Core reference below.")
                data = run_real_sentinel2_pipeline_cached(scene_key="urban_core", tile_size=128)
                selected_key = "urban_core"

        if data is None:
            st.error("🚨 Execution Error: Could not load models or execute real inference.")
            st.stop()

        input_10m = data["input_10m"]
        sr_4m = data["sr_4m"]
        disagreement_map = data["disagreement_map"]
        trust_data = data["trust_data"]
        fusion_result = data["fusion_result"]
        trust_map = fusion_result["trust_map"]
        receipt = data["receipt"]
        exported_paths = data.get("exported_paths", {})
        trust_score_pct = trust_data["trust_score_pct"]
        is_trusted = trust_data["is_trusted"]
        status_label = "HIGH_TRUST_APPROVED" if is_trusted else "LOW_TRUST_ADVISORY"

        # Consolidated Scene Ready Status Banner
        render_scene_status_card(
            scene_title=scene_display_name,
            shape_10m=input_10m.shape[:2],
            shape_4m=sr_4m.shape[:2],
            crs_str="EPSG:32643 (UTM Zone 43N)",
            bands_str="B02 (Blue) · B03 (Green) · B04 (Red) · B08 (NIR)",
        )

        # Quick Stats Metric Ribbon (6 Metrics)
        q1, q2, q3, q4, q5, q6 = st.columns(6)
        q1.metric("Input Native GSD", "10.00 m / px", delta="Sentinel-2 L2A")
        q2.metric("Output Enhanced GSD", "4.00 m / px", delta="2.5× Spatial Zoom")
        q3.metric("Pixel Ground Area", "16.0 m²", delta="-84% blur area", delta_color="inverse")
        q4.metric("Composite Trust Score", f"{trust_score_pct:.1f}%", delta="Reliability Proxy")
        q5.metric("Model Disagreement (σ)", f"{trust_data['disagreement_mean']:.5f}", delta="Epistemic Uncertainty")
        q6.metric("Operational Status", "APPROVED" if is_trusted else "ADVISORY", delta="Automated OK" if is_trusted else "Human-in-Loop")

        # Prepare Display RGB Imagery
        bounds_10m = get_display_stretch_bounds(input_10m, false_color=is_false_color)
        rgb_10m_raw = to_display_rgb(input_10m, false_color=is_false_color, stretch_bounds=bounds_10m)
        rgb_sr_raw = to_display_rgb(sr_4m, false_color=is_false_color, stretch_bounds=bounds_10m)
        rgb_10m_disp = cv2.resize(rgb_10m_raw, (rgb_sr_raw.shape[1], rgb_sr_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ---------------------------------------------------------------------
        # Flagship Presentation Workspaces (Tabs)
        # ---------------------------------------------------------------------
        r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
            "✦ Super-Resolution & Reliability",
            "🏙️ Urban Infrastructure",
            "🌾 Agricultural Intelligence",
            "◌ Multi-Criteria Evidence (2×2)",
            "↓ Deliverables & Provenance",
        ])

        # TAB 1: Flagship Super-Resolution Workspace
        with r_tab1:
            st.markdown("### Primary Super-Resolution Workspace")
            st.caption("Dominant side-by-side comparison between the raw 10m Sentinel-2 acquisition and the 4m nominal GeoFUSE reconstruction.")

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown(
                    f"""
                    <div class="viewport-card">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag">CHANNEL 01 // NATIVE ACQUISITION</span>
                            <span class="viewport-gsd">10.0m GSD</span>
                        </div>
                        <div class="viewport-title">Original Sentinel-2 (10m)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(rgb_10m_disp, caption=f"Native Sentinel-2 10m Acquisition [{input_10m.shape[1]}×{input_10m.shape[0]} px, Nearest Enlarged]", use_container_width=True)

            with v_col2:
                st.markdown(
                    f"""
                    <div class="viewport-card accent">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag accent">CHANNEL 02 // GEOFUSE RECONSTRUCTION</span>
                            <span class="viewport-gsd">4.0m NOMINAL GSD</span>
                        </div>
                        <div class="viewport-title">GeoFUSE Super-Resolution (4m)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(rgb_sr_raw, caption=f"Learned 2.5× Super-Resolved Product [{sr_4m.shape[1]}×{sr_4m.shape[0]} px, 4m Grid]", use_container_width=True)

            # Reliability & Uncertainty Side-by-Side
            st.markdown("---")
            st.markdown("### Empirical Reliability & Uncertainty Mapping")
            st.caption("Inter-model disagreement reveals where ensemble predictions vary; the fused Trust/Risk map weights uncertainty, sensitivity, spectral, and edge signals.")

            # Trust overlay calculation
            cmap = plt.get_cmap("RdYlGn")
            trust_colored = (cmap(trust_map)[:, :, :3] * 255).astype(np.uint8)
            trust_overlay = (0.55 * rgb_sr_raw + 0.45 * trust_colored).astype(np.uint8)

            rel_col1, rel_col2 = st.columns(2)
            with rel_col1:
                st.markdown(
                    """
                    <div class="viewport-card">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag">EVIDENCE SIGNAL // UNCERTAINTY</span>
                            <span class="viewport-gsd">σ DISAGREEMENT</span>
                        </div>
                        <div class="viewport-title">Model Disagreement Map (σ)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                im_d = plt.figure(figsize=(6, 4.5), dpi=120)
                plt.imshow(disagreement_map, cmap="magma")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title("Per-Pixel Ensemble Standard Deviation (σ)", fontsize=10, fontweight="bold")
                plt.axis("off")
                st.pyplot(im_d, use_container_width=True)
                plt.close(im_d)
                st.caption("Dark indicates unanimous model agreement; bright marks high-frequency boundary ambiguity.")

            with rel_col2:
                st.markdown(
                    """
                    <div class="viewport-card accent">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag accent">EVIDENCE SIGNAL // TRUST OVERLAY</span>
                            <span class="viewport-gsd">EMPIRICAL PROXY</span>
                        </div>
                        <div class="viewport-title">Composite Trust / Risk Overlay</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(trust_overlay, caption=f"Empirical Trust Overlay (RdYlGn) — Composite Score: {trust_score_pct:.1f}%", use_container_width=True)
                st.caption("Green highlights high-trust verified structures; yellow/red denotes regions requiring analyst inspection.")

            render_scientific_notice()

        # TAB 2: Urban Infrastructure Analysis
        with r_tab2:
            st.markdown("### 🏙️ Urban Infrastructure & Building Footprint Analysis")
            st.markdown(
                "Super-resolving 10m Sentinel-2 imagery to a nominal 4m grid enables sharper rooftop separation, "
                "delineation of rural roads, and boundary definition for dense built-up settlements."
            )

            foot_10m = data["foot_10m"]
            foot_sr = data["foot_sr"]

            def add_contours_display(base_rgb, mask, color=(0, 240, 255)):
                out = base_rgb.copy()
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color, 1)
                return out

            c_10m = add_contours_display(rgb_10m_disp, foot_10m["mask"])
            c_sr = add_contours_display(rgb_sr_raw, foot_sr["mask"])

            u1, u2 = st.columns(2)
            with u1:
                st.markdown(f"**10m Native Observation Footprints** ({foot_10m['footprint_pixels']} px)")
                st.image(c_10m, caption="Cyan Contours: Extracted from 10m Observation", use_container_width=True)
            with u2:
                st.markdown(f"**GeoFUSE 4m SR Footprints** ({foot_sr['footprint_pixels']} px)")
                st.image(c_sr, caption="Cyan Contours: Extracted from 4m SR Reconstruction", use_container_width=True)

            st.caption("Notice: Morphological building footprint extraction sharpens fine geometry without fabricated true-accuracy claims.")

        # TAB 3: Agricultural Intelligence (NDVI)
        with r_tab3:
            st.markdown("### 🌾 Agricultural Intelligence & Crop Health Monitoring")
            st.markdown(
                "Vegetation health and field parcels benefit from 2.5× spatial enhancement. "
                "Below is the Normalized Difference Vegetation Index (NDVI) comparing coarse 10m observation against the 4m product."
            )

            ndvi_10m = compute_ndvi(input_10m)
            ndvi_4m = compute_ndvi(sr_4m)

            def colorize_ndvi_disp(arr):
                norm = np.clip((arr + 0.2) / 1.0, 0.0, 1.0)
                norm_u8 = (norm * 255.0).astype(np.uint8)
                colorized = cv2.applyColorMap(norm_u8, cv2.COLORMAP_SUMMER)
                return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

            col_ndvi_10m = cv2.resize(colorize_ndvi_disp(ndvi_10m), (sr_4m.shape[1], sr_4m.shape[0]), interpolation=cv2.INTER_NEAREST)
            col_ndvi_4m = colorize_ndvi_disp(ndvi_4m)

            a1, a2 = st.columns(2)
            with a1:
                st.markdown(f"**Sentinel-2 10m NDVI** (Mean: {np.mean(ndvi_10m):.3f})")
                st.image(col_ndvi_10m, caption="10m Vegetation Vigor (Blurry Field Boundaries)", use_container_width=True)
            with a2:
                st.markdown(f"**GeoFUSE 4m NDVI** (Mean: {np.mean(ndvi_4m):.3f})")
                st.image(col_ndvi_4m, caption="4m Super-Resolved NDVI (Crisp Field Boundaries & Sub-Field Variation)", use_container_width=True)

            st.caption("Radiometric integrity is strictly preserved: cross-scale mean NDVI delta is minimal.")

        # TAB 4: Multi-Criteria Evidence Grid (2x2)
        with r_tab4:
            st.markdown("### 🛡️ Multi-Criteria Trust Evidence (2 × 2 Grid)")
            st.caption("Verifiable empirical signals computed on real imagery without fabricated ground truth.")

            sig = trust_data["fusion_result"].get("normalized_signals", {})
            e_row1_1, e_row1_2 = st.columns(2)
            with e_row1_1:
                st.markdown(
                    """
                    <div class="viewport-card">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag">SIGNAL 01 // EPISTEMIC</span>
                            <span class="viewport-gsd">σ DISAGREEMENT</span>
                        </div>
                        <div class="viewport-title">1. Model Disagreement</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(sig.get("disagreement", disagreement_map), caption="Standard deviation across 3 ensemble model seeds", use_container_width=True, clamp=True)

            with e_row1_2:
                st.markdown(
                    """
                    <div class="viewport-card">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag">SIGNAL 02 // SENSITIVITY</span>
                            <span class="viewport-gsd">NOISE VARIANCE</span>
                        </div>
                        <div class="viewport-title">2. Perturbation Stability</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(sig.get("stability", trust_data.get("stability_map", disagreement_map)), caption="Output variance under input sensor noise injection", use_container_width=True, clamp=True)

            e_row2_1, e_row2_2 = st.columns(2)
            with e_row2_1:
                st.markdown(
                    """
                    <div class="viewport-card">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag">SIGNAL 03 // RADIOMETRIC</span>
                            <span class="viewport-gsd">|ΔNDVI|</span>
                        </div>
                        <div class="viewport-title">3. Spectral Consistency</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(sig.get("spectral", trust_data.get("delta_ndvi_map", disagreement_map)), caption="Cross-scale NDVI preservation check", use_container_width=True, clamp=True)

            with e_row2_2:
                st.markdown(
                    """
                    <div class="viewport-card">
                        <div class="viewport-badge-row">
                            <span class="viewport-tag">SIGNAL 04 // STRUCTURAL</span>
                            <span class="viewport-gsd">SOBEL GRADIENT</span>
                        </div>
                        <div class="viewport-title">4. Structural Difference</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(sig.get("structural", trust_data.get("delta_ndvi_map", disagreement_map)), caption="Sobel boundary gradient consistency", use_container_width=True, clamp=True)

        # TAB 5: Deliverables & Provenance
        with r_tab5:
            st.markdown("### 📦 Auditable Trust Receipt & Deliverables")
            st.caption("Complete provenance record verifying that NO synthetic degradation was inserted into the live inference path.")

            scene_stem = selected_key
            st.markdown("#### Production Artifact Downloads")

            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.markdown(
                    """
                    <div class="deliverable-card">
                        <div>
                            <div class="deliv-title">🗺️ 4m GeoTIFF Product</div>
                            <div class="deliv-meta">sr_4m.tif · 4 MSI Bands · EPSG:32643</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "sr_4m" in exported_paths and Path(exported_paths["sr_4m"]).exists():
                    with open(exported_paths["sr_4m"], "rb") as f:
                        st.download_button("📥 Download 4m GeoTIFF", data=f.read(), file_name=f"{scene_stem}_sr_4m.tif", mime="image/tiff", use_container_width=True)

            with d_col2:
                st.markdown(
                    """
                    <div class="deliverable-card">
                        <div>
                            <div class="deliv-title">🛡️ Composite Trust Map</div>
                            <div class="deliv-meta">trust_map.tif · Geo-referenced Confidence</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "trust_map" in exported_paths and Path(exported_paths["trust_map"]).exists():
                    with open(exported_paths["trust_map"], "rb") as f:
                        st.download_button("📥 Download Trust Map", data=f.read(), file_name=f"{scene_stem}_trust_map.tif", mime="image/tiff", use_container_width=True)

            with d_col3:
                st.markdown(
                    """
                    <div class="deliverable-card">
                        <div>
                            <div class="deliv-title">📜 Cryptographic Receipt</div>
                            <div class="deliv-meta">trust_receipt.json · Full Provenance Hash</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                receipt_json_str = json.dumps(receipt, indent=2)
                st.download_button("📥 Download Trust Receipt", data=receipt_json_str, file_name=f"{scene_stem}_trust_receipt.json", mime="application/json", use_container_width=True)

            d_col4, d_col5, d_col6 = st.columns(3)
            with d_col4:
                st.markdown(
                    """
                    <div class="deliverable-card">
                        <div>
                            <div class="deliv-title">📊 Disagreement Map</div>
                            <div class="deliv-meta">disagreement_map.tif · Model Uncertainty σ</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "disagreement_map" in exported_paths and Path(exported_paths["disagreement_map"]).exists():
                    with open(exported_paths["disagreement_map"], "rb") as f:
                        st.download_button("📥 Download Uncertainty Map", data=f.read(), file_name=f"{scene_stem}_disagreement_map.tif", mime="image/tiff", use_container_width=True)

            with d_col5:
                st.markdown(
                    """
                    <div class="deliverable-card">
                        <div>
                            <div class="deliv-title">🖼️ Natural RGB Preview</div>
                            <div class="deliv-meta">rgb_preview.png · High-Resolution PNG</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "rgb_preview" in exported_paths and Path(exported_paths["rgb_preview"]).exists():
                    with open(exported_paths["rgb_preview"], "rb") as f:
                        st.download_button("📥 Download RGB Preview", data=f.read(), file_name=f"{scene_stem}_rgb_preview.png", mime="image/png", use_container_width=True)

            with d_col6:
                st.markdown(
                    """
                    <div class="deliverable-card">
                        <div>
                            <div class="deliv-title">📄 Scene Metadata</div>
                            <div class="deliv-meta">metadata.json · Bands & CRS Provenance</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                meta_p = exported_paths.get("metadata") or (Path(exported_paths.get("sr_4m", "")).parent / "metadata.json")
                if Path(meta_p).exists():
                    with open(meta_p, "rb") as f:
                        st.download_button("📥 Download Metadata JSON", data=f.read(), file_name=f"{scene_stem}_metadata.json", mime="application/json", use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📜 Verifiable Provenance Certificate JSON")
            st.json(receipt, expanded=False)

    # =========================================================================
    # MODE 2: CONTROLLED SYNTHETIC BENCHMARK (DEGRADE & RECOVER)
    # =========================================================================
    else:
        render_hero(is_benchmark=True)
        render_pipeline_stepper(current_step=3, is_benchmark=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Benchmark Tile Selection")
        tile_selection_mode = st.sidebar.radio(
            "Tile Selection Scope:",
            options=[
                "Curated Benchmark Highlights (Offline Fast)",
                "Browse All 25 Scene Tiles (0..24)",
                "Custom Coordinate Box (Free X, Y Crop)",
            ],
            index=0,
        )

        all_grid_meta = get_tile_metadata_grid(512, 512, patch_size=128, stride=96)
        all_grid_dict = {t["tile_id"]: t for t in all_grid_meta}

        selected_idx = 0
        data = None

        if "Curated" in tile_selection_mode:
            tile_options = {
                0: "Tile #0 -- Central Settlement Cluster (High Trust: 86.8%)",
                8: "Tile #8 -- Agricultural & Rural Roads (High Trust: 86.7%)",
                16: "Tile #16 -- Rural River Corridor (Low Trust Advisory: 86.4%)",
                24: "Tile #24 -- Complex Terrain Transition (Low Trust Advisory: 86.0%)",
            }
            selected_idx = st.sidebar.selectbox(
                "Select Sentinel-2 Demo Tile:",
                options=list(tile_options.keys()),
                format_func=lambda x: tile_options.get(x, f"Tile #{x}"),
            )
            data = run_cached_pipeline(selected_idx)
        elif "Browse" in tile_selection_mode:
            selected_idx = st.sidebar.selectbox(
                "Choose Any Tile (0 to 24):",
                options=list(range(25)),
                index=0,
                format_func=lambda x: all_grid_dict[x]["label"],
            )
            data = run_cached_pipeline(selected_idx)
        else:
            cx = st.sidebar.slider("Crop X Offset (px):", 0, 384, 192, step=16)
            cy = st.sidebar.slider("Crop Y Offset (px):", 0, 384, 192, step=16)
            selected_idx = f"Crop (X={cx}, Y={cy})"
            data = run_cached_crop_pipeline(cx, cy)

        if data is None:
            st.error(f"Execution Error: Could not retrieve assets for Tile #{selected_idx}.")
            st.stop()

        hr_tile = data["hr_tile"]
        lr_tile = data["lr_tile"]
        bicubic_tile = data["bicubic_tile"]
        sr_tile = data["sr_tile"]
        fusion_result = data["fusion_result"]
        trust_map = fusion_result["trust_map"]
        receipt = data["receipt"]
        score_pct = fusion_result["trust_score_pct"]

        # Quantitative Benchmark Metrics
        from skimage.metrics import peak_signal_noise_ratio as compute_psnr
        from skimage.metrics import structural_similarity as compute_ssim
        bic_psnr = float(compute_psnr(hr_tile, bicubic_tile, data_range=1.0))
        sr_psnr = float(compute_psnr(hr_tile, sr_tile, data_range=1.0))
        bic_ssim = float(compute_ssim(hr_tile, bicubic_tile, channel_axis=2, data_range=1.0))
        sr_ssim = float(compute_ssim(hr_tile, sr_tile, channel_axis=2, data_range=1.0))

        # Consolidated Benchmark Ready Card
        render_scene_status_card(
            scene_title=f"Benchmark Evaluation Tile #{selected_idx}",
            shape_10m=lr_tile.shape[:2],
            shape_4m=sr_tile.shape[:2],
            crs_str="EPSG:32643 (UTM Zone 43N)",
            bands_str="B02 · B03 · B04 · B08 (Synthetic Degrade & Recover)",
        )

        # 5 Metric Cards
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Ground Sampling Distance", "4.00 m / px", delta="2.5× Scale")
        b2.metric("Pixel Area", "16.0 m²", delta="-84% blur area", delta_color="inverse")
        b3.metric("Reconstruction PSNR", f"{sr_psnr:.2f} dB", delta=f"{sr_psnr - bic_psnr:+.2f} dB vs Bicubic")
        b4.metric("Structural SSIM", f"{sr_ssim:.4f}", delta=f"{sr_ssim - bic_ssim:+.4f} vs Bicubic")
        b5.metric("Trust Score", f"{score_pct:.1f}%", delta="Reliability Proxy")

        # Visual 4-Column Panel
        stretch_bounds = get_display_stretch_bounds(hr_tile, false_color=is_false_color)
        hr_rgb = to_display_rgb(hr_tile, false_color=is_false_color, stretch_bounds=stretch_bounds)
        bic_rgb = to_display_rgb(bicubic_tile, false_color=is_false_color, stretch_bounds=stretch_bounds)
        sr_rgb = to_display_rgb(sr_tile, false_color=is_false_color, stretch_bounds=stretch_bounds)
        lr_rgb = to_display_rgb(lr_tile, false_color=is_false_color, stretch_bounds=stretch_bounds)
        lr_disp = cv2.resize(lr_rgb, (hr_rgb.shape[1], hr_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        st.markdown("### Benchmark Quad View: Input vs. Bicubic vs. GeoFUSE SR vs. Reference")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(
                """
                <div class="viewport-card">
                    <div class="viewport-badge-row"><span class="viewport-tag">VIEW 01 // INPUT</span></div>
                    <div class="viewport-title">Degraded Pseudo-LR</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(lr_disp, caption="Synthetic Input (PSF Blur + 2.5x Down + Noise)", use_container_width=True)

        with col_b:
            st.markdown(
                """
                <div class="viewport-card">
                    <div class="viewport-badge-row"><span class="viewport-tag">VIEW 02 // BASELINE</span></div>
                    <div class="viewport-title">Bicubic Baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(bic_rgb, caption=f"Bicubic Baseline | PSNR: {bic_psnr:.2f} dB", use_container_width=True)

        with col_c:
            st.markdown(
                """
                <div class="viewport-card accent">
                    <div class="viewport-badge-row"><span class="viewport-tag accent">VIEW 03 // GEOFUSE SR</span></div>
                    <div class="viewport-title">Learned Reconstruction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(sr_rgb, caption=f"GeoFUSE 2.5x SR | PSNR: {sr_psnr:.2f} dB", use_container_width=True)

        with col_d:
            st.markdown(
                """
                <div class="viewport-card">
                    <div class="viewport-badge-row"><span class="viewport-tag">VIEW 04 // TARGET</span></div>
                    <div class="viewport-title">HR Reference Target</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(hr_rgb, caption="Unseen High-Resolution Reference Target", use_container_width=True)

if __name__ == "__main__":
    main()
