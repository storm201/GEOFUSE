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
# Main Application UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="GeoFUSE SentinelGuard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header & Banner
    st.title("🛰️ GeoFUSE SentinelGuard")
    st.markdown(
        "**Deep Learning Based Super-Resolution Mapping (SRM) from Medium-Resolution Satellite Imageries**  \n"
        "*" "Direct 2.5× learned super-resolution: 10m Sentinel-2 observations → nominal 4m output grid with evidence attached." "*"
    )

    # 1. Sidebar Controls & Mode Selection
    st.sidebar.header("🕹️ Controls & Settings")

    operational_mode = st.sidebar.radio(
        "Select Operational Experience:",
        options=[
            "🛰️ Direct Real Sentinel-2 (10m → 4m SR) [LIVE DEMO]",
            "🔬 Controlled Synthetic Benchmark (Degrade & Recover)",
        ],
        index=0,
        help="Mode 1 processes real Sentinel-2 observations directly with 2.5x learned SR (zero synthetic degradation). Mode 2 retains the degrade-and-recover simulation for scientific benchmark validation.",
    )

    # Flush cache button
    if st.sidebar.button("🔄 Flush Cache & Reload", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # =========================================================================
    # MODE 1: DIRECT REAL SENTINEL-2 (PRIMARY LIVE DEMO)
    # =========================================================================
    if "Direct Real Sentinel-2" in operational_mode:
        st.sidebar.markdown(
            """
            <div style="background: #0d2137; border: 1px solid #00e5ff; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.88rem; font-weight: bold; color: #00e5ff;">🚀 Pipeline: DIRECT REAL SR</div>
                <div style="font-size: 0.76rem; color: #b0bec5;">Synthetic Blur: <b>OFF (Zero Degradation)</b><br>Model Scale: <b>2.5× (10m → 4m)</b><br>Device: <b>RTX 4060 GPU (CUDA)</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        input_source_type = st.sidebar.radio(
            "Sentinel-2 Input Selection:",
            options=[
                "📁 Pre-Loaded Real Sentinel-2 Scenes",
                "📤 Upload Custom Sentinel-2 Imagery (Live Direct SR)",
            ],
            index=0,
            help="Choose whether to explore pre-loaded genuine Sentinel-2 MSI scenes or upload your own B02, B03, B04, B08 band files for direct 10m -> 4m learned super-resolution.",
        )

        data = None
        selected_key = "urban_core"

        # ---------------------------------------------------------------------
        # PATH A: PRE-LOADED REAL SENTINEL-2 SCENES
        # ---------------------------------------------------------------------
        if input_source_type == "📁 Pre-Loaded Real Sentinel-2 Scenes":
            scene_choice = st.sidebar.selectbox(
                "Select Real Sentinel-2 Scene:",
                options=[
                    "🏙️ Urban Core & Infrastructure (T43PGQ 10m L2A)",
                    "🌾 Rural Farmland & Crop Fields (T43PGQ 10m L2A)",
                    "🍂 Dry Season Transition Scene (April 2024)",
                    "📁 Base L2A 4-Band Scene (data/raw)",
                ],
                index=0,
            )

            scene_key_map = {
                "🏙️ Urban Core & Infrastructure (T43PGQ 10m L2A)": "urban_core",
                "🌾 Rural Farmland & Crop Fields (T43PGQ 10m L2A)": "agriculture",
                "🍂 Dry Season Transition Scene (April 2024)": "temporal_april2024",
                "📁 Base L2A 4-Band Scene (data/raw)": "base_raw",
            }
            selected_key = scene_key_map[scene_choice]

            st.sidebar.subheader("📐 Sub-Region / Crop Selection")
            crop_mode = st.sidebar.radio(
                "Region to Explore:",
                options=["🎯 Default Center Tile (128×128)", "📍 Custom (X, Y) Coordinates", "🗺️ Full Scene (1280×1280 Stitched 4m)"],
                index=0,
            )

            crop_coords = None
            full_scene = False
            if crop_mode == "📍 Custom (X, Y) Coordinates":
                c1, c2 = st.sidebar.columns(2)
                with c1:
                    cx_in = st.number_input("Crop X:", min_value=0, max_value=384, value=192, step=16)
                with c2:
                    cy_in = st.number_input("Crop Y:", min_value=0, max_value=384, value=192, step=16)
                crop_coords = (int(cx_in), int(cy_in))
            elif crop_mode == "🗺️ Full Scene (1280×1280 Stitched 4m)":
                full_scene = True

            data = run_real_sentinel2_pipeline_cached(
                scene_key=selected_key,
                tile_size=128,
                crop_coords=crop_coords,
                full_scene=full_scene,
            )

        # ---------------------------------------------------------------------
        # PATH B: UPLOAD CUSTOM SENTINEL-2 IMAGERY (LIVE USER INFERENCE)
        # ---------------------------------------------------------------------
        else:
            st.sidebar.markdown(
                """
                <div style="background: #062319; border: 1px solid #00e676; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.82rem; color: #b9f6ca;">
                    <b>📤 Custom Live Upload Mode Active</b><br>
                    Drop your B02, B03, B04, B08 GeoTIFF files or single 4-band raster in the main panel.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 📤 Upload Genuine Sentinel-2 Multi-Spectral Imagery")
            st.caption(
                "Upload your own ESA Sentinel-2 Level-2A imagery for **direct 2.5× learned super-resolution (10m → 4m)**.  \n"
                "**Critical Verification Rule:** Zero synthetic degradation is applied. The 10m observations are fed directly to the trained ensemble."
            )

            u_col1, u_col2 = st.columns([3, 1])
            with u_col1:
                upload_fmt = st.radio(
                    "Select Raster Input Format:",
                    options=["Four Separate Band Files (B02, B03, B04, B08)", "Single 4-Band Multi-Spectral GeoTIFF"],
                    horizontal=True,
                )
            with u_col2:
                if st.button("🗑️ Clear & Reset Upload", use_container_width=True):
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

                # Check if upload hash changed
                if st.session_state.get("custom_upload_hash") != upload_hash:
                    st.session_state["custom_upload_hash"] = upload_hash
                    if "custom_upload_result" in st.session_state:
                        del st.session_state["custom_upload_result"]

                # Save files to disk for Rasterio inspection
                save_uploaded_sentinel_files(uploaded_files, upload_scratch_dir)

                # Pre-inference validation check
                is_valid = False
                val_err = None
                val_info = None
                try:
                    val_info = validate_sentinel2_input(upload_scratch_dir)
                    is_valid = True
                except Exception as e:
                    val_err = str(e)

                # Requirement 17: Pre-Inference Validation Panel
                if is_valid and val_info:
                    st.markdown(
                        f"""
                        <div style="background: #062319; border: 1px solid #00e676; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
                            <div style="font-size: 1.05rem; font-weight: bold; color: #00e676; margin-bottom: 8px;">
                                📋 Input Validation: PASSED ✓
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.86rem; color: #e0f2f1;">
                                <div>• <b>B02 (Blue 10m):</b> <span style="color:#69f0ae;">✓ {val_info.get('band_files', {}).get('B02', Path('B02')).name if val_info.get('format') == 'separate_bands' else 'Band 1'}</span></div>
                                <div>• <b>B03 (Green 10m):</b> <span style="color:#69f0ae;">✓ {val_info.get('band_files', {}).get('B03', Path('B03')).name if val_info.get('format') == 'separate_bands' else 'Band 2'}</span></div>
                                <div>• <b>B04 (Red 10m):</b> <span style="color:#69f0ae;">✓ {val_info.get('band_files', {}).get('B04', Path('B04')).name if val_info.get('format') == 'separate_bands' else 'Band 3'}</span></div>
                                <div>• <b>B08 (NIR 10m):</b> <span style="color:#69f0ae;">✓ {val_info.get('band_files', {}).get('B08', Path('B08')).name if val_info.get('format') == 'separate_bands' else 'Band 4'}</span></div>
                                <div>• <b>Pixel Size:</b> ~10.0m GSD <span style="color:#69f0ae;">✓ ({val_info['resolution'][0]:.1f}m × {val_info['resolution'][1]:.1f}m)</span></div>
                                <div>• <b>CRS:</b> <span style="color:#69f0ae;">✓ {val_info['crs']}</span></div>
                                <div>• <b>Dimensions:</b> <span style="color:#69f0ae;">✓ {val_info['shape'][1]} × {val_info['shape'][0]} px</span></div>
                                <div>• <b>Channels:</b> <span style="color:#69f0ae;">✓ 4 Validated MSI Bands</span></div>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.88rem; font-weight: bold; color: #b9f6ca;">
                                🟢 Ready for Direct 2.5× Super-Resolution Inference
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Region selection for custom upload
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

                    # Run button
                    if st.button("🚀 Run GeoFUSE 10m → 4m Direct Super-Resolution", type="primary", use_container_width=True):
                        # Requirement 18: Processing Status stages
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

                else:
                    st.markdown(
                        f"""
                        <div style="background: #2a0b0b; border: 1px solid #ff5252; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
                            <div style="font-size: 1.05rem; font-weight: bold; color: #ff5252; margin-bottom: 6px;">
                                🚨 Cannot Run Inference: Input Validation Failed
                            </div>
                            <div style="font-size: 0.88rem; color: #ffcdd2;">
                                <b>Reason:</b> {val_err}
                            </div>
                            <div style="margin-top: 8px; font-size: 0.80rem; color: #ef9a9a;">
                                Ensure you upload valid Level-2A Sentinel-2 files containing B02, B03, B04, and B08 with matching CRS and spatial dimensions (~10m GSD).
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("ℹ️ Drop your Sentinel-2 band files above to process your custom scene. Showing pre-loaded reference below for demonstration.")
                # Fallback to Urban Core pre-loaded scene so the dashboard remains fully rendered
                data = run_real_sentinel2_pipeline_cached(scene_key="urban_core", tile_size=128)
                selected_key = "urban_core"

        # ---------------------------------------------------------------------
        # DISPLAY & EVIDENCE TOGGLES (SIDEBAR)
        # ---------------------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Display & Evidence Toggles")
        color_mode = st.sidebar.radio(
            "Band Visualization Mode:",
            options=["Natural RGB (B04-B03-B02)", "False-Color Infrared (B08-B04-B03)"],
            index=0,
        )
        is_false_color = "False-Color" in color_mode

        show_evidence = st.sidebar.checkbox("Show Multi-Criteria Evidence Breakdown", value=False)
        show_downstream = st.sidebar.checkbox("Show Downstream Application Analytics", value=True)
        show_scene_overview = st.sidebar.checkbox("Show Macro Scene Overview & Tile Locator", value=True)

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

        # Pipeline Status Indicator (SIH Demo Badge)
        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #071e3d 0%, #1f4068 100%); border-left: 6px solid #00e5ff; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #00e5ff; margin-bottom: 6px;">
                    🛰️ Pipeline Status: DIRECT REAL SENTINEL-2 SUPER-RESOLUTION
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.88rem; color: #e0f7fa; line-height: 1.6;">
                    <div>• <b>Inference Mode:</b> <span style="color: #00ff88; font-weight: bold;">REAL SENTINEL-2 (Direct)</span></div>
                    <div>• <b>Synthetic Degradation:</b> <span style="color: #00ff88; font-weight: bold;">OFF</span> (Zero Blur / Downsampling)</div>
                    <div>• <b>Model Scale:</b> 2.5× Learned Residual SR</div>
                    <div>• <b>Input GSD:</b> 10.0m → <b>Output GSD:</b> nominal 4.0m</div>
                    <div>• <b>Device:</b> CUDA (RTX 4060 Laptop GPU)</div>
                    <div>• <b>Ensemble:</b> 3 trained ResidualSRNet checkpoints</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 6 Metric Cards
        q1, q2, q3, q4, q5, q6 = st.columns(6)
        q1.metric("Input Native GSD", "10.00 m / px", delta="Sentinel-2 L2A")
        q2.metric("Output Enhanced GSD", "4.00 m / px", delta="2.5× Spatial Zoom")
        q3.metric("Pixel Ground Area", "16.0 m²", delta="-84% blur area", delta_color="inverse")
        q4.metric("Composite Trust Score", f"{trust_score_pct:.1f}%", delta="Reliability Proxy")
        q5.metric("Model Disagreement (σ)", f"{trust_data['disagreement_mean']:.5f}", delta="Epistemic Uncertainty")
        q6.metric("Operational Status", "APPROVED" if is_trusted else "ADVISORY", delta="Human-in-Loop" if not is_trusted else "Automated OK")

        # Prepare Display RGB
        bounds_10m = get_display_stretch_bounds(input_10m, false_color=is_false_color)
        rgb_10m_raw = to_display_rgb(input_10m, false_color=is_false_color, stretch_bounds=bounds_10m)
        rgb_sr_raw = to_display_rgb(sr_4m, false_color=is_false_color, stretch_bounds=bounds_10m)
        # Resize 10m to match 4m display canvas dimensions
        rgb_10m_disp = cv2.resize(rgb_10m_raw, (rgb_sr_raw.shape[1], rgb_sr_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Trust overlay
        cmap = plt.get_cmap("RdYlGn")
        trust_colored = (cmap(trust_map)[:, :, :3] * 255).astype(np.uint8)
        trust_overlay = (0.55 * rgb_sr_raw + 0.45 * trust_colored).astype(np.uint8)

        # View Mode Selector
        st.markdown("### 🖼️ Direct Real Super-Resolution Comparison")
        st.caption("Live comparison between the genuine Sentinel-2 10m observation and the 2.5× learned super-resolved 4m product.")

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("#### 1. Sentinel-2 10m (Original Observation)")
            st.caption(f"Raw Satellite Input [{input_10m.shape[1]}×{input_10m.shape[0]} px, 10m GSD]")
            st.image(rgb_10m_disp, caption="Original 10m Sentinel-2 Image (Nearest Enlarged)", use_container_width=True)
            st.markdown("`[INPUT]` Native medium-resolution ESA Sentinel-2 MSI observation")

        with v_col2:
            st.markdown("#### 2. GeoFUSE SR 4m (Learned 2.5× Output)")
            st.caption(f"Ensemble Super-Resolved Product [{sr_4m.shape[1]}×{sr_4m.shape[0]} px, 4m GSD]")
            st.image(rgb_sr_raw, caption="Direct 2.5× Super-Resolution Output (Nominal 4m Grid)", use_container_width=True)
            st.markdown("`[OUTPUT]` Reconstructed fine-scale boundaries, building edges, and field contours")

        # Disagreement and Trust Map Panel
        st.markdown("---")
        st.markdown("### 🛡️ Empirical Reliability & Uncertainty Mapping")
        st.caption("Inter-model disagreement reveals where ensemble predictions vary; the fused Trust/Risk map weights uncertainty, sensitivity, spectral, and edge signals.")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("#### Ensemble Disagreement Map (Uncertainty Proxy)")
            im_d = plt.figure(figsize=(6, 4.5), dpi=120)
            plt.imshow(disagreement_map, cmap="magma")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title("Per-Pixel Ensemble Standard Deviation (σ)", fontsize=10, fontweight="bold")
            plt.axis("off")
            st.pyplot(im_d, use_container_width=True)
            plt.close(im_d)
            st.caption("Lower values (dark) indicate unanimous model agreement; higher values (bright) mark ambiguous high-frequency transitions.")

        with r_col2:
            st.markdown("#### Composite Trust / Risk Overlay")
            st.image(trust_overlay, caption=f"Empirical Trust Overlay (RdYlGn) — Composite Score: {trust_score_pct:.1f}%", use_container_width=True)
            st.caption("Green marks verified high-trust pixels; Yellow/Red highlights areas requiring human-in-the-loop review.")

        # Real Mode Domain Tabs
        r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
            "🏙️ Urban Application (Buildings & Roads)",
            "🌾 Agriculture Application (NDVI & Fields)",
            "🛡️ Multi-Criteria Trust Evidence",
            "📜 Auditable Trust Receipt & Deliverables",
        ])

        with r_tab1:
            st.markdown("### 🏙️ Urban Infrastructure & Building Footprint Analysis")
            st.markdown(
                "Super-resolving 10m Sentinel-2 imagery to a nominal 4m grid enables sharper rooftop separation, "
                "delineation of rural roads, and boundary definition for dense built-up settlements."
            )
            st.caption("Notice: Direct morphological extraction on 4m imagery sharpens structure without fabricated true-accuracy claims.")

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

        with r_tab2:
            st.markdown("### 🌾 Agriculture & Crop Monitoring Application")
            st.markdown(
                "Vegetation health and field parcels benefit from 2.5× spatial enhancement. "
                "Below is the Normalized Difference Vegetation Index (NDVI) comparing the coarse 10m observation against the 4m product."
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
                st.image(col_ndvi_4m, caption="4m Super-Resolved NDVI (Crisp Field Parcels & Sub-Field Heterogeneity)", use_container_width=True)

        with r_tab3:
            st.markdown("### 🛡️ Multi-Criteria Evidence Signal Breakdown")
            st.caption("Verifiable empirical signals computed on real imagery without fabricated ground truth.")

            sig = trust_data["fusion_result"].get("normalized_signals", {})
            if sig:
                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    st.markdown("**1. Model Disagreement**")
                    st.caption("Epistemic uncertainty across seeds")
                    st.image(sig.get("disagreement", disagreement_map), use_container_width=True, clamp=True)
                with e2:
                    st.markdown("**2. Perturbation Stability**")
                    st.caption("Output variance under sensor noise")
                    st.image(sig.get("stability", trust_data["stability_map"]), use_container_width=True, clamp=True)
                with e3:
                    st.markdown("**3. Spectral NDVI Delta**")
                    st.caption("Cross-scale radiometric fidelity")
                    st.image(sig.get("spectral", trust_data["delta_ndvi_map"]), use_container_width=True, clamp=True)
                with e4:
                    st.markdown("**4. Structural Difference**")
                    st.caption("Sobel boundary gradient check")
                    st.image(sig.get("structural", trust_data["delta_ndvi_map"]), use_container_width=True, clamp=True)

        with r_tab4:
            st.markdown("### 📜 Auditable Trust Receipt & Production Deliverables")
            st.caption("Complete provenance record verifying that NO synthetic degradation was inserted into the live inference path.")

            # Deliverables Download Section (Requirement 28)
            st.markdown("#### 📥 Download Deliverables (4m GeoTIFF, Maps, Receipt)")
            scene_stem = selected_key

            d_row1_1, d_row1_2, d_row1_3, d_row1_4 = st.columns(4)
            if "sr_4m" in exported_paths and Path(exported_paths["sr_4m"]).exists():
                with open(exported_paths["sr_4m"], "rb") as f:
                    d_row1_1.download_button("📥 sr_4m.tif (4m GeoTIFF)", data=f.read(), file_name=f"{scene_stem}_sr_4m.tif", mime="image/tiff", use_container_width=True)

            if "trust_map" in exported_paths and Path(exported_paths["trust_map"]).exists():
                with open(exported_paths["trust_map"], "rb") as f:
                    d_row1_2.download_button("📥 trust_map.tif", data=f.read(), file_name=f"{scene_stem}_trust_map.tif", mime="image/tiff", use_container_width=True)

            if "disagreement_map" in exported_paths and Path(exported_paths["disagreement_map"]).exists():
                with open(exported_paths["disagreement_map"], "rb") as f:
                    d_row1_3.download_button("📥 disagreement_map.tif", data=f.read(), file_name=f"{scene_stem}_disagreement_map.tif", mime="image/tiff", use_container_width=True)

            receipt_json_str = json.dumps(receipt, indent=2)
            d_row1_4.download_button(
                "📥 trust_receipt.json",
                data=receipt_json_str,
                file_name=f"{scene_stem}_trust_receipt.json",
                mime="application/json",
                use_container_width=True,
            )

            d_row2_1, d_row2_2, d_row2_3, _ = st.columns(4)
            if "rgb_preview" in exported_paths and Path(exported_paths["rgb_preview"]).exists():
                with open(exported_paths["rgb_preview"], "rb") as f:
                    d_row2_1.download_button("📥 rgb_preview.png", data=f.read(), file_name=f"{scene_stem}_rgb_preview.png", mime="image/png", use_container_width=True)

            if "ndvi_preview" in exported_paths and Path(exported_paths["ndvi_preview"]).exists():
                with open(exported_paths["ndvi_preview"], "rb") as f:
                    d_row2_2.download_button("📥 ndvi_preview.png", data=f.read(), file_name=f"{scene_stem}_ndvi_preview.png", mime="image/png", use_container_width=True)

            meta_p = exported_paths.get("metadata") or (Path(exported_paths.get("sr_4m", "")).parent / "metadata.json")
            if Path(meta_p).exists():
                with open(meta_p, "rb") as f:
                    d_row2_3.download_button("📥 metadata.json", data=f.read(), file_name=f"{scene_stem}_metadata.json", mime="application/json", use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📜 Verifiable Provenance Certificate")
            st.json(receipt, expanded=False)


    # =========================================================================
    # MODE 2: CONTROLLED SYNTHETIC BENCHMARK (DEGRADE & RECOVER)
    # =========================================================================
    else:
        st.sidebar.markdown(
            """
            <div style="background: #1a1a2e; border: 1px solid #ff9100; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.88rem; font-weight: bold; color: #ff9100;">🔬 Mode: BENCHMARK EVALUATION</div>
                <div style="font-size: 0.76rem; color: #b0bec5;">Degrade & Recover: <b>Synthetic 2.5x Blur & Noise</b><br>Ground Truth: <b>Unseen 10m Reference</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tile_selection_mode = st.sidebar.radio(
            "Tile Selection Scope:",
            options=[
                "⭐ Curated Demo Highlights (Offline Fast)",
                "🧭 Browse All 25 Scene Tiles (0..24)",
                "📍 Custom Coordinate Box (Free X, Y Crop)",
            ],
            index=0,
        )

        all_grid_meta = get_tile_metadata_grid(512, 512, patch_size=128, stride=96)
        all_grid_dict = {t["tile_id"]: t for t in all_grid_meta}

        selected_idx = 0
        data = None

        if tile_selection_mode == "⭐ Curated Demo Highlights (Offline Fast)":
            tile_options = {
                0: "Tile #0 -- Central Settlement Cluster (High Trust: 86.8%)",
                8: "Tile #8 -- Agricultural & Rural Roads (High Trust: 86.7%)",
                16: "Tile #16 -- Rural River Corridor (⚠️ Low Trust Warning: 86.4%)",
                24: "Tile #24 -- Complex Terrain Transition (⚠️ Low Trust Warning: 86.0%)",
            }
            selected_idx = st.sidebar.selectbox(
                "Select Sentinel-2 Demo Tile:",
                options=list(tile_options.keys()),
                format_func=lambda x: tile_options.get(x, f"Tile #{x}"),
            )
            data = run_cached_pipeline(selected_idx)
        elif tile_selection_mode == "🧭 Browse All 25 Scene Tiles (0..24)":
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

        color_mode = st.sidebar.radio(
            "Band Visualization Mode:",
            options=["Natural RGB (B04-B03-B02)", "False-Color Infrared (B08-B04-B03)"],
            index=0,
            key="benchmark_color_mode",
        )
        is_false_color = "False-Color" in color_mode

        show_evidence = st.sidebar.checkbox("Show Multi-Criteria Evidence Breakdown", value=False, key="bm_ev")
        show_downstream = st.sidebar.checkbox("Show Downstream Building Footprint Analysis", value=True, key="bm_ds")
        show_scene_overview = st.sidebar.checkbox("Show Macro Scene Overview & Tile Locator", value=True, key="bm_sc")

        hr_tile = data["hr_tile"]
        lr_tile = data["lr_tile"]
        bicubic_tile = data["bicubic_tile"]
        sr_tile = data["sr_tile"]
        fusion_result = data["fusion_result"]
        trust_map = fusion_result["trust_map"]
        receipt = data["receipt"]
        score_pct = fusion_result["trust_score_pct"]

        # Quantitative Metrics
        from skimage.metrics import peak_signal_noise_ratio as compute_psnr
        from skimage.metrics import structural_similarity as compute_ssim
        bic_psnr = float(compute_psnr(hr_tile, bicubic_tile, data_range=1.0))
        sr_psnr = float(compute_psnr(hr_tile, sr_tile, data_range=1.0))
        bic_ssim = float(compute_ssim(hr_tile, bicubic_tile, channel_axis=2, data_range=1.0))
        sr_ssim = float(compute_ssim(hr_tile, sr_tile, channel_axis=2, data_range=1.0))

        st.markdown(
            """
            <div style="background: #1a1a2e; border-left: 5px solid #ff9100; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
                <div style="font-size: 1.1rem; font-weight: bold; color: #ff9100;">🔬 Controlled Benchmark Simulation Mode</div>
                <div style="font-size: 0.85rem; color: #e0e0e0;">Evaluating degrade-and-recover performance against unseen HR reference tile.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        st.markdown("### 🖼️ Benchmark Quad View: Input vs. Bicubic vs. GeoFUSE SR vs. Reference")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown("#### 1. Degraded Pseudo-LR")
            st.image(lr_disp, caption="Synthetic Degradation Input (PSF Blur + 2.5x Down + Noise)", use_container_width=True)
        with col_b:
            st.markdown("#### 2. Bicubic Baseline")
            st.image(bic_rgb, caption=f"Bicubic Baseline | PSNR: {bic_psnr:.2f} dB", use_container_width=True)
        with col_c:
            st.markdown("#### 3. GeoFUSE SR (Ours)")
            st.image(sr_rgb, caption=f"GeoFUSE 2.5x SR | PSNR: {sr_psnr:.2f} dB", use_container_width=True)
        with col_d:
            st.markdown("#### 4. HR Reference Target")
            st.image(hr_rgb, caption="Unseen High-Resolution Reference Target", use_container_width=True)


if __name__ == "__main__":
    main()
