"""GeoFUSE Direct Real Sentinel-2 Super-Resolution Runner.

Executes direct 2.5× learned super-resolution on genuine 10m Sentinel-2 multi-spectral imagery
(B02, B03, B04, B08) with ZERO synthetic degradation in the real inference path.
Produces a nominal 4m super-resolved GeoTIFF product with empirical uncertainty and trust receipts.

Also provides a controlled benchmark mode for degrade-and-recover evaluation against reference targets.

Usage:
    python scripts/run_custom_input.py --input path/to/sentinel_scene --mode real
    python scripts/run_custom_input.py --input path/to/benchmark_tile.tif --mode benchmark
"""


import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import (
    bicubic_upsample,
    evaluate_reconstruction_fidelity,
    synthesize_pseudo_lr,
)
from src.data.tiling import extract_tiles
from src.evaluation.downstream_eval import (
    compare_downstream_footprints,
    extract_building_footprints,
)
from src.evaluation.edge_check import (
    compute_edge_consistency,
    compute_gradient_magnitude,
)
from src.evaluation.fusion import fuse_trust_risk_maps, render_trust_risk_overlay
from src.evaluation.spectral_check import compute_ndvi, compute_spectral_consistency
from src.evaluation.stability import compute_stability_map

from src.evaluation.trust_receipt import generate_trust_receipt
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def match_sentinel_bands(paths: List[Path]) -> Dict[str, Path]:
    """Identify B02, B03, B04, B08 band files from a list of paths."""
    band_aliases = {
        "B02": ["B02", "B2", "BLUE", "BAND2", "BAND_2", "BAND02"],
        "B03": ["B03", "B3", "GREEN", "BAND3", "BAND_3", "BAND03"],
        "B04": ["B04", "B4", "RED", "BAND4", "BAND_4", "BAND04"],
        "B08": ["B08", "B8", "NIR", "BAND8", "BAND_8", "BAND08", "B8A"],
    }
    matched: Dict[str, Path] = {}
    assigned_paths = set()

    for band_key, aliases in band_aliases.items():
        for p in paths:
            if p in assigned_paths:
                continue
            stem = p.stem.upper()
            tokens = [t for t in stem.replace("-", "_").replace(".", "_").replace(" ", "_").split("_") if t]

            # 1. Exact token match (e.g. "B02", "BLUE", "B2")
            found = False
            for alias in aliases:
                if alias in tokens or any(t == alias for t in tokens):
                    matched[band_key] = p
                    assigned_paths.add(p)
                    found = True
                    break
            if found:
                continue

            # 2. Substring match (e.g. "_B02_" or "B02_10M" or "BAND2")
            for alias in aliases:
                if f"_{alias}" in stem or f"{alias}_" in stem or stem.endswith(alias) or stem.startswith(alias):
                    matched[band_key] = p
                    assigned_paths.add(p)
                    break

    # If exactly 4 files were supplied and not all 4 were matched via pattern,
    # assign the remaining in alphabetical sorted order to standard Sentinel-2 bands
    if len(matched) < 4 and len(paths) == 4:
        sorted_paths = sorted(paths, key=lambda x: x.name.lower())
        band_keys = ["B02", "B03", "B04", "B08"]
        for i, b_key in enumerate(band_keys):
            if b_key not in matched:
                for sp in sorted_paths:
                    if sp not in matched.values():
                        matched[b_key] = sp
                        break

    return matched


def load_user_image(
    image_input: Union[str, Path, List[Union[str, Path]]],
    target_size: int = 128,
    tile_idx: Optional[int] = None,
    crop_coords: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load custom user image inputs and format into a unified (H, W, 4) reflectance array.

    Supports:
    1. Multiple band files (e.g. [B02.tif, B03.tif, B04.tif, B08.tif] uploaded together).
    2. A directory containing separate Sentinel-2 band GeoTIFFs.
    3. A single multi-band GeoTIFF (>= 4 bands).
    4. A standard 3-band RGB image (PNG, JPG, BMP) with synthesized NIR proxy.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - 4-band array of shape (target_size, target_size, 4), dtype float32, in [0.0, 1.0].
            - Metadata dictionary describing the matched bands and source attributes.
    """
    meta: Dict[str, Any] = {
        "is_multi_band_stack": False,
        "matched_bands": {},
    }

    # Normalize input into a list of Path objects or single Path
    candidate_paths: List[Path] = []
    if isinstance(image_input, (list, tuple)):
        candidate_paths = [Path(p) for p in image_input if Path(p).exists()]
    else:
        single_path = Path(image_input)
        if not single_path.exists():
            raise FileNotFoundError(f"Custom input not found at: {single_path.resolve()}")
        if single_path.is_dir():
            # Directory containing band files
            for ext in ("*.tif", "*.tiff", "*.jp2", "*.TIF", "*.png", "*.jpg", "*.jpeg"):
                candidate_paths.extend(single_path.glob(ext))
        else:
            candidate_paths = [single_path]

    arr: Optional[np.ndarray] = None

    # CASE 1: Multiple band files or directory of bands provided
    if len(candidate_paths) >= 4:
        matched = match_sentinel_bands(candidate_paths)
        if all(b in matched for b in ["B02", "B03", "B04", "B08"]):
            band_arrays = []
            for b in ["B02", "B03", "B04", "B08"]:
                band_path = matched[b]
                b_data = None
                try:
                    with rasterio.open(band_path) as src:
                        b_data = src.read(1).astype(np.float32)
                except Exception:
                    pass

                if b_data is None:
                    # Fallback for standard image files (PNG/JPG)
                    img_read = cv2.imread(str(band_path), cv2.IMREAD_UNCHANGED)
                    if img_read is not None:
                        if img_read.ndim == 3:
                            b_data = img_read[:, :, 0].astype(np.float32)
                        else:
                            b_data = img_read.astype(np.float32)

                if b_data is None:
                    raise ValueError(f"Could not read band image data: {band_path.name}")

                band_arrays.append(b_data)

            # Validate dimensions match
            h0, w0 = band_arrays[0].shape
            for i in range(1, len(band_arrays)):
                if band_arrays[i].shape != (h0, w0):
                    band_arrays[i] = cv2.resize(band_arrays[i], (w0, h0), interpolation=cv2.INTER_AREA)

            arr = np.stack(band_arrays, axis=-1)
            # Normalize Sentinel-2 raw digital numbers (DN > 10) to [0.0, 1.0] reflectance
            if np.max(arr) > 10.0:
                arr = arr / 10000.0 if np.max(arr) > 255.0 else arr / 255.0
            arr = np.clip(arr, 0.0, 1.2).astype(np.float32)

            meta["is_multi_band_stack"] = True
            meta["is_geotiff"] = any(matched[b].suffix.lower() in [".tif", ".tiff", ".jp2"] for b in matched)
            meta["matched_bands"] = {b: matched[b].name for b in ["B02", "B03", "B04", "B08"]}
            meta["filename"] = f"4-Band Multi-Spectral Stack ({matched['B04'].stem})"

    # CASE 2: Single file input (Multi-band GeoTIFF or standard RGB image)
    if arr is None and len(candidate_paths) == 1:
        path = candidate_paths[0]
        meta["filename"] = path.name
        meta["suffix"] = path.suffix.lower()
        meta["is_geotiff"] = False

        if path.suffix.lower() in [".tif", ".tiff", ".jp2"]:
            try:
                with rasterio.open(path) as src:
                    count = src.count
                    meta["crs"] = str(src.crs)
                    meta["is_geotiff"] = True
                    meta["source_bands"] = count
                    meta["source_shape"] = (src.height, src.width)

                    if count >= 4:
                        # True multi-spectral (read first 4 bands: B02, B03, B04, B08)
                        bands = [src.read(i + 1).astype(np.float32) for i in range(4)]
                        arr = np.stack(bands, axis=-1)
                    elif count == 3:
                        # RGB GeoTIFF
                        r = src.read(1).astype(np.float32)
                        g = src.read(2).astype(np.float32)
                        b = src.read(3).astype(np.float32)
                        nir = np.clip(1.2 * g - 0.2 * r, 0.05, None)
                        arr = np.stack([b, g, r, nir], axis=-1)
                    else:
                        gray = src.read(1).astype(np.float32)
                        arr = np.stack([gray, gray, gray, gray], axis=-1)

                if np.max(arr) > 10.0:
                    arr = arr / 10000.0 if np.max(arr) > 255.0 else arr / 255.0
                arr = np.clip(arr, 0.0, 1.2).astype(np.float32)
            except Exception:
                arr = None

        if arr is None:
            img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise ValueError(f"Could not decode image file: {path.name}")

            meta["source_shape"] = img_bgr.shape[:2]
            if img_bgr.ndim == 2:
                gray = (img_bgr.astype(np.float32) / 255.0)
                arr = np.stack([gray, gray, gray, gray], axis=-1)
            elif img_bgr.shape[2] == 4:
                b = img_bgr[:, :, 0].astype(np.float32) / 255.0
                g = img_bgr[:, :, 1].astype(np.float32) / 255.0
                r = img_bgr[:, :, 2].astype(np.float32) / 255.0
                nir = img_bgr[:, :, 3].astype(np.float32) / 255.0
                arr = np.stack([b, g, r, nir], axis=-1)
            else:
                b = img_bgr[:, :, 0].astype(np.float32) / 255.0
                g = img_bgr[:, :, 1].astype(np.float32) / 255.0
                r = img_bgr[:, :, 2].astype(np.float32) / 255.0
                nir = np.clip(1.25 * g - 0.25 * r, 0.05, 1.0).astype(np.float32)
                arr = np.stack([b, g, r, nir], axis=-1)

    if arr is None:
        raise ValueError(f"Unable to process input: provided {len(candidate_paths)} files but could not match 4 Sentinel bands or decode as an image.")

    # Record Full Scene before cropping
    h, w, c = arr.shape
    meta["full_image"] = arr.copy()
    meta["full_shape"] = (h, w)

    tile_w = min(target_size, w)
    tile_h = min(target_size, h)

    if h >= target_size and w >= target_size:
        if crop_coords is not None:
            cx, cy = crop_coords
            start_x = max(0, min(int(cx), w - target_size))
            start_y = max(0, min(int(cy), h - target_size))
            arr_tile = arr[start_y : start_y + target_size, start_x : start_x + target_size, :]
            meta["selected_mode"] = "custom_coords"
        elif tile_idx is not None:
            tiles = extract_tiles(arr, patch_size=target_size, stride=96)
            if 0 <= tile_idx < len(tiles):
                t_info = tiles[tile_idx]
                start_x = t_info["x"]
                start_y = t_info["y"]
                arr_tile = t_info["data"]
                meta["tile_idx"] = tile_idx
                meta["selected_mode"] = "tile_grid"
            else:
                start_y = (h - target_size) // 2
                start_x = (w - target_size) // 2
                arr_tile = arr[start_y : start_y + target_size, start_x : start_x + target_size, :]
        else:
            start_y = (h - target_size) // 2
            start_x = (w - target_size) // 2
            arr_tile = arr[start_y : start_y + target_size, start_x : start_x + target_size, :]
    else:
        start_x = 0
        start_y = 0
        arr_tile = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)

    meta["crop_box"] = {"x": start_x, "y": start_y, "w": tile_w, "h": tile_h}
    return arr_tile, meta


def run_custom_pipeline(
    input_image_path: Union[str, Path, List[Union[str, Path]]],
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = "real",
    save_receipt: bool = True,
    save_plot: bool = True,
    tile_idx: Optional[int] = None,
    crop_coords: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Execute Sentinel-2 Super-Resolution inference.

    Modes:
    1. 'real' (DEFAULT): Direct real Sentinel-2 10m -> 4m (2.5x) super-resolution with
       ZERO synthetic degradation. Feeds the 10m observation directly to the ensemble.
    2. 'benchmark': Controlled degrade-and-recover experiment (synthetic PSF blur + 2.5x
       downsampling + sensor noise -> SR -> PSNR/SSIM evaluation against reference).

    Args:
        input_image_path: Image file path, directory of bands, or list of band files.
        output_dir: Destination directory.
        mode: 'real' (direct live SR) or 'benchmark' (degrade-and-recover).
        save_receipt: Save JSON trust receipt.
        save_plot: Save side-by-side inspection plot.
        tile_idx: Optional tile partition index (0..24) to crop from scene.
        crop_coords: Optional (crop_x, crop_y) pixel offset for custom crop.

    Returns:
        Dict[str, Any]: Comprehensive results dictionary.
    """
    root = get_project_root()
    config = load_config()
    device = get_device(config)

    if output_dir is None:
        if mode == "real":
            out_path = root / "outputs" / "real_inference"
        else:
            out_path = root / "outputs" / "custom_runs"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"   GeoFUSE SentinelGuard -- Mode: {mode.upper()}")
    print("=" * 80)
    print(f"Input image(s)  : {input_image_path}")
    print(f"Output directory: {out_path.resolve()}")
    print(f"Compute device  : {device}")
    print(f"Synthetic Blur  : {'DISABLED (Direct Real Inference)' if mode == 'real' else 'ENABLED (Benchmark Simulation)'}")

    # =========================================================================
    # BRANCH A: DIRECT REAL SENTINEL-2 INFERENCE (ZERO SYNTHETIC DEGRADATION)
    # =========================================================================
    if mode == "real":
        from src.inference.real_inference import (
            compute_real_inference_trust,
            export_real_inference_products,
            generate_real_trust_receipt,
            load_and_preprocess_sentinel2,
            run_direct_sr_scene,
            run_direct_sr_tile,
        )

        # 1. Load real Sentinel-2 10m imagery
        scene_10m, val_info = load_and_preprocess_sentinel2(input_image_path)
        h0, w0 = scene_10m.shape[:2]

        if isinstance(input_image_path, (list, tuple)):
            base_stem = Path(input_image_path[0]).stem.split("_")[0] or "sentinel2_real"
        elif Path(input_image_path).is_dir():
            base_stem = Path(input_image_path).name
        else:
            base_stem = Path(input_image_path).stem

        # Crop if requested
        if crop_coords is not None:
            cx, cy = crop_coords
            start_x = max(0, min(int(cx), w0 - 64))
            start_y = max(0, min(int(cy), h0 - 64))
            sub_10m = scene_10m[start_y : start_y + 128, start_x : start_x + 128, :]
        elif tile_idx is not None:
            tiles = extract_tiles(scene_10m, patch_size=64, stride=48)
            t_idx = min(max(0, tile_idx), len(tiles) - 1)
            sub_10m = tiles[t_idx]["data"]
        else:
            sub_10m = scene_10m

        print(f"  [1/4] Loaded Sentinel-2 10m scene: shape={sub_10m.shape}, range=[{sub_10m.min():.3f}, {sub_10m.max():.3f}]")

        # 2. Load trained ensemble models
        ckpt_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
        if not (ckpt_dir / "ensemble_member_0.pth").exists():
            ckpt_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
        ckpt_paths = [ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)]
        models = load_ensemble_members(ckpt_paths, config=config, device=device)

        # 3. Direct 2.5x SR (NO SYNTHETIC DEGRADATION!)
        scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))
        if sub_10m.shape[0] <= 128 and sub_10m.shape[1] <= 128:
            sr_4m, disagreement_map = run_direct_sr_tile(models, sub_10m, device=device)
        else:
            sr_4m, disagreement_map = run_direct_sr_scene(models, sub_10m, scale_factor=scale_factor, device=device)

        print(f"  [2/4] Direct 2.5x Super-Resolution completed: 10m {sub_10m.shape[:2]} -> 4m {sr_4m.shape[:2]}")

        # 4. Multi-Criteria Trust Verification
        trust_data = compute_real_inference_trust(
            sr_4m=sr_4m,
            lr_10m=sub_10m,
            disagreement_map=disagreement_map,
            models=models,
            device=device,
            config=config,
        )
        fusion_result = trust_data["fusion_result"]
        trust_score_pct = trust_data["trust_score_pct"]
        is_trusted = trust_data["is_trusted"]
        status_label = "HIGH_TRUST_APPROVED" if is_trusted else "LOW_TRUST_ADVISORY"

        print(f"  [3/4] Trust score evaluated: {trust_score_pct:.2f}% [{status_label}] (Mean Disagreement: {trust_data['disagreement_mean']:.6f})")

        # 5. Auditable Trust Receipt
        dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        receipt = generate_real_trust_receipt(
            scene_name=base_stem,
            val_info=val_info,
            trust_data=trust_data,
            config=config,
            device_name=dev_name,
        )

        # 6. Export Products
        exported_paths = export_real_inference_products(
            output_dir=out_path,
            scene_name=base_stem,
            sr_4m=sr_4m,
            lr_10m=sub_10m,
            disagreement_map=disagreement_map,
            trust_map=fusion_result["trust_map"],
            val_info=val_info,
            receipt=receipt,
        )
        print(f"  [4/4] Output products exported to: {out_path / base_stem}")
        print(f"        -> sr_4m.tif (4m GeoTIFF, Transform & CRS updated)")
        print(f"        -> trust_receipt.json (synthetic_degradation: false)")

        plot_path = exported_paths.get("rgb_preview")
        receipt_path = exported_paths.get("trust_receipt")

        # Visual side-by-side inspection figure
        if save_plot:
            fig_path = out_path / base_stem / f"{base_stem}_side_by_side.png"
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)

            def to_rgb(t):
                r, g, b = t[:, :, 2], t[:, :, 1], t[:, :, 0]
                rgb = np.stack([r, g, b], axis=-1)
                v_min, v_max = np.percentile(rgb, (2, 98))
                if v_max > v_min:
                    rgb = np.clip((rgb - v_min) / (v_max - v_min), 0.0, 1.0)
                return rgb

            orig_rgb = to_rgb(sub_10m)
            sr_rgb = to_rgb(sr_4m)
            orig_disp = cv2.resize(orig_rgb, (sr_4m.shape[1], sr_4m.shape[0]), interpolation=cv2.INTER_NEAREST)

            cmap = plt.get_cmap("RdYlGn")
            trust_colored = (cmap(fusion_result["trust_map"])[:, :, :3])
            trust_overlay = np.clip(0.55 * sr_rgb + 0.45 * trust_colored, 0.0, 1.0)

            axes[0].imshow(orig_disp)
            axes[0].set_title(f"1. Sentinel-2 10m Input\n[{base_stem[:25]}]", fontsize=10, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(sr_rgb)
            axes[1].set_title("2. GeoFUSE SR 4m (Learned 2.5×)\n[Direct Model Inference]", fontsize=10, fontweight="bold", color="#2ea043")
            axes[1].axis("off")

            im_disag = axes[2].imshow(disagreement_map, cmap="magma")
            axes[2].set_title(f"3. Ensemble Disagreement Map\n[Uncertainty Proxy, Mean: {trust_data['disagreement_mean']:.5f}]", fontsize=10, fontweight="bold")
            axes[2].axis("off")
            plt.colorbar(im_disag, ax=axes[2], fraction=0.046, pad=0.04)

            axes[3].imshow(trust_overlay)
            axes[3].set_title(f"4. Composite Trust / Risk Map\nTrust Score: {trust_score_pct:.2f}% [{status_label}]", fontsize=10, fontweight="bold")
            axes[3].axis("off")

            plt.tight_layout()
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            plot_path = fig_path

        # Downstream footprint extraction on real 4m product
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

        ndvi_10m = compute_ndvi(sub_10m)
        ndvi_4m = compute_ndvi(sr_4m)

        return {
            "mode": "real",
            "input_scene": sub_10m,
            "sr_scene": sr_4m,
            "input_10m": sub_10m,
            "sr_4m": sr_4m,
            "hr_tile": sub_10m,  # backward-compatibility alias
            "lr_tile": sub_10m,  # real 10m input is the LR observation
            "sr_tile": sr_4m,
            "disagreement_map": disagreement_map,
            "stability_map": trust_data["stability_map"],
            "trust_map": fusion_result["trust_map"],
            "spectral_metrics": {
                "delta_ndvi_mean": trust_data["delta_ndvi_mean"],
                "delta_ndvi_map": trust_data["delta_ndvi_map"],
            },
            "edge_metrics": {
                "structural_diff_mean": trust_data["structural_diff_mean"],
            },
            "fusion_result": fusion_result,
            "urban_analysis": {
                "footprints": foot_sr,
                "footprint_pixels": foot_sr["footprint_pixels"],
            },
            "agriculture_analysis": {
                "ndvi_10m": ndvi_10m,
                "ndvi_4m": ndvi_4m,
                "mean_ndvi_4m": float(np.mean(ndvi_4m)),
            },
            "foot_sr": foot_sr,
            "receipt": receipt,
            "input_meta": val_info,
            "output_meta": {
                "shape": list(sr_4m.shape),
                "input_shape": list(sub_10m.shape),
                "scale_factor": scale_factor,
                "input_gsd_m": 10.0,
                "output_gsd_m": 4.0,
                "crs": str(val_info.get("crs")),
            },
            "output_paths": exported_paths,
            "exported_paths": exported_paths,
            "trust_score_pct": trust_score_pct,
            "mean_risk_score": fusion_result["mean_risk_score"],
            "status": status_label,
            "is_trusted": is_trusted,
            "sr_psnr": "N/A (Real Scene)",
            "bicubic_psnr": "N/A (Real Scene)",
            "synthetic_degradation_used": False,
            "plot_path": str(plot_path) if plot_path else None,
            "receipt_path": str(receipt_path) if receipt_path else None,
        }


    # =========================================================================
    # BRANCH B: CONTROLLED SYNTHETIC BENCHMARK (DEGRADE & RECOVER)
    # =========================================================================
    hr_tile, input_meta = load_user_image(
        input_image_path,
        target_size=128,
        tile_idx=tile_idx,
        crop_coords=crop_coords,
    )

    scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))
    lr_tile = synthesize_pseudo_lr(
        hr_tile,
        downsample_factor=scale_factor,
        blur_kernel_size=3,
        blur_sigma=0.5,
        noise_std=0.01,
        seed=42,
    )
    bicubic_tile = bicubic_upsample(lr_tile, scale_factor=scale_factor, target_shape=hr_tile.shape[:2])

    ckpt_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (ckpt_dir / "ensemble_member_0.pth").exists():
        ckpt_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    ckpt_paths = [ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)]
    models = load_ensemble_members(ckpt_paths, config=config, device=device)

    sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)
    if sr_tile.shape[:2] != hr_tile.shape[:2]:
        sr_tile = cv2.resize(sr_tile, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)
        disagreement_map = cv2.resize(disagreement_map, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)

    fid_sr = evaluate_reconstruction_fidelity(hr_tile, sr_tile)
    fid_bic = evaluate_reconstruction_fidelity(hr_tile, bicubic_tile)

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

    spec_cfg = config.get("verification", {}).get("spectral_consistency", {})
    spectral_metrics = compute_spectral_consistency(
        gt_tile=hr_tile,
        sr_tile=sr_tile,
        ndvi_threshold=float(spec_cfg.get("ndvi_threshold", 0.05)),
        red_idx=2,
        nir_idx=3,
        sensor_noise_floor=float(spec_cfg.get("sensor_noise_floor", 0.03)),
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

    min_thresh = float(config.get("trust_receipt", {}).get("min_trust_score_threshold", 86.5))
    is_trusted = bool(fusion_result["trust_score_pct"] >= min_thresh)
    status_label = "HIGH_TRUST_APPROVED" if is_trusted else "LOW_TRUST_WARNING"

    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})
    foot_bic = extract_building_footprints(
        bicubic_tile,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=2,
        nir_idx=3,
    )
    foot_sr = extract_building_footprints(
        sr_tile,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=2,
        nir_idx=3,
    )
    foot_ref = extract_building_footprints(
        hr_tile,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=2,
        nir_idx=3,
    )
    downstream_comp = compare_downstream_footprints(
        mask_bicubic=foot_bic["mask"],
        mask_sr=foot_sr["mask"],
        trust_map=fusion_result["trust_map"],
        ref_mask=foot_ref["mask"],
        trust_threshold=float(down_cfg.get("trust_partition_threshold", 0.85)),
    )

    if isinstance(input_image_path, (list, tuple)):
        base_stem = Path(input_image_path[0]).stem
    else:
        base_stem = Path(input_image_path).stem

    plot_path = None
    if save_plot:
        plot_path = out_path / f"{base_stem}_inspection.png"
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=150)

        def to_rgb(t):
            r, g, b = t[:, :, 2], t[:, :, 1], t[:, :, 0]
            rgb = np.stack([r, g, b], axis=-1)
            v_min, v_max = np.percentile(rgb, (2, 98))
            if v_max > v_min:
                rgb = np.clip((rgb - v_min) / (v_max - v_min), 0.0, 1.0)
            return rgb

        hr_rgb = to_rgb(hr_tile)
        bic_rgb = to_rgb(bicubic_tile)
        sr_rgb = to_rgb(sr_tile)
        lr_display = cv2.resize(to_rgb(lr_tile), (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_NEAREST)

        cmap = plt.get_cmap("RdYlGn")
        trust_colored = (cmap(fusion_result["trust_map"])[:, :, :3])
        trust_overlay = np.clip(0.55 * sr_rgb + 0.45 * trust_colored, 0.0, 1.0)

        display_name = input_meta.get("filename", base_stem)
        axes[0].imshow(hr_rgb)
        axes[0].set_title(f"1. Benchmark HR Reference\n[{display_name[:25]}]", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(lr_display)
        axes[1].set_title("2. Degraded Pseudo-LR\n[Blur + 2.5x Down + Noise]", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(bic_rgb)
        axes[2].set_title(f"3. Bicubic Baseline\nPSNR: {fid_bic['psnr_db']:.2f} dB", fontsize=10, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(sr_rgb)
        axes[3].set_title(f"4. GeoFUSE Ensemble SR\nPSNR: {fid_sr['psnr_db']:.2f} dB", fontsize=10, fontweight="bold", color="#2ea043")
        axes[3].axis("off")

        axes[4].imshow(trust_overlay)
        axes[4].set_title(f"5. Trust / Risk Map\nScore: {fusion_result['trust_score_pct']:.2f}% [{status_label}]", fontsize=10, fontweight="bold")
        axes[4].axis("off")

        plt.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

    pipeline_data = {
        "fusion_result": fusion_result,
        "spectral_metrics": spectral_metrics,
        "edge_metrics": edge_metrics,
        "downstream_comp": downstream_comp,
    }
    receipt = generate_trust_receipt(
        tile_idx=999,
        raw_dir=root / "data/raw",
        config=config,
        pipeline_data=pipeline_data,
        min_trust_threshold=min_thresh,
    )
    receipt["input_image_metadata"] = {k: v for k, v in input_meta.items() if k != "full_image"}

    receipt_path = None
    if save_receipt:
        receipt_path = out_path / f"{base_stem}_receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

    return {
        "mode": "benchmark",
        "hr_tile": hr_tile,
        "lr_tile": lr_tile,
        "bicubic_tile": bicubic_tile,
        "sr_tile": sr_tile,
        "disagreement_map": disagreement_map,
        "stability_map": stability_map,
        "structural_diff": structural_diff,
        "spectral_metrics": spectral_metrics,
        "edge_metrics": edge_metrics,
        "fusion_result": fusion_result,
        "foot_bic": foot_bic,
        "foot_sr": foot_sr,
        "foot_ref": foot_ref,
        "downstream_comp": downstream_comp,
        "receipt": receipt,
        "input_meta": input_meta,
        "full_scene": input_meta.get("full_image"),
        "crop_box": input_meta.get("crop_box"),
        "trust_score_pct": fusion_result["trust_score_pct"],
        "mean_risk_score": fusion_result["mean_risk_score"],
        "status": status_label,
        "is_trusted": is_trusted,
        "sr_psnr": fid_sr["psnr_db"],
        "bicubic_psnr": fid_bic["psnr_db"],
        "synthetic_degradation_used": True,
        "plot_path": str(plot_path) if plot_path else None,
        "receipt_path": str(receipt_path) if receipt_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description="GeoFUSE SentinelGuard — Super-Resolution Runner")
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to Sentinel-2 raster files (folder with B02/B03/B04/B08, 4 band files, or 4-band GeoTIFF)",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["real", "benchmark"],
        default="real",
        help="Inference mode: 'real' (direct 2.5x SR with zero synthetic degradation) or 'benchmark' (controlled degrade-and-recover simulation)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save output products (sr_4m.tif, trust maps, previews, receipts)",
    )
    parser.add_argument(
        "--tile", "--tile-idx",
        type=int,
        default=None,
        help="Optional tile partition index (0..24) to extract",
    )
    parser.add_argument("--crop-x", type=int, default=None, help="Optional X pixel offset for custom crop window")
    parser.add_argument("--crop-y", type=int, default=None, help="Optional Y pixel offset for custom crop window")
    parser.add_argument("--no-plot", action="store_true", default=False, help="Disable saving side-by-side visual plot")

    args = parser.parse_args()
    input_val = args.input if len(args.input) > 1 else args.input[0]

    coords = None
    if args.crop_x is not None and args.crop_y is not None:
        coords = (args.crop_x, args.crop_y)

    try:
        run_custom_pipeline(
            input_image_path=input_val,
            output_dir=args.output_dir,
            mode=args.mode,
            save_plot=not args.no_plot,
            tile_idx=args.tile,
            crop_coords=coords,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
