"""CLI script for Phase 9: Downstream Task Evaluation (Building Footprints).

Runs morphological building footprint extraction identically on:
1. Bicubic Baseline (2x)
2. Super-Resolution Ensemble Mean Reconstruction (2x)
3. Pre-degradation Reference Tile (10m)

Compares extracted footprints across High-Trust vs. Low-Trust geographic zones
using the Phase 8 Trust Map, and reports quantitative agreement statistics.
"""

import sys
from pathlib import Path

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.data.degrade import bicubic_upsample, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.downstream_eval import (
    FootprintExtractionError,
    compare_downstream_footprints,
    extract_building_footprints,
    render_downstream_overlay,
)
from src.evaluation.edge_check import compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def main() -> int:
    print("=" * 86)
    print("      GeoFUSE SentinelGuard -- Phase 9: Downstream Building Footprint Evaluation")
    print("=" * 86)

    root = get_project_root()
    config = load_config()
    device = get_device(config)

    # Downstream settings from config.yaml
    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})
    kernel_size = int(morph_cfg.get("tophat_kernel_size", 7))
    tophat_thresh = int(morph_cfg.get("tophat_threshold", 25))
    max_ndvi = float(morph_cfg.get("max_ndvi", 0.20))
    min_area = int(morph_cfg.get("min_area", 4))
    max_area = int(morph_cfg.get("max_area", 600))
    trust_thresh = float(down_cfg.get("trust_partition_threshold", 0.85))

    checkpoints_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (checkpoints_dir / "ensemble_member_0.pth").exists():
        checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    previews_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = [
        checkpoints_dir / "ensemble_member_0.pth",
        checkpoints_dir / "ensemble_member_1.pth",
        checkpoints_dir / "ensemble_member_2.pth",
    ]

    print(f"\n[1/5] Loading 3 ensemble models onto {device}...")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    print(f"      Successfully loaded {len(models)} models.")

    print("\n[2/5] Ingesting Sentinel-2 scene and extracting test tiles...")
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, meta = load_sentinel2_stack(raw_dir)
    print(f"      Scene shape: {stack.shape}, CRS: {meta.get('crs')}")

    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    patch_size = 128
    tiles = extract_tiles(stack, patch_size=patch_size, stride=96)
    sample_indices = [0, len(tiles) // 3, (2 * len(tiles)) // 3, len(tiles) - 1]
    print(f"      Selected {len(sample_indices)} geographically distinct test tiles.")

    print(f"\n[3/5] Extractor Configuration:")
    print(f"      - Kernel Size: {kernel_size}x{kernel_size} | TopHat Thresh: {tophat_thresh}")
    print(f"      - Max NDVI: {max_ndvi} | Component Area: [{min_area}, {max_area}] px")
    print(f"      - Trust Partition Threshold: {trust_thresh}")

    print("\n[4/5] Running Downstream Footprint Extraction & Trust Correlation...")
    print("=" * 86)
    print("SCIENTIFIC HONESTY NOTICE:")
    print("No independent vector ground truth exists. Metrics are reported strictly as:")
    print("1. Bicubic vs. SR Reconstruction Agreement (IoU & Dice)")
    print("2. Relative Agreement Against Pre-Degradation Reference HR Tile")
    print("=" * 86)
    print(
        f"{'Tile':<6} | {'Bic-SR IoU':<12} | {'Hi-Trust IoU':<13} | {'Lo-Trust IoU':<13} | "
        f"{'SR-Ref IoU':<12} | {'Hi-Trust %':<11}"
    )
    print("-" * 86)

    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))

    for idx, t_idx in enumerate(sample_indices):
        hr_tile = tiles[t_idx]["data"]
        lr_tile = synthesize_pseudo_lr(
            hr_tile,
            downsample_factor=2,
            blur_kernel_size=3,
            noise_std=0.01,
            seed=800 + idx,
        )

        # Baseline: Bicubic upsampling
        bicubic_tile = bicubic_upsample(lr_tile, scale_factor=2)

        # SR ensemble reconstruction & disagreement
        sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)

        # Stability map
        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
            device=device,
        )

        # Spectral check
        spectral_metrics = compute_spectral_consistency(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        delta_ndvi = spectral_metrics["delta_ndvi"]

        # Structural gradient difference
        grad_gt = compute_gradient_magnitude(hr_tile)
        grad_sr = compute_gradient_magnitude(sr_tile)
        structural_diff = np.abs(grad_sr - grad_gt)

        # Phase 8 Evidence Fusion -> Trust Map
        fusion_result = fuse_trust_risk_maps(
            disagreement_map=disagreement_map,
            stability_map=stability_map,
            delta_ndvi_map=delta_ndvi,
            structural_diff_map=structural_diff,
        )
        trust_map = fusion_result["trust_map"]

        # 4. Extract Footprints Identically
        try:
            foot_bic = extract_building_footprints(
                bicubic_tile,
                tophat_kernel_size=kernel_size,
                tophat_threshold=tophat_thresh,
                max_ndvi=max_ndvi,
                min_area=min_area,
                max_area=max_area,
                red_idx=red_idx,
                nir_idx=nir_idx,
            )
            foot_sr = extract_building_footprints(
                sr_tile,
                tophat_kernel_size=kernel_size,
                tophat_threshold=tophat_thresh,
                max_ndvi=max_ndvi,
                min_area=min_area,
                max_area=max_area,
                red_idx=red_idx,
                nir_idx=nir_idx,
            )
            foot_ref = extract_building_footprints(
                hr_tile,
                tophat_kernel_size=kernel_size,
                tophat_threshold=tophat_thresh,
                max_ndvi=max_ndvi,
                min_area=min_area,
                max_area=max_area,
                red_idx=red_idx,
                nir_idx=nir_idx,
            )
        except FootprintExtractionError as e:
            print(f"\n[STOP CONDITION TRIGGERED: Footprint Extraction Error]\n{e}")
            return 2

        # STOP CONDITION CHECK: Check if extraction produces nonsense
        if foot_sr["footprint_pixels"] == 0 and foot_ref["footprint_pixels"] > 100:
            print(f"\n[STOP CONDITION WARNING] Extractor produced 0 footprints on sample #{idx}!")
            return 2

        # Compare footprints
        comp_stats = compare_downstream_footprints(
            mask_bicubic=foot_bic["mask"],
            mask_sr=foot_sr["mask"],
            trust_map=trust_map,
            ref_mask=foot_ref["mask"],
            trust_threshold=trust_thresh,
        )

        overall_iou = comp_stats["overall_bic_sr"]["iou"]
        high_iou = comp_stats["high_trust_bic_sr"]["iou"]
        low_iou = comp_stats["low_trust_bic_sr"]["iou"]
        sr_ref_iou = comp_stats["reference_comparison"]["sr_vs_ref_iou"]
        high_pct = comp_stats["high_trust_area_pct"]

        print(
            f"#{idx:<5} | "
            f"{overall_iou:<12.4f} | "
            f"{high_iou:<13.4f} | "
            f"{low_iou:<13.4f} | "
            f"{sr_ref_iou:<12.4f} | "
            f"{high_pct:<10.1f}%"
        )

        # Render visual overlay
        preview_path = previews_dir / f"downstream_footprint_overlay_sample_{idx}.png"
        render_downstream_overlay(
            gt_tile=hr_tile,
            bicubic_tile=bicubic_tile,
            sr_tile=sr_tile,
            mask_bicubic=foot_bic["mask"],
            mask_sr=foot_sr["mask"],
            trust_map=trust_map,
            comparison_stats=comp_stats,
            sample_index=idx,
            output_path=str(preview_path),
        )

    print("-" * 86)
    print(f"\n[5/5] Diagnostic visualizations saved to: {previews_dir.resolve()}")
    for idx in range(len(sample_indices)):
        print(f"  - downstream_footprint_overlay_sample_{idx}.png")

    print("\n[SUCCESS] Phase 9 Downstream Task Evaluation completed and verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
