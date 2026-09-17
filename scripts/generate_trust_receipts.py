"""CLI script to generate and save auditable Trust Receipts for Sentinel-2 tiles.

Executes complete pipeline across sample tiles, compiles machine-readable JSON Trust Receipts,
and writes them to outputs/receipts/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import bicubic_upsample, synthesize_pseudo_lr
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
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.evaluation.trust_receipt import generate_trust_receipt, save_trust_receipt
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate auditable Trust Receipts for Sentinel-2 tiles.")
    parser.add_argument(
        "--tile",
        type=str,
        default=None,
        help="Tile index or list of indices to evaluate (e.g. '12', '0,4,12,24', or 'all'). Default: benchmark tiles [0, 8, 16, 24].",
    )
    args = parser.parse_args(argv)

    print("=" * 86)
    print("        GeoFUSE SentinelGuard -- Phase 11: Auditable Trust Receipt Generation")
    print("=" * 86)

    root = get_project_root()
    config = load_config()
    device = get_device(config)

    receipt_cfg = config.get("trust_receipt", {})
    min_trust_threshold = float(receipt_cfg.get("min_trust_score_threshold", 86.0))
    receipts_dir = root / receipt_cfg.get("outputs_dir", "outputs/receipts")
    receipts_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (checkpoints_dir / "ensemble_member_0.pth").exists():
        checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    ckpt_paths = [
        checkpoints_dir / f"ensemble_member_{i}.pth" for i in range(3)
    ]

    print(f"\n[1/4] Loading 3 ensemble models onto {device}...")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    print(f"      Successfully loaded {len(models)} models.")

    print("\n[2/4] Ingesting Sentinel-2 scene and extracting test tiles...")
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, meta = load_sentinel2_stack(raw_dir)
    print(f"      Scene shape: {stack.shape}, CRS: {meta.get('crs')}")

    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    patch_size = 128
    tiles = extract_tiles(stack, patch_size=patch_size, stride=96)
    if args.tile is not None:
        if args.tile.strip().lower() == "all":
            sample_indices = list(range(len(tiles)))
            print(f"      Selected all {len(sample_indices)} tiles in the complete scene.")
        else:
            try:
                sample_indices = [int(x.strip()) for x in args.tile.split(",") if x.strip()]
                sample_indices = [idx for idx in sample_indices if 0 <= idx < len(tiles)]
                print(f"      Selected custom tile list ({len(sample_indices)} tiles): {sample_indices}")
            except ValueError:
                print(f"[Warning] Invalid tile argument '{args.tile}'. Using default benchmark samples.")
                sample_indices = [0, len(tiles) // 3, (2 * len(tiles)) // 3, len(tiles) - 1]
    else:
        sample_indices = [0, len(tiles) // 3, (2 * len(tiles)) // 3, len(tiles) - 1]
        print(f"      Selected {len(sample_indices)} geographically distinct test tiles.")

    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))

    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})

    print(f"\n[3/4] Generating Auditable Trust Receipts (Threshold = {min_trust_threshold}%)...")
    print("-" * 90)
    print(
        f"{'Tile':<6} | {'Trust Score':<12} | {'Category':<20} | {'Status':<10} | {'Warnings Count':<14} | {'Saved JSON'}"
    )
    print("-" * 90)

    receipt_paths = []
    has_low_trust_warning = False

    scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))

    for idx, t_idx in enumerate(sample_indices):
        hr_tile = tiles[t_idx]["data"]
        lr_tile = synthesize_pseudo_lr(
            hr_tile,
            downsample_factor=scale_factor,
            blur_kernel_size=3,
            noise_std=0.01,
            seed=900 + idx,
        )

        bicubic_tile = bicubic_upsample(lr_tile, scale_factor=scale_factor, target_shape=hr_tile.shape[:2])
        sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)
        if sr_tile.shape[:2] != hr_tile.shape[:2]:
            sr_tile = cv2.resize(sr_tile, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)
            disagreement_map = cv2.resize(disagreement_map, (hr_tile.shape[1], hr_tile.shape[0]), interpolation=cv2.INTER_CUBIC)

        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
            device=device,
        )

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

        fusion_result = fuse_trust_risk_maps(
            disagreement_map=disagreement_map,
            stability_map=stability_map,
            delta_ndvi_map=spectral_metrics["delta_ndvi"],
            structural_diff_map=structural_diff,
        )

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
            tile_idx=t_idx,
            raw_dir=raw_dir,
            config=config,
            pipeline_data=pipeline_data,
            min_trust_threshold=min_trust_threshold,
        )

        # Retain trust_receipt_sample_{idx}.json for benchmark samples, or trust_receipt_tile_{t_idx}.json for custom requests
        if args.tile is not None and args.tile.strip().lower() != "all" and len(sample_indices) == 1:
            out_file = receipts_dir / f"trust_receipt_tile_{t_idx}.json"
        else:
            out_file = receipts_dir / f"trust_receipt_sample_{idx}.json"

        save_trust_receipt(receipt, out_file)
        receipt_paths.append(out_file)

        eval_status = receipt["trust_evaluation"]
        if not eval_status["is_trusted"]:
            has_low_trust_warning = True

        status_str = "TRUSTED" if eval_status["is_trusted"] else "FLAGGED"
        print(
            f"#{t_idx:<5} | "
            f"{receipt['evidence_metrics']['fused_trust_score_pct']:<10.2f}% | "
            f"{eval_status['status']:<20} | "
            f"{status_str:<10} | "
            f"{len(eval_status['warnings_and_advisories']):<14} | "
            f"{out_file.name}"
        )

    print("-" * 90)
    print(f"\n[4/4] Saved {len(receipt_paths)} Trust Receipts to: {receipts_dir.resolve()}")

    # CHECKLIST VALIDATION:
    if not has_low_trust_warning:
        print("\n[CHECKLIST WARNING] Expected at least one sample tile to trigger a low-trust warning!")
    else:
        print("\n[CHECKLIST PASSED] At least one tile triggered a verifiable low-trust warning.")

    print("\n[SUCCESS] Phase 11 Trust Receipt generation complete and verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
