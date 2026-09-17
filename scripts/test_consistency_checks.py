"""CLI script to run Spectral Consistency (NDVI) and Structural Edge Checks.

Compares 2x Super-Resolution outputs against reference 10m Sentinel-2 tiles
and saves multi-panel overlay visualizations.
"""

import sys
from pathlib import Path

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.data.degrade import synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.edge_check import (
    compute_edge_consistency,
    render_edge_consistency_overlay,
)
from src.evaluation.spectral_check import (
    SpectralBandError,
    compute_spectral_consistency,
    render_spectral_ndvi_overlay,
)
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def main() -> int:
    print("=" * 76)
    print("   GeoFUSE SentinelGuard -- Phase 7: Spectral & Structural Consistency")
    print("=" * 76)

    root = get_project_root()
    config = load_config()
    device = get_device(config)

    # Thresholds from config.yaml
    spec_cfg = config.get("verification", {}).get("spectral_consistency", {})
    ndvi_thresh = float(spec_cfg.get("ndvi_threshold", 0.05))

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

    print(f"\n[1/4] Loading ensemble models onto {device}...")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    print(f"      Loaded {len(models)} models.")

    # Load Sentinel-2 scene
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, meta = load_sentinel2_stack(raw_dir)
    print(f"      Loaded scene: shape={stack.shape}, bands={meta.get('bands')}")

    # Check for NIR band (Stop condition check)
    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    if "B08" not in bands or "B04" not in bands:
        print("\n[STOP CONDITION TRIGGERED] Missing NIR (B08) or Red (B04) band in configuration!")
        print(f"Configured bands: {bands}. NDVI requires NIR and Red bands.")
        return 2

    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    # Extract tiles (at least 2 different tiles required by checklist)
    patch_size = 128
    tiles = extract_tiles(stack, patch_size=patch_size, stride=96)
    sample_indices = [0, len(tiles) // 3, (2 * len(tiles)) // 3, len(tiles) - 1]
    print(f"\n[2/4] Selected {len(sample_indices)} diverse test tiles for evaluation.")

    print("\n[3/4] Running Spectral (NDVI) & Edge Consistency Checks...")
    print("-" * 80)
    print(f"{'Sample':<7} | {'Mean Delta-NDVI':<15} | {'NDVI Flag %':<12} | {'Edge IoU':<10} | {'Edge F1':<9} | {'Grad Corr':<10}")
    print("-" * 80)

    for idx, t_idx in enumerate(sample_indices):
        hr_tile = tiles[t_idx]["data"]  # Reference 10m tile (pre-degradation)
        lr_tile = synthesize_pseudo_lr(hr_tile, downsample_factor=2, blur_kernel_size=3, noise_std=0.01, seed=999 + idx)

        # 1. Ensemble reconstruction
        sr_tile, _, _ = predict_ensemble(models, lr_tile, device=device)

        # 2. Spectral Consistency Check
        try:
            spectral_metrics = compute_spectral_consistency(
                gt_tile=hr_tile,
                sr_tile=sr_tile,
                ndvi_threshold=ndvi_thresh,
                red_idx=red_idx,
                nir_idx=nir_idx,
            )
        except SpectralBandError as e:
            print(f"\n[STOP CONDITION TRIGGERED: Spectral Band Error]\n{e}")
            return 2

        # 3. Edge / Structural Consistency Check
        edge_metrics = compute_edge_consistency(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            canny_low=40,
            canny_high=120,
        )

        print(
            f"#{idx:<6} | "
            f"{spectral_metrics['mean_delta_ndvi']:<15.5f} | "
            f"{spectral_metrics['pct_inconsistent_pixels']:<11.1f}% | "
            f"{edge_metrics['edge_iou']:<10.4f} | "
            f"{edge_metrics['edge_f1']:<9.4f} | "
            f"{edge_metrics['gradient_correlation']:<10.4f}"
        )

        # 4. Save Overlay Visualizations
        spectral_out = previews_dir / f"spectral_ndvi_overlay_sample_{idx}.png"
        render_spectral_ndvi_overlay(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            spectral_metrics=spectral_metrics,
            sample_index=idx,
            output_path=str(spectral_out),
        )

        edge_out = previews_dir / f"edge_structural_overlay_sample_{idx}.png"
        render_edge_consistency_overlay(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            edge_metrics=edge_metrics,
            sample_index=idx,
            output_path=str(edge_out),
        )

    print("-" * 80)
    print(f"\n[4/4] Saved overlay figures to: {previews_dir.resolve()}")
    for idx in range(len(sample_indices)):
        print(f"  - spectral_ndvi_overlay_sample_{idx}.png")
        print(f"  - edge_structural_overlay_sample_{idx}.png")

    print("\n[SUCCESS] Phase 7 Spectral & Structural Consistency checks completed and verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
