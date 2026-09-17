"""CLI script to generate multi-evidence Trust/Risk maps and overlays.

Executes the complete evidence-fusion pipeline on held-out Sentinel-2 tiles:
- Ensemble Disagreement (Phase 5)
- Perturbation Stability (Phase 6)
- Spectral NDVI Fidelity (Phase 7)
- Structural Edge Fidelity (Phase 7)
- Composite Trust & Risk Map Fusion (Phase 8)
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
from src.evaluation.edge_check import compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps, render_trust_risk_overlay
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def main() -> int:
    print("=" * 84)
    print("      GeoFUSE SentinelGuard -- Phase 8: Multi-Evidence Trust/Risk Map Fusion")
    print("=" * 84)

    root = get_project_root()
    config = load_config()
    device = get_device(config)

    fusion_cfg = config.get("verification", {}).get("evidence_fusion", {})
    weights = fusion_cfg.get("weights", {
        "disagreement": 0.25,
        "stability": 0.25,
        "spectral": 0.25,
        "structural": 0.25,
    })
    high_risk_thresh = float(fusion_cfg.get("high_risk_threshold", 0.65))

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

    print(f"\n[2/5] Ingesting Sentinel-2 reference scene...")
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, meta = load_sentinel2_stack(raw_dir)
    print(f"      Scene shape: {stack.shape}, CRS: {meta.get('crs')}")

    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    # Extract held-out test tiles
    patch_size = 128
    tiles = extract_tiles(stack, patch_size=patch_size, stride=96)
    sample_indices = [0, len(tiles) // 3, (2 * len(tiles)) // 3, len(tiles) - 1]
    print(f"      Selected {len(sample_indices)} geographically distinct test tiles for fusion.")

    # Perturbation parameters from config.yaml
    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))

    print(f"\n[3/5] Configured Fusion Weights from config.yaml:")
    for k, v in weights.items():
        print(f"      - {k:<14}: {v}")

    print("\n[4/5] Executing Evidence Fusion Pipeline...")
    print("-" * 88)
    print(
        f"{'Tile':<6} | {'Trust Score':<12} | {'Risk Score':<11} | {'High-Risk %':<12} | "
        f"{'Disag (std)':<12} | {'Spatial Var?':<12}"
    )
    print("-" * 88)

    for idx, t_idx in enumerate(sample_indices):
        hr_tile = tiles[t_idx]["data"]
        lr_tile = synthesize_pseudo_lr(
            hr_tile,
            downsample_factor=2,
            blur_kernel_size=3,
            noise_std=0.01,
            seed=700 + idx,
        )

        # Signal 1: Ensemble SR & Disagreement Map
        sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)

        # Signal 2: Perturbation Stability Map
        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
            device=device,
        )

        # Signal 3: Spectral Inconsistency (Delta-NDVI Map)
        spectral_metrics = compute_spectral_consistency(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            red_idx=red_idx,
            nir_idx=nir_idx,
        )
        delta_ndvi = spectral_metrics["delta_ndvi"]

        # Signal 4: Structural Edge Error (Gradient Magnitude Difference)
        grad_gt = compute_gradient_magnitude(hr_tile)
        grad_sr = compute_gradient_magnitude(sr_tile)
        structural_diff = np.abs(grad_sr - grad_gt)

        # STOP CONDITION CHECK:
        # Scale mismatch check before normalization:
        max_disag = float(np.max(disagreement_map))
        max_stab = float(np.max(stability_map))
        max_ndvi = float(np.max(delta_ndvi))
        max_struct = float(np.max(structural_diff))

        # Check raw scale ranges
        if max_struct > 0 and (max_disag == 0 or max_stab == 0):
            print("\n[STOP CONDITION WARNING] A component has degenerate scale!")
            return 2

        # Fuse evidence with per-metric [0, 1] normalization
        fusion_result = fuse_trust_risk_maps(
            disagreement_map=disagreement_map,
            stability_map=stability_map,
            delta_ndvi_map=delta_ndvi,
            structural_diff_map=structural_diff,
            weights=weights,
            high_risk_threshold=high_risk_thresh,
        )

        # CHECKLIST ITEM: Risk map shows spatial variation, not a flat color
        if not fusion_result["has_spatial_variation"]:
            print(f"\n[CHECKLIST FAILURE] Risk map for sample #{idx} is degenerate (flat uniform color)!")
            return 3

        print(
            f"#{idx:<5} | "
            f"{fusion_result['trust_score_pct']:<10.2f}% | "
            f"{fusion_result['mean_risk_score']:<11.4f} | "
            f"{fusion_result['pct_high_risk_pixels']:<11.2f}% | "
            f"{fusion_result['std_risk']:<12.5f} | "
            f"{'YES' if fusion_result['has_spatial_variation'] else 'NO':<12}"
        )

        # Render diagnostic overlay
        preview_path = previews_dir / f"trust_risk_overlay_sample_{idx}.png"
        render_trust_risk_overlay(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            fusion_result=fusion_result,
            sample_index=idx,
            output_path=str(preview_path),
        )

    print("-" * 88)
    print(f"\n[5/5] Diagnostic visualizations saved to: {previews_dir.resolve()}")
    for idx in range(len(sample_indices)):
        print(f"  - trust_risk_overlay_sample_{idx}.png")

    print("\n[SUCCESS] Phase 8 Multi-Evidence Trust/Risk Map Fusion successfully verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
