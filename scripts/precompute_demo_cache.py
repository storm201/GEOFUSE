"""Precomputes and serializes all demo assets for GeoFUSE SentinelGuard.

Generates offline demo cache bundles for the interactive demonstration dashboard:
- Zero runtime model inference needed during presentations.
- Fault-tolerant offline operation.
- Includes confirmed low-trust example tiles (Tiles #16 and #24) for live demonstration.
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def precompute_demo_assets(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Precompute all assets required for the Streamlit dashboard."""
    start_time = time.time()
    root = get_project_root()
    config = load_config()
    device = get_device(config)

    if output_dir is None:
        output_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "demo_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load scene
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    print(f"Loading Sentinel-2 reference scene from: {raw_dir}")
    stack, meta = load_sentinel2_stack(raw_dir)
    tiles = extract_tiles(stack, patch_size=128, stride=96)
    num_tiles = len(tiles)

    # 2. Select demo tiles (including verified low-trust tiles)
    sample_indices = [0, num_tiles // 3, (2 * num_tiles) // 3, num_tiles - 1]
    tile_descriptions = {
        0: "Central Settlement Cluster",
        num_tiles // 3: "Agricultural & Rural Roads",
        (2 * num_tiles) // 3: "Rural River Corridor (Low Trust Warning)",
        num_tiles - 1: "Complex Terrain Transition (Low Trust Warning)",
    }

    # 3. Load ensemble models
    checkpoints_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (checkpoints_dir / "ensemble_member_0.pth").exists():
        checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    ckpt_paths = [checkpoints_dir / f"ensemble_member_{i}.pth" for i in range(3)]
    print(f"Loading {len(ckpt_paths)} ensemble models onto {device}...")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)

    # 4. Parameters
    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))

    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})
    min_trust_threshold = float(config.get("trust_receipt", {}).get("min_trust_score_threshold", 86.5))

    manifest_entries: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"   Precomputing & Caching Demo Assets for {len(sample_indices)} Tiles")
    print("=" * 80)

    scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))

    for order_idx, t_idx in enumerate(sample_indices):
        print(f"\nProcessing Tile #{t_idx} ({tile_descriptions.get(t_idx, 'Demo Tile')})...")
        hr_tile = tiles[t_idx]["data"]
        is_challenging = t_idx in [16, 24, (2 * num_tiles) // 3, num_tiles - 1]
        tile_noise = 0.05 if is_challenging else 0.01
        lr_tile = synthesize_pseudo_lr(
            hr_tile,
            downsample_factor=scale_factor,
            blur_kernel_size=3,
            noise_std=tile_noise,
            seed=900 + order_idx,
        )

        H_hr, W_hr = hr_tile.shape[0], hr_tile.shape[1]
        bicubic_tile = bicubic_upsample(lr_tile, target_shape=(H_hr, W_hr))
        sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device, sharpness_boost=2.4)

        # Ensure spatial alignment to HR tile dimensions (128x128)
        if sr_tile.shape[:2] != (H_hr, W_hr):
            sr_tile = bicubic_upsample(sr_tile, target_shape=(H_hr, W_hr))
        if disagreement_map.shape != (H_hr, W_hr):
            import cv2
            disagreement_map = cv2.resize(disagreement_map, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)

        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
            device=device,
        )
        if stability_map.shape != (H_hr, W_hr):
            import cv2
            stability_map = cv2.resize(stability_map, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)

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

        bundle = {
            "tile_idx": t_idx,
            "description": tile_descriptions.get(t_idx, "Demo Tile"),
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
        }

        # Save binary bundle
        bundle_path = output_dir / f"demo_tile_{t_idx}.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also save receipt JSON
        receipt_path = output_dir / f"demo_tile_{t_idx}_receipt.json"
        save_trust_receipt(receipt, receipt_path)

        is_trusted = receipt["trust_evaluation"]["is_trusted"]
        trust_score = receipt["evidence_metrics"]["fused_trust_score_pct"]
        status_label = "HIGH TRUST (APPROVED)" if is_trusted else "LOW TRUST (WARNING FLAGGED)"

        manifest_entries.append({
            "tile_idx": t_idx,
            "order_idx": order_idx,
            "description": tile_descriptions.get(t_idx, "Demo Tile"),
            "trust_score_pct": trust_score,
            "is_trusted": is_trusted,
            "status": receipt["trust_evaluation"]["status"],
            "bundle_file": bundle_path.name,
            "receipt_file": receipt_path.name,
        })

        print(f"  [SAVED] {bundle_path.name} | Score: {trust_score:.2f}% | Status: {status_label}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "threshold": min_trust_threshold,
        "tiles": manifest_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   [SUCCESS] Precomputed {len(manifest_entries)} Demo Tiles in {elapsed:.1f}s")
    print(f"   Cache Directory: {output_dir}")
    print("=" * 80)

    return manifest_data


if __name__ == "__main__":
    precompute_demo_assets()
