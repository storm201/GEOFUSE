"""End-to-End Reproducibility Script for GeoFUSE SentinelGuard.

Executes the entire research and evaluation pipeline from scratch or using existing checkpoints:
1. Environment and configuration verification
2. Data preparation and baseline degrade-and-recover verification (Phase 2)
3. Sequential training of all 3 ensemble members with geographic hold-out (Phases 4 & 5)
4. Ensemble inference and disagreement uncertainty mapping (Phase 5)
5. Controlled input-perturbation stability testing (Phase 6)
6. Spectral (delta-NDVI) and structural (Sobel gradient / Canny edge) consistency checks (Phase 7)
7. Multi-evidence trust/risk map fusion (Phase 8)
8. Downstream building footprint extraction and trust correlation (Phase 9)
9. Auditable trust receipt generation (Phase 11)
10. Test suite verification (PyTest)
11. Optional Streamlit dashboard launch (Phase 10)

All parameters, hyperparameter values, and file paths are loaded strictly
from config.yaml (no hardcoded paths or credentials).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import bicubic_upsample, evaluate_reconstruction_fidelity, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.downstream_eval import compare_downstream_footprints, extract_building_footprints
from src.evaluation.edge_check import compute_edge_consistency, compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.evaluation.trust_receipt import generate_trust_receipt, save_trust_receipt
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.models.train import train_ensemble
from src.utils.config import get_device, get_project_root, load_config


def step_banner(step_num: int, total_steps: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"[{step_num}/{total_steps}] {title}")
    print("=" * 80)


def run_pipeline(
    skip_training: bool = False,
    force_retrain: bool = False,
    skip_tests: bool = False,
    launch_dashboard: bool = False,
    epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the complete end-to-end GeoFUSE pipeline reproducibly."""
    total_steps = 11 if not launch_dashboard else 12
    start_time = time.time()
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Step 1: Configuration & Environment Verification
    # -------------------------------------------------------------------------
    step_banner(1, total_steps, "Configuration & Compute Device Resolution")
    root = get_project_root()
    config = load_config()
    device = get_device(config)

    print(f"Project Root     : {root}")
    print(f"Project Name     : {config.get('project', {}).get('name')}")
    print(f"Version          : {config.get('project', {}).get('version')}")
    print(f"Compute Device   : {device}")
    results["device"] = str(device)

    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    outputs_dir = root / config.get("paths", {}).get("outputs_dir", "outputs")
    checkpoints_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (checkpoints_dir / "ensemble_member_0.pth").exists() and (outputs_dir / "checkpoints" / "ensemble_member_0.pth").exists():
        checkpoints_dir = outputs_dir / "checkpoints"
    previews_dir = outputs_dir / "previews"
    receipts_dir = outputs_dir / "receipts"

    for d in [checkpoints_dir, previews_dir, receipts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 2: Data Ingestion & Integrity Check
    # -------------------------------------------------------------------------
    step_banner(2, total_steps, "Sentinel-2 Data Ingestion & Preprocessing")
    print(f"Ingesting raw multi-band Sentinel-2 stack from: {raw_dir}")
    stack, profile = load_sentinel2_stack(raw_dir)
    print(f"  Shape      : {stack.shape} (H, W, Bands)")
    print(f"  CRS        : {profile.get('crs')}")
    print(f"  Resolution : {profile.get('transform')[0]}m GSD")
    print(f"  Data Range : [{stack.min():.4f}, {stack.max():.4f}]")
    results["stack_shape"] = stack.shape

    # -------------------------------------------------------------------------
    # Step 3: Degrade-and-Recover Baseline Generation (Phase 2)
    # -------------------------------------------------------------------------
    step_banner(3, total_steps, "Baseline Degrade-and-Recover Verification (Phase 2)")
    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    raw_tiles = extract_tiles(stack, patch_size=128, stride=96)
    sample_indices = [0, len(raw_tiles) // 3, (2 * len(raw_tiles)) // 3, len(raw_tiles) - 1]
    print(f"Extracted {len(raw_tiles)} candidate tiles; selected {len(sample_indices)} geographically distinct test tiles.")
    
    scale_factor = float(config.get("model", {}).get("scale_factor", 2.5))
    eval_tiles = [raw_tiles[idx]["data"] for idx in sample_indices]
    baseline_psnrs = []
    baseline_ssims = []
    for i, hr_tile in enumerate(eval_tiles):
        H_hr, W_hr = hr_tile.shape[0], hr_tile.shape[1]
        pseudo_lr = synthesize_pseudo_lr(hr_tile, downsample_factor=scale_factor, blur_kernel_size=3, noise_std=0.01, seed=900 + i)
        bicubic_sr = bicubic_upsample(pseudo_lr, target_shape=(H_hr, W_hr))
        fidelity = evaluate_reconstruction_fidelity(hr_tile, bicubic_sr)
        baseline_psnrs.append(fidelity["psnr_db"])
        baseline_ssims.append(fidelity["ssim"])
        print(f"  Tile #{i}: Bicubic {scale_factor}x PSNR = {fidelity['psnr_db']:.2f} dB, SSIM = {fidelity['ssim']:.4f}")

    avg_bic_psnr = float(sum(baseline_psnrs) / len(baseline_psnrs))
    avg_bic_ssim = float(sum(baseline_ssims) / len(baseline_ssims))
    print(f"  Average Bicubic Baseline (Held-Out Preview): PSNR = {avg_bic_psnr:.2f} dB, SSIM = {avg_bic_ssim:.4f}")
    results["baseline_psnr"] = avg_bic_psnr
    results["baseline_ssim"] = avg_bic_ssim

    # -------------------------------------------------------------------------
    # Step 4: Ensemble Model Training (Phases 4 & 5)
    # -------------------------------------------------------------------------
    step_banner(4, total_steps, "Ensemble Training on Geographic Hold-Out (Phases 4 & 5)")
    member_seeds = config.get("ensemble", {}).get("member_seeds", [42, 101, 2024])
    ckpt_paths = [checkpoints_dir / f"ensemble_member_{idx}.pth" for idx in range(len(member_seeds))]
    all_ckpts_exist = all(p.exists() for p in ckpt_paths)

    if all_ckpts_exist and skip_training:
        print("Existing ensemble checkpoints found and --skip-training requested:")
        for p in ckpt_paths:
            print(f"  - Reusing checkpoint: {p.name}")
    else:
        print(f"Training {len(member_seeds)} ensemble members sequentially with seeds: {member_seeds}...")
        train_ensemble(config=config, member_seeds=member_seeds, epochs=epochs_override)

    # -------------------------------------------------------------------------
    # Step 5: Ensemble Inference & Disagreement Map (Phase 5)
    # -------------------------------------------------------------------------
    step_banner(5, total_steps, "Ensemble Inference & Disagreement Uncertainty (Phase 5)")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    print(f"Loaded {len(models)} ensemble models onto {device}.")

    disagreement_maps = []
    sr_outputs = []
    pseudo_lrs = []

    for i, hr_tile in enumerate(eval_tiles):
        H_hr, W_hr = hr_tile.shape[0], hr_tile.shape[1]
        lr_tile = synthesize_pseudo_lr(hr_tile, downsample_factor=scale_factor, blur_kernel_size=3, noise_std=0.01, seed=900 + i)
        pseudo_lrs.append(lr_tile)
        mean_sr, disag_map, _ = predict_ensemble(models, lr_tile, device=device)
        if mean_sr.shape[:2] != (H_hr, W_hr):
            mean_sr = bicubic_upsample(mean_sr, target_shape=(H_hr, W_hr))
        if disag_map.shape != (H_hr, W_hr):
            import cv2
            disag_map = cv2.resize(disag_map, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)
        sr_outputs.append(mean_sr)
        disagreement_maps.append(disag_map)
        fid = evaluate_reconstruction_fidelity(hr_tile, mean_sr)
        print(f"  Tile #{i}: Ensemble SR PSNR = {fid['psnr_db']:.2f} dB, SSIM = {fid['ssim']:.4f} | Disag Std = {disag_map.mean():.5f}")

    # -------------------------------------------------------------------------
    # Step 6: Input-Perturbation Stability Testing (Phase 6)
    # -------------------------------------------------------------------------
    step_banner(6, total_steps, "Perturbation Stability Analysis (Phase 6)")
    stability_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = stability_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(stability_cfg.get("brightness_jitter_std", 0.02))

    stability_maps = []
    for i, (hr_tile, lr_tile) in enumerate(zip(eval_tiles, pseudo_lrs)):
        H_hr, W_hr = hr_tile.shape[0], hr_tile.shape[1]
        stab_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            device=device,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=2,
        )
        if stab_map.shape != (H_hr, W_hr):
            import cv2
            stab_map = cv2.resize(stab_map, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR)
        stability_maps.append(stab_map)
        print(f"  Tile #{i}: Mean Stability Variance = {stab_map.mean():.6f}")

    # -------------------------------------------------------------------------
    # Step 7: Radiometric Spectral & Structural Consistency (Phase 7)
    # -------------------------------------------------------------------------
    step_banner(7, total_steps, "Radiometric & Structural Consistency Checks (Phase 7)")
    spectral_metrics_list = []
    edge_metrics_list = []
    structural_diff_list = []

    spectral_cfg = config.get("verification", {}).get("spectral_consistency", {})
    sensor_noise_floor = float(spectral_cfg.get("sensor_noise_floor", 0.03))
    ndvi_thresh = float(spectral_cfg.get("ndvi_threshold", 0.05))

    for i, (hr_tile, sr_tile) in enumerate(zip(eval_tiles, sr_outputs)):
        spec_res = compute_spectral_consistency(
            gt_tile=hr_tile,
            sr_tile=sr_tile,
            ndvi_threshold=ndvi_thresh,
            red_idx=2,
            nir_idx=3,
            sensor_noise_floor=sensor_noise_floor,
        )
        edge_res = compute_edge_consistency(hr_tile, sr_tile)

        grad_hr = compute_gradient_magnitude(hr_tile)
        grad_sr = compute_gradient_magnitude(sr_tile)
        structural_diff = np.abs(grad_sr - grad_hr)

        spectral_metrics_list.append(spec_res)
        edge_metrics_list.append(edge_res)
        structural_diff_list.append(structural_diff)

        print(f"  Tile #{i}: Mean Delta-NDVI = {spec_res['mean_delta_ndvi']:.4f} | Edge Grad Corr r = {edge_res['gradient_correlation']:.4f}")

    # -------------------------------------------------------------------------
    # Step 8: Multi-Evidence Trust/Risk Map Fusion (Phase 8)
    # -------------------------------------------------------------------------
    step_banner(8, total_steps, "Multi-Evidence Trust/Risk Map Fusion (Phase 8)")
    fusion_results = []
    trust_scores = []

    for i in range(len(eval_tiles)):
        fusion_res = fuse_trust_risk_maps(
            disagreement_map=disagreement_maps[i],
            stability_map=stability_maps[i],
            delta_ndvi_map=spectral_metrics_list[i]["delta_ndvi"],
            structural_diff_map=structural_diff_list[i],
        )
        fusion_results.append(fusion_res)
        trust_scores.append(fusion_res["trust_score_pct"])
        print(f"  Tile #{i}: Composite Trust Score = {fusion_res['trust_score_pct']:.2f}% | Mean Risk = {fusion_res['mean_risk_score']:.4f}")

    results["trust_scores"] = trust_scores

    # -------------------------------------------------------------------------
    # Step 9: Downstream Building Footprint Evaluation (Phase 9)
    # -------------------------------------------------------------------------
    step_banner(9, total_steps, "Downstream Task Evaluation (Phase 9)")
    downstream_results = []
    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})

    for i in range(len(eval_tiles)):
        hr_tile = eval_tiles[i]
        lr_tile = pseudo_lrs[i]
        sr_tile = sr_outputs[i]
        t_map = fusion_results[i]["trust_map"]

        bic_sr = bicubic_upsample(lr_tile, target_shape=(hr_tile.shape[0], hr_tile.shape[1]))
        foot_bic = extract_building_footprints(
            bic_sr,
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

        down_cmp = compare_downstream_footprints(
            mask_bicubic=foot_bic["mask"],
            mask_sr=foot_sr["mask"],
            trust_map=t_map,
            ref_mask=foot_ref["mask"],
            trust_threshold=float(down_cfg.get("trust_partition_threshold", 0.85)),
        )
        downstream_results.append(down_cmp)
        print(f"  Tile #{i}: Bicubic vs. SR IoU = {down_cmp['overall_bic_sr']['iou']:.4f} (High-Trust: {down_cmp['high_trust_bic_sr']['iou']:.4f}, Low-Trust: {down_cmp['low_trust_bic_sr']['iou']:.4f})")

    # -------------------------------------------------------------------------
    # Step 10: Auditable Trust Receipt Generation (Phase 11)
    # -------------------------------------------------------------------------
    step_banner(10, total_steps, "Auditable Trust Receipt Generation (Phase 11)")
    min_thresh = float(config.get("trust_receipt", {}).get("min_trust_score_threshold", 86.5))
    saved_receipt_paths = []

    for i in range(len(eval_tiles)):
        pipeline_data = {
            "fusion_result": fusion_results[i],
            "spectral_metrics": spectral_metrics_list[i],
            "edge_metrics": edge_metrics_list[i],
            "downstream_comp": downstream_results[i],
        }
        receipt = generate_trust_receipt(
            tile_idx=i,
            raw_dir=raw_dir,
            config=config,
            pipeline_data=pipeline_data,
            min_trust_threshold=min_thresh,
        )
        out_path = receipts_dir / f"trust_receipt_sample_{i}.json"
        save_trust_receipt(receipt, out_path)
        saved_receipt_paths.append(out_path)
        status_flag = "FLAGGED [LOW TRUST]" if not receipt["trust_evaluation"]["is_trusted"] else "APPROVED [HIGH TRUST]"
        print(f"  Tile #{i}: Trust Score {trust_scores[i]:.2f}% -> {status_flag} -> {out_path.name}")

    results["receipt_paths"] = [str(p) for p in saved_receipt_paths]

    # -------------------------------------------------------------------------
    # Step 11: Automated Test Suite Execution
    # -------------------------------------------------------------------------
    if not skip_tests:
        step_banner(11, total_steps, "PyTest Test Suite Execution")
        test_cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        print(f"Executing: {' '.join(test_cmd)}")
        ret = subprocess.run(test_cmd, cwd=str(root))
        if ret.returncode != 0:
            raise RuntimeError(f"PyTest regression suite failed with return code {ret.returncode}")
        print("  [SUCCESS] All unit and integration tests passed!")
    else:
        print("\n[INFO] Skipping PyTest suite (--skip-tests specified).")

    # -------------------------------------------------------------------------
    # Step 12: Optional Dashboard Launch
    # -------------------------------------------------------------------------
    if launch_dashboard:
        step_banner(12, total_steps, "Launching Streamlit Dashboard (Phase 10)")
        dash_cmd = ["streamlit", "run", "src/dashboard/app.py"]
        print(f"Launching: {' '.join(dash_cmd)}")
        subprocess.run(dash_cmd, cwd=str(root))

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   [SUCCESS] GeoFUSE SentinelGuard End-to-End Pipeline Completed in {elapsed:.1f}s")
    print("=" * 80)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-End Reproducibility Runner for GeoFUSE SentinelGuard")
    parser.add_argument("--skip-training", action="store_true", default=True, help="Reuse existing checkpoints in outputs/checkpoints/")
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retraining of ensemble members from scratch")
    parser.add_argument("--skip-tests", action="store_true", default=False, help="Skip automated pytest execution")
    parser.add_argument("--launch-dashboard", action="store_true", default=False, help="Launch Streamlit dashboard after pipeline execution")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs from config.yaml")

    args = parser.parse_args()

    try:
        run_pipeline(
            skip_training=args.skip_training and not args.force_retrain,
            force_retrain=args.force_retrain,
            skip_tests=args.skip_tests,
            launch_dashboard=args.launch_dashboard,
            epochs_override=args.epochs,
        )
        return 0
    except Exception as e:
        print(f"\n[FATAL ERROR] Reproducibility execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
