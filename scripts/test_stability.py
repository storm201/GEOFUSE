"""CLI script to execute input-perturbation stability testing on Sentinel-2 tiles.

Validates the second independent uncertainty signal (input sensitivity)
against the ensemble disagreement map (model variance).
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import evaluate_reconstruction_fidelity, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.stability import (
    compare_disagreement_and_stability,
    compute_stability_map,
)
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def create_rgb_preview(tile_4band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Render normalized 8-bit true-color RGB composite."""
    r = tile_4band[:, :, 2]
    g = tile_4band[:, :, 1]
    b = tile_4band[:, :, 0]
    rgb_channels = []
    for ch in [r, g, b]:
        valid = ch[ch > 0]
        if valid.size > 0:
            vmin = np.percentile(valid, p_low)
            vmax = np.percentile(valid, p_high)
            norm = np.clip((ch - vmin) / max(1e-5, vmax - vmin), 0.0, 1.0)
        else:
            norm = np.zeros_like(ch, dtype=np.float32)
        rgb_channels.append((norm * 255.0).astype(np.uint8))
    return np.stack(rgb_channels, axis=-1)


def render_stability_comparison_figure(
    lr_tile: np.ndarray,
    ensemble_mean: np.ndarray,
    disagreement_map: np.ndarray,
    stability_map: np.ndarray,
    diff_map: np.ndarray,
    corr_info: dict,
    sample_index: int,
    output_path: Path,
) -> None:
    """Save 5-panel figure comparing input stability with ensemble disagreement."""
    lr_rgb_native = create_rgb_preview(lr_tile)
    mean_rgb = create_rgb_preview(ensemble_mean)

    # Nearest display of LR
    lr_disp = cv2.resize(
        lr_rgb_native, (mean_rgb.shape[1], mean_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5), dpi=150)
    fig.patch.set_facecolor("#181818")

    titles = [
        f"Input Pseudo-LR (2x)\n[{lr_tile.shape[0]}x{lr_tile.shape[1]} px, Nearest Display]",
        f"Ensemble Disagreement (Phase 5)\n[Model Variance sigma, Mean: {np.mean(disagreement_map):.4f}]",
        f"Perturbation Stability (Phase 6)\n[Input Variance, Mean: {np.mean(stability_map):.5f}]",
        f"Absolute Difference (|D - S|)\n[Pearson r = {corr_info['pearson_r']:.3f}]",
        f"Ensemble Mean SR (2x)\n[Reconstructed Multi-Spectral]",
    ]

    # 1. LR
    axes[0].imshow(lr_disp)

    # 2. Disagreement Map
    im1 = axes[1].imshow(disagreement_map, cmap="magma")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color="white")

    # 3. Stability Map
    im2 = axes[2].imshow(stability_map, cmap="viridis")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="white")

    # 4. Difference Map
    im3 = axes[3].imshow(diff_map, cmap="coolwarm")
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar3.ax.axes, "yticklabels"), color="white")

    # 5. Ensemble Mean SR
    axes[4].imshow(mean_rgb)

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard — Dual Uncertainty Signals: Disagreement vs. Stability (Sample #{sample_index})",
        color="white",
        fontsize=13,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("=" * 76)
    print("   GeoFUSE SentinelGuard — Phase 6: Input-Perturbation Stability Testing")
    print("=" * 76)

    root = get_project_root()
    config = load_config()
    device = get_device(config)

    # Load perturbation configuration
    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))
    num_trials = int(pert_cfg.get("num_trials", 3))

    print(f"\nPerturbation Parameters from config.yaml:")
    print(f"  Noise Levels (sigma_noise) : {noise_levels}")
    print(f"  Brightness Jitter (sigma_j) : {jitter_std}")
    print(f"  Trials per Level      : {num_trials}")

    # Checkpoint paths
    checkpoints_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (checkpoints_dir / "ensemble_member_0.pth").exists():
        checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    previews_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "previews"
    ckpt_paths = [
        checkpoints_dir / "ensemble_member_0.pth",
        checkpoints_dir / "ensemble_member_1.pth",
        checkpoints_dir / "ensemble_member_2.pth",
    ]

    print(f"\n[1/3] Loading trained ensemble members...")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    print(f"      Loaded {len(models)} models onto {device}.")

    # Load data
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, _ = load_sentinel2_stack(raw_dir)
    tiles = extract_tiles(stack, patch_size=128, stride=96)
    sample_indices = [0, len(tiles) // 3, (2 * len(tiles)) // 3, len(tiles) - 1]

    print(f"\n[2/3] Running stability testing across {len(sample_indices)} test tiles...")
    print("-" * 76)
    print(f"{'Sample':<8} | {'Disag Mean':<12} | {'Stab Mean':<12} | {'Pearson r':<11} | {'Status':<14}")
    print("-" * 76)

    for idx, t_idx in enumerate(sample_indices):
        hr_tile = tiles[t_idx]["data"]
        lr_tile = synthesize_pseudo_lr(hr_tile, downsample_factor=2, blur_kernel_size=3, noise_std=0.01, seed=888 + idx)

        # 1. Compute Phase 5 Disagreement Map
        mean_recon, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)

        # 2. Compute Phase 6 Stability Map
        stability_map, _, _ = compute_stability_map(
            models=models,
            lr_tile=lr_tile,
            noise_levels=noise_levels,
            brightness_jitter_std=jitter_std,
            num_trials=num_trials,
            device=device,
            base_seed=1000 + idx * 50,
        )

        # 3. Compare Signals
        cmp_result = compare_disagreement_and_stability(disagreement_map, stability_map)

        # Check Stop Condition: Degenerate or identical
        if cmp_result["is_degenerate"]:
            print(f"\n[STOP CONDITION TRIGGERED] Stability map is degenerate (all zero/NaN) in Sample #{idx}!")
            return 2
        if cmp_result["is_identical"]:
            print(f"\n[STOP CONDITION TRIGGERED] Stability map is identical to disagreement map in Sample #{idx}!")
            return 2

        status_str = "VALID [OK]" if cmp_result["checklist_passed"] else "BORDERLINE"

        print(
            f"#{idx:<7} | "
            f"{np.mean(disagreement_map):<12.5f} | "
            f"{np.mean(stability_map):<12.5f} | "
            f"{cmp_result['pearson_r']:<11.4f} | "
            f"{status_str}"
        )

        # 4. Render and Save Comparison Preview Figure
        out_fig_path = previews_dir / f"stability_preview_sample_{idx}.png"
        render_stability_comparison_figure(
            lr_tile=lr_tile,
            ensemble_mean=mean_recon,
            disagreement_map=disagreement_map,
            stability_map=stability_map,
            diff_map=cmp_result["diff_map"],
            corr_info=cmp_result,
            sample_index=idx,
            output_path=out_fig_path,
        )

    print("-" * 76)
    print(f"\n[3/3] Stability comparison figures saved to: {previews_dir.resolve()}")
    for idx in range(len(sample_indices)):
        print(f"  - stability_preview_sample_{idx}.png")

    print("\n[SUCCESS] Phase 6 Input-Perturbation Stability Testing completed and verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
