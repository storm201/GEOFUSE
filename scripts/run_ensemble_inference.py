"""Ensemble Inference and Disagreement Map Generation Script.

Executes sequential multi-model ensemble inference on Sentinel-2 tiles to produce:
1. Ensemble Mean Reconstruction: Finer-resolution reconstructed multi-spectral tile.
2. Ensemble Disagreement Map: Per-pixel standard deviation across ensemble members,
   acting as an empirical uncertainty proxy indicating reconstruction instability.
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import evaluate_reconstruction_fidelity, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def create_rgb_preview(tile_4band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Create normalized 8-bit true-color RGB from (H, W, 4) multi-spectral tile."""
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


def render_ensemble_figure(
    lr_tile: np.ndarray,
    ensemble_mean: np.ndarray,
    hr_tile: np.ndarray,
    disagreement_map: np.ndarray,
    metrics: dict,
    sample_index: int,
    output_path: Path,
) -> None:
    """Save multi-panel visualization showing LR, Ensemble Mean, Ground Truth, and Disagreement."""
    hr_rgb = create_rgb_preview(hr_tile)
    mean_rgb = create_rgb_preview(ensemble_mean)

    # Nearest-neighbor upscale of LR for visual alignment
    lr_rgb_native = create_rgb_preview(lr_tile)
    lr_rgb_disp = cv2.resize(
        lr_rgb_native, (hr_rgb.shape[1], hr_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # Residual error map
    residual_error = np.abs(hr_tile.astype(float) - ensemble_mean.astype(float)).mean(axis=-1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=150)
    fig.patch.set_facecolor("#181818")

    titles = [
        f"Input Pseudo-LR (2x Downscaled)\n[{lr_tile.shape[0]}x{lr_tile.shape[1]} px, Nearest]",
        f"Ensemble Mean (2x SR)\n[PSNR: {metrics['psnr_db']:.2f} dB | SSIM: {metrics['ssim']:.4f}]",
        f"Pseudo-HR Ground Truth (10m)\n[{hr_tile.shape[0]}x{hr_tile.shape[1]} px, Reference]",
        f"Ensemble Disagreement Map\n[Uncertainty Proxy (sigma), Mean: {np.mean(disagreement_map):.4f}]",
        f"True Residual Error Map\n[MAE: {metrics['mae']:.4f}]",
    ]

    # Panel 0: LR
    axes[0].imshow(lr_rgb_disp)
    # Panel 1: Ensemble Mean
    axes[1].imshow(mean_rgb)
    # Panel 2: Ground Truth
    axes[2].imshow(hr_rgb)
    # Panel 3: Disagreement Heatmap
    im_disag = axes[3].imshow(disagreement_map, cmap="magma")
    cbar3 = plt.colorbar(im_disag, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar3.ax.axes, "yticklabels"), color="white")

    # Panel 4: True Error Heatmap
    im_err = axes[4].imshow(residual_error, cmap="inferno")
    cbar4 = plt.colorbar(im_err, ax=axes[4], fraction=0.046, pad=0.04)
    cbar4.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar4.ax.axes, "yticklabels"), color="white")

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard — Ensemble Reconstruction & Uncertainty Proxy (Sample #{sample_index})",
        color="white",
        fontsize=13,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def run_ensemble_inference(
    config_path: Optional[str] = None,
    num_samples: int = 4,
) -> int:
    print("=" * 76)
    print("   GeoFUSE SentinelGuard — Ensemble Inference & Uncertainty Analysis")
    print("=" * 76)

    root = get_project_root()
    config = load_config(config_path)
    device = get_device(config)

    checkpoints_dir = root / config.get("paths", {}).get("checkpoints_dir", "checkpoints")
    if not (checkpoints_dir / "ensemble_member_0.pth").exists():
        checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    previews_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "previews"

    # Identify ensemble member checkpoints
    ckpt_paths = [
        checkpoints_dir / "ensemble_member_0.pth",
        checkpoints_dir / "ensemble_member_1.pth",
        checkpoints_dir / "ensemble_member_2.pth",
    ]

    for p in ckpt_paths:
        if not p.exists():
            print(f"[ERROR] Checkpoint missing: {p.resolve()}")
            print("Please run ensemble training first before running inference.")
            return 1

    # Load ensemble
    print(f"\n[1/3] Loading {len(ckpt_paths)} ensemble members onto {device}...")
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    print(f"      Successfully loaded {len(models)} models into memory.")

    # Load data
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, meta = load_sentinel2_stack(raw_dir)
    print(f"      Loaded test scene: {stack.shape}, CRS: {meta['crs']}")

    # Extract sample tiles (including the geographic hold-out southeast quadrant)
    patch_size = 128
    tiles = extract_tiles(stack, patch_size=patch_size, stride=96)

    # Pick representative samples
    sample_indices = [
        0,                      # Northwest corner
        len(tiles) // 3,        # Central road corridor
        (2 * len(tiles)) // 3,  # Agricultural border
        len(tiles) - 1,         # Southeast validation hold-out
    ][:num_samples]

    print(f"\n[2/3] Running ensemble inference on {len(sample_indices)} test tiles...")
    print("-" * 76)
    print(f"{'Sample':<8} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Disag Mean':<12} | {'Disag Max':<10} | {'Disag Std':<10}")
    print("-" * 76)

    for idx, t_idx in enumerate(sample_indices):
        hr_tile = tiles[t_idx]["data"]
        lr_tile = synthesize_pseudo_lr(hr_tile, downsample_factor=2, blur_kernel_size=3, noise_std=0.01, seed=777 + idx)

        # Run ensemble prediction
        mean_recon, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)

        # Metrics
        metrics = evaluate_reconstruction_fidelity(hr_tile, mean_recon, data_range=1.0)

        disag_mean = float(np.mean(disagreement_map))
        disag_max = float(np.max(disagreement_map))
        disag_std = float(np.std(disagreement_map))

        # Checklist verification:
        # Disagreement map should show spatial variation (not uniformly zero and not uniformly maxed out)
        if disag_max == 0.0 or disag_std == 0.0:
            print(f"[WARNING] Disagreement map is uniformly zero in Sample #{idx}!")
            return 2

        print(
            f"#{idx:<7} | "
            f"{metrics['psnr_db']:<10.2f} | "
            f"{metrics['ssim']:<8.4f} | "
            f"{disag_mean:<12.5f} | "
            f"{disag_max:<10.5f} | "
            f"{disag_std:<10.5f}"
        )

        out_path = previews_dir / f"ensemble_preview_sample_{idx}.png"
        render_ensemble_figure(
            lr_tile=lr_tile,
            ensemble_mean=mean_recon,
            hr_tile=hr_tile,
            disagreement_map=disagreement_map,
            metrics=metrics,
            sample_index=idx,
            output_path=out_path,
        )

    print("-" * 76)
    print(f"\n[3/3] Preview figures saved to: {previews_dir.resolve()}")
    for idx in range(len(sample_indices)):
        print(f"  - ensemble_preview_sample_{idx}.png")

    print("\n[SUCCESS] Phase 5 Ensemble Inference and Disagreement Heatmap verified!")
    return 0


def main() -> int:
    return run_ensemble_inference()


if __name__ == "__main__":
    sys.exit(main())
