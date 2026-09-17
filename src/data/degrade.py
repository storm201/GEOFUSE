"""Synthetic Degradation and Bicubic Baseline Module for GeoFUSE SentinelGuard.

Provides:
1. Synthetic degrade-and-recover pipeline:
   - Downsampling (e.g., 2x) via bicubic interpolation
   - Mild Gaussian blur simulating sensor point spread function (PSF)
   - Additive Gaussian sensor noise (configurable std)
2. 2x Bicubic upsampling baseline to establish benchmark fidelity to beat.
3. Quantitative baseline metrics (PSNR, SSIM, MAE) computed per-tile.
"""

from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def bicubic_downsample(image: np.ndarray, scale_factor: Union[int, float] = 2) -> np.ndarray:
    """Downsample an image by scale_factor using bicubic interpolation.

    Args:
        image: Array of shape (H, W) or (H, W, C) in float or uint format.
        scale_factor: Float or integer downsampling factor (e.g., 2 or 2.5).

    Returns:
        np.ndarray: Downsampled array of shape (H_down, W_down, [C]).
    """
    h, w = image.shape[:2]
    new_h = int(round(h / scale_factor))
    new_w = int(round(w / scale_factor))
    downsampled = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )
    return downsampled


def bicubic_upsample(
    image: np.ndarray,
    scale_factor: Union[int, float] = 2,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Upsample an image by scale_factor using bicubic interpolation.

    Args:
        image: Array of shape (H, W) or (H, W, C).
        scale_factor: Float or integer upsampling factor (e.g., 2 or 2.5).
        target_shape: Optional (target_h, target_w) to match target dimensions exactly.

    Returns:
        np.ndarray: Upsampled array of shape (target_h, target_w, [C]).
    """
    if target_shape is not None:
        target_h, target_w = target_shape
    else:
        h, w = image.shape[:2]
        target_h = int(round(h * scale_factor))
        target_w = int(round(w * scale_factor))

    upsampled = cv2.resize(
        image, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )
    return upsampled


def apply_sensor_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 0.5) -> np.ndarray:
    """Apply mild Gaussian blur simulating sensor point-spread function (PSF).

    Args:
        image: Array of shape (H, W) or (H, W, C).
        kernel_size: Gaussian kernel size (must be positive odd integer).
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Blurred array.
    """
    if kernel_size <= 1:
        return image.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return blurred


def add_sensor_noise(image: np.ndarray, noise_std: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
    """Add zero-mean Gaussian noise to simulate sensor radiometric noise.

    Operates in the normalized reflectance scale [0, 1] or raw DN range.

    Args:
        image: Array of shape (H, W) or (H, W, C).
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        np.ndarray: Noisy array clipped to valid range [0, max(image)].
    """
    if noise_std <= 0.0:
        return image.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=image.shape).astype(image.dtype)
    noisy = image + noise

    # Preserve physical non-negativity
    if np.issubdtype(image.dtype, np.floating):
        noisy = np.clip(noisy, 0.0, None)
    else:
        noisy = np.clip(noisy, 0, 65535)

    return noisy


def synthesize_pseudo_lr(
    hr_tile: np.ndarray,
    downsample_factor: int = 2,
    blur_kernel_size: int = 3,
    blur_sigma: float = 0.5,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a pseudo-LR tile from a pseudo-HR tile using synthetic degradation.

    Degradation pipeline:
    1. Mild Gaussian blur (simulates sensor optical point-spread function)
    2. Bicubic spatial downsampling (simulates coarser spatial resolution)
    3. Mild Gaussian noise (simulates sensor radiometric noise)

    Args:
        hr_tile: Multi-band pseudo-HR tile (e.g. 64x64x4 or 128x128x4).
        downsample_factor: Downscaling factor (e.g., 2 for 2x super-resolution).
        blur_kernel_size: Optical blur kernel size.
        blur_sigma: Optical blur sigma.
        noise_std: Radiometric noise standard deviation.
        seed: Random seed for reproducible noise.

    Returns:
        np.ndarray: Degraded pseudo-LR tile.
    """
    # 1. Optical PSF blur before sampling
    blurred = apply_sensor_blur(hr_tile, kernel_size=blur_kernel_size, sigma=blur_sigma)

    # 2. Downsampling
    lr = bicubic_downsample(blurred, scale_factor=downsample_factor)

    # 3. Radiometric sensor noise
    if noise_std > 0:
        lr = add_sensor_noise(lr, noise_std=noise_std, seed=seed)

    return lr


def evaluate_reconstruction_fidelity(
    ground_truth_hr: np.ndarray,
    reconstructed_hr: np.ndarray,
    data_range: Optional[float] = None,
) -> Dict[str, float]:
    """Compute standard quantitative fidelity metrics between ground truth and reconstruction.

    Args:
        ground_truth_hr: Original HR tile (H, W, C) or (H, W).
        reconstructed_hr: Reconstructed / baseline HR tile of identical shape.
        data_range: Dynamic range of input data (e.g., 1.0 for normalized reflectance,
                    or 10000.0 for raw Sentinel-2 DN).

    Returns:
        Dict[str, float]: Metrics dictionary including PSNR (dB), SSIM, and MAE.
    """
    if ground_truth_hr.shape != reconstructed_hr.shape:
        raise ValueError(
            f"Shape mismatch: ground truth {ground_truth_hr.shape} vs "
            f"reconstruction {reconstructed_hr.shape}"
        )

    gt = ground_truth_hr.astype(np.float64)
    rec = reconstructed_hr.astype(np.float64)

    if data_range is None:
        data_range = float(np.max(gt) - np.min(gt))
        if data_range <= 0:
            data_range = 1.0

    # Mean Absolute Error
    mae = float(np.mean(np.abs(gt - rec)))

    # PSNR (handle exact match without divide-by-zero warning)
    mse = float(np.mean((gt - rec) ** 2))
    if mse < 1e-12:
        psnr = 99.0
    else:
        psnr = float(compute_psnr(gt, rec, data_range=data_range))

    # Multi-channel SSIM
    if gt.ndim == 3 and gt.shape[2] > 1:
        ssim_val = float(compute_ssim(gt, rec, data_range=data_range, channel_axis=2))
    else:
        ssim_val = float(compute_ssim(gt, rec, data_range=data_range))

    return {
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim_val, 4),
        "mae": round(mae, 4),
    }
