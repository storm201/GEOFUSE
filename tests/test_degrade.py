"""Unit tests for Phase 2: Synthetic degradation and bicubic baseline."""

import numpy as np
import pytest

from src.data.degrade import (
    add_sensor_noise,
    apply_sensor_blur,
    bicubic_downsample,
    bicubic_upsample,
    evaluate_reconstruction_fidelity,
    synthesize_pseudo_lr,
)


def test_bicubic_downsample_and_upsample():
    # 64x64 multi-channel patch
    patch = np.random.uniform(0.1, 0.8, size=(64, 64, 4)).astype(np.float32)

    lr = bicubic_downsample(patch, scale_factor=2)
    assert lr.shape == (32, 32, 4)

    recon = bicubic_upsample(lr, scale_factor=2, target_shape=(64, 64))
    assert recon.shape == (64, 64, 4)


def test_bicubic_downsample_and_upsample_fractional():
    # 160x160 patch downsampled by 2.5x -> 64x64
    patch = np.random.uniform(0.1, 0.8, size=(160, 160, 4)).astype(np.float32)

    lr = bicubic_downsample(patch, scale_factor=2.5)
    assert lr.shape == (64, 64, 4)

    recon = bicubic_upsample(lr, scale_factor=2.5, target_shape=(160, 160))
    assert recon.shape == (160, 160, 4)


def test_apply_sensor_blur():
    # Step edge image
    patch = np.zeros((32, 32, 4), dtype=np.float32)
    patch[:, 16:, :] = 1.0

    blurred = apply_sensor_blur(patch, kernel_size=3, sigma=0.5)
    assert blurred.shape == patch.shape
    # Boundary pixel between 15 and 16 should be smoothed
    assert 0.0 < blurred[16, 16, 0] < 1.0
    assert 0.0 < blurred[16, 15, 0] < 1.0


def test_add_sensor_noise():
    patch = np.full((32, 32, 4), 0.5, dtype=np.float32)
    noisy = add_sensor_noise(patch, noise_std=0.02, seed=42)

    assert noisy.shape == patch.shape
    assert not np.array_equal(patch, noisy)
    assert np.all(noisy >= 0.0)  # Non-negative reflectance preserved
    # Mean of noise should be close to 0
    diff = noisy - patch
    assert abs(np.mean(diff)) < 0.01


def test_synthesize_pseudo_lr_pipeline():
    # 64x64 HR tile
    hr = np.random.uniform(0.1, 0.9, size=(64, 64, 4)).astype(np.float32)
    lr = synthesize_pseudo_lr(
        hr_tile=hr,
        downsample_factor=2,
        blur_kernel_size=3,
        noise_std=0.01,
        seed=123,
    )
    assert lr.shape == (32, 32, 4)
    assert not np.isnan(lr).any()
    assert not np.isinf(lr).any()


def test_evaluate_reconstruction_fidelity():
    hr = np.random.uniform(0.1, 0.9, size=(64, 64, 4)).astype(np.float32)
    # Perfect reconstruction
    perfect_metrics = evaluate_reconstruction_fidelity(hr, hr, data_range=1.0)
    assert perfect_metrics["ssim"] == 1.0
    assert perfect_metrics["mae"] == 0.0

    # Perturbed reconstruction
    perturbed = hr + 0.05
    perturbed_metrics = evaluate_reconstruction_fidelity(hr, perturbed, data_range=1.0)
    assert perturbed_metrics["ssim"] < 1.0
    assert perturbed_metrics["mae"] == pytest.approx(0.05, abs=1e-4)
    assert perturbed_metrics["psnr_db"] < 40.0
