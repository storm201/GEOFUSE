"""Unit tests for Phase 4: Training Pipeline, Compound Loss, and Geographic Partitioning."""

import numpy as np
import pytest
import torch

from src.data.dataset import SentinelSRDataset
from src.models.loss import CompoundSRLoss, SobelGradientLoss
from src.models.model import ResidualSRNet


def test_sobel_gradient_loss():
    loss_fn = SobelGradientLoss(channels=4)

    # Constant images: gradient should be exactly 0
    const_img1 = torch.full((1, 4, 32, 32), 0.5, dtype=torch.float32)
    const_img2 = torch.full((1, 4, 32, 32), 0.5, dtype=torch.float32)
    loss_zero = loss_fn(const_img1, const_img2)
    assert loss_zero.item() == pytest.approx(0.0, abs=1e-6)

    # Edge images: gradient should be > 0
    edge_img = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    edge_img[:, :, :, 16:] = 1.0
    loss_edge = loss_fn(edge_img, const_img1)
    assert loss_edge.item() > 0.0


def test_compound_loss_components():
    loss_fn = CompoundSRLoss(channels=4, grad_weight=0.1, fft_weight=0.0, lap_weight=0.0, var_weight=0.0)
    pred = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randn(2, 4, 32, 32, requires_grad=False)

    total_loss, loss_dict = loss_fn(pred, target)

    assert total_loss.dim() == 0  # Scalar
    assert "l1" in loss_dict
    assert "grad" in loss_dict
    assert "total" in loss_dict
    assert loss_dict["total"] == pytest.approx(
        loss_dict["l1"] + 0.1 * loss_dict["grad"], rel=1e-4
    )

    # Verify backpropagation
    total_loss.backward()
    assert pred.grad is not None


def test_geographic_partitioning_disjoint():
    # 512x512 multi-band dummy scene
    dummy_scene = np.random.uniform(0.1, 0.9, size=(512, 512, 4)).astype(np.float32)
    val_quadrant = (256, 512, 256, 512)  # SE quadrant
    patch_size = 128
    stride = 64

    train_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
    )

    val_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
    )

    assert len(train_ds) > 0, "Train dataset should contain patches"
    assert len(val_ds) > 0, "Validation dataset should contain patches"

    train_centers = set()
    for rec in train_ds.patches:
        cy = rec["y"] + patch_size // 2
        cx = rec["x"] + patch_size // 2
        train_centers.add((cy, cx))
        # Ensure no train center is inside val quadrant
        assert not (256 <= cy < 512 and 256 <= cx < 512), (
            f"Train patch center ({cy}, {cx}) leaked into validation quadrant!"
        )

    val_centers = set()
    for rec in val_ds.patches:
        cy = rec["y"] + patch_size // 2
        cx = rec["x"] + patch_size // 2
        val_centers.add((cy, cx))
        # Ensure every val center is inside val quadrant
        assert (256 <= cy < 512 and 256 <= cx < 512), (
            f"Validation patch center ({cy}, {cx}) outside validation quadrant!"
        )

    # Completely disjoint
    overlap = train_centers.intersection(val_centers)
    assert len(overlap) == 0, f"Found overlapping patch centers: {overlap}"


def test_dataset_item_shapes():
    dummy_scene = np.random.uniform(0.1, 0.9, size=(512, 512, 4)).astype(np.float32)
    val_quadrant = (256, 512, 256, 512)

    ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=128,
        stride=64,
        split="val",
        val_quadrant=val_quadrant,
        downsample_factor=2,
    )

    lr_t, hr_t, meta = ds[0]
    assert lr_t.shape == (4, 64, 64), f"Unexpected LR shape: {lr_t.shape}"
    assert hr_t.shape == (4, 128, 128), f"Unexpected HR shape: {hr_t.shape}"
    assert meta["split"] == "val"
