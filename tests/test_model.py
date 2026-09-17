"""Unit tests for Phase 3: Lightweight Super-Resolution Model."""

import torch
import pytest

from src.models.model import ResidualSRNet, count_parameters, build_model
from src.utils.config import load_config


def test_model_forward_pass_shape():
    """Verify that 2x forward pass transforms (B, C, H, W) to (B, C, 2H, 2W)."""
    batch_size = 2
    in_channels = 4
    h, w = 32, 32
    scale_factor = 2

    model = ResidualSRNet(
        in_channels=in_channels,
        out_channels=in_channels,
        num_features=48,
        num_blocks=6,
        scale_factor=scale_factor,
    )
    model.eval()

    dummy_lr = torch.randn(batch_size, in_channels, h, w, dtype=torch.float32)

    with torch.no_grad():
        out_sr = model(dummy_lr)

    expected_shape = (batch_size, in_channels, h * scale_factor, w * scale_factor)
    assert out_sr.shape == expected_shape, (
        f"Output shape mismatch: Expected {expected_shape}, got {out_sr.shape}"
    )
    assert not torch.isnan(out_sr).any(), "Output contains NaN values"
    assert not torch.isinf(out_sr).any(), "Output contains Inf values"


def test_model_fractional_forward_pass_shape():
    """Verify that 2.5x fractional forward pass transforms (B, C, 64, 64) to (B, C, 160, 160)."""
    model = ResidualSRNet(
        in_channels=4,
        out_channels=4,
        num_features=48,
        num_blocks=4,
        scale_factor=2.5,
    )
    model.eval()
    dummy_lr = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        out_sr = model(dummy_lr)
    assert out_sr.shape == (2, 4, 160, 160)
    assert not torch.isnan(out_sr).any()
    assert not torch.isinf(out_sr).any()


def test_model_parameter_count_budget():
    """Verify parameter count is under the ~1.5M hardware budget."""
    model = ResidualSRNet(
        in_channels=4,
        out_channels=4,
        num_features=48,
        num_blocks=6,
        scale_factor=2,
    )
    num_params = count_parameters(model)
    print(f"\n[Model Parameter Check] Total trainable parameters: {num_params:,}")

    # Maximum parameter budget allowed by prompt
    max_budget = 1_500_000
    assert num_params < max_budget, (
        f"Parameter count ({num_params:,}) exceeds budget ({max_budget:,})"
    )
    # Ensure it is a non-trivial model (> 100k params)
    assert num_params > 100_000, f"Parameter count ({num_params:,}) unexpectedly small"


def test_model_backward_gradient_flow():
    """Verify that gradients propagate to all layers during backprop."""
    model = ResidualSRNet(
        in_channels=4,
        out_channels=4,
        num_features=32,
        num_blocks=4,
        scale_factor=2,
    )
    model.train()

    dummy_lr = torch.randn(1, 4, 16, 16, requires_grad=False)
    dummy_hr = torch.randn(1, 4, 32, 32, requires_grad=False)

    out = model(dummy_lr)
    loss = torch.nn.functional.l1_loss(out, dummy_hr)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in: {name}"


def test_build_model_from_config():
    """Verify model construction from project config.yaml."""
    config = load_config()
    model = build_model(config)
    assert isinstance(model, ResidualSRNet)
    assert model.scale_factor == config.get("model", {}).get("scale_factor", 2)
    assert model.in_channels == config.get("model", {}).get("num_channels", 4)
