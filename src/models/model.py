"""Direct High-Frequency Super-Resolution Architecture for GeoFUSE SentinelGuard.

Designed specifically for 4.0m resolution satellite super-resolution:
- Multi-scale LR feature extraction (6 residual blocks with LeakyReLU)
- Direct feature projection to target 4.0m spatial grid (no lossy bilinear downsampling)
- High-frequency synthesis backbone (4 residual blocks operating directly at target 4.0m resolution)
- Active residual gain parameter (res_gain) to synthesize true edge contrast (rooftops, roads)
- Lightweight parameter budget: ~800K parameters (< 1.5M ceiling)
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention for spectral band weighting."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions, LeakyReLU and Channel Attention."""

    def __init__(self, channels: int, res_scale: float = 0.2) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(self.body(x))
        return x + out * self.res_scale


class ResidualSRNet(nn.Module):
    """Direct 4.0m Multi-Spectral Super-Resolution Network.

    Args:
        in_channels: Number of input spectral bands (4 for B02, B03, B04, B08).
        out_channels: Number of output spectral bands (matches in_channels).
        num_features: Intermediate feature depth (default: 64).
        scale_factor: Spatial upsampling factor (2.5 for 4.0m GSD from Sentinel-2 10m).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        num_features: int = 64,
        num_blocks: int = 6,
        scale_factor: Union[int, float] = 2.5,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = float(scale_factor)

        # 1. LR Feature Extraction Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # 2. LR Deep Contextual Backbone
        self.lr_blocks = nn.Sequential(
            *[ResidualBlock(channels=num_features, res_scale=res_scale) for _ in range(num_blocks)]
        )
        self.lr_trunk = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)

        # 3. Target Grid High-Resolution Synthesis Backbone (Operates directly at 4.0m GSD)
        self.hr_blocks = nn.Sequential(
            *[ResidualBlock(channels=num_features, res_scale=res_scale) for _ in range(4)]
        )

        # 4. Multi-Scale Detail Reconstruction Head
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_features // 2, out_channels, kernel_size=3, padding=1, bias=True),
        )

        # 5. Learnable residual amplification gain (initialized to 1.5 to recover crisp edge transitions)
        self.res_gain = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (Batch, Channels, H, W).

        Returns:
            torch.Tensor: Super-resolved tensor of shape (Batch, Channels, H * scale, W * scale).
        """
        target_h = int(round(x.shape[2] * self.scale_factor))
        target_w = int(round(x.shape[3] * self.scale_factor))

        # Base interpolation provides smooth photometric baseline
        base = F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        # Low-resolution feature extraction
        f_init = self.head(x)
        f_lr = self.lr_trunk(self.lr_blocks(f_init)) + f_init

        # Project feature representation to target 4.0m spatial grid
        f_hr = F.interpolate(
            f_lr,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        # High-frequency edge and texture synthesis directly at 4.0m resolution
        f_sharp = self.hr_blocks(f_hr)
        residual_hr = self.tail(f_sharp)

        # Reconstructed HR = Base + Amplified High-Frequency Detail
        out = base + residual_hr * self.res_gain
        return out


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: Optional[Dict[str, Any]] = None) -> ResidualSRNet:
    """Factory function to instantiate ResidualSRNet from project config.yaml."""
    if config is None:
        return ResidualSRNet()

    model_cfg = config.get("model", {})
    return ResidualSRNet(
        in_channels=model_cfg.get("num_channels", 4),
        out_channels=model_cfg.get("num_channels", 4),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_residual_blocks", 6),
        scale_factor=model_cfg.get("scale_factor", 2.5),
        res_scale=model_cfg.get("res_scale", 0.2),
    )
