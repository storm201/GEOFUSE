"""Loss functions for GeoFUSE SentinelGuard Super-Resolution Training.

Implements:
1. SobelGradientLoss: Multi-channel spatial gradient magnitude alignment loss.
2. FFTSpectralLoss: 2D Fourier transform spectral magnitude loss for high-frequency detail recovery.
3. LaplacianSharpnessLoss: Second-order differential sharpness penalty.
4. EdgeVarianceLoss: Statistical edge distribution matching to prevent over-smoothing.
5. CompoundSRLoss: Combined high-frequency enforcing loss to achieve crisp, reference-matching <4m resolution.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradientLoss(nn.Module):
    """Multi-channel Sobel gradient loss to penalize blurred edge transitions.

    Computes both directional gradient differences and gradient magnitude matching
    to force sharp edge boundaries (roads, rooftops, parcel lines).

    Args:
        channels: Number of spectral channels (e.g., 4 for B02, B03, B04, B08).
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels

        # Standard 3x3 Sobel kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        weight_x = sobel_x.repeat(channels, 1, 1, 1)
        weight_y = sobel_y.repeat(channels, 1, 1, 1)

        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient difference and magnitude loss."""
        wx = self.weight_x.to(dtype=pred.dtype, device=pred.device)
        wy = self.weight_y.to(dtype=pred.dtype, device=pred.device)

        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        pred_dx = F.conv2d(pred_pad, wx, groups=self.channels)
        pred_dy = F.conv2d(pred_pad, wy, groups=self.channels)

        target_dx = F.conv2d(target_pad, wx, groups=self.channels)
        target_dy = F.conv2d(target_pad, wy, groups=self.channels)

        # Directional gradient difference
        dir_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        # Gradient magnitude contrast matching
        pred_mag = torch.sqrt(pred_dx ** 2 + pred_dy ** 2 + 1e-6)
        target_mag = torch.sqrt(target_dx ** 2 + target_dy ** 2 + 1e-6)
        mag_loss = F.l1_loss(pred_mag, target_mag)

        return dir_loss + 1.5 * mag_loss


class FFTSpectralLoss(nn.Module):
    """2D Fast Fourier Transform spectral magnitude loss.

    Penalizes spectral blur and over-smoothing by comparing the 2D log frequency
    magnitude spectrum between the super-resolved prediction and ground truth.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 2D FFT spectral magnitude loss."""
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")

        pred_mag = torch.log1p(torch.abs(pred_fft) + self.eps)
        target_mag = torch.log1p(torch.abs(target_fft) + self.eps)

        return F.l1_loss(pred_mag, target_mag).to(dtype=pred.dtype)


class LaplacianSharpnessLoss(nn.Module):
    """Second-order Laplacian sharpness loss.

    Penalizes blurry outputs by enforcing second-derivative (curvature) similarity
    and matching the variance of the edge energy distribution.

    Args:
        channels: Number of spectral channels.
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels

        # 3x3 Laplacian kernel
        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        weight = laplacian.repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Laplacian sharpness error and variance difference."""
        w = self.weight.to(dtype=pred.dtype, device=pred.device)

        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        pred_lap = F.conv2d(pred_pad, w, groups=self.channels)
        target_lap = F.conv2d(target_pad, w, groups=self.channels)

        lap_l1 = F.l1_loss(pred_lap, target_lap)

        # Variance matching: forces output to have same sharpness contrast distribution as reference
        p_var = torch.var(pred_lap, dim=(-2, -1))
        t_var = torch.var(target_lap, dim=(-2, -1))
        var_loss = F.l1_loss(p_var, t_var)

        return lap_l1, var_loss


class CompoundSRLoss(nn.Module):
    """Compound Super-Resolution Loss: L1 + Sobel Gradient + FFT Spectral + Laplacian Sharpness + Variance.

    Calibrated specifically for satellite super-resolution to prevent over-smoothing
    and recover crisp edges matching the original high-resolution reference.
    """

    def __init__(
        self,
        channels: int = 4,
        grad_weight: float = 1.5,
        fft_weight: float = 0.5,
        lap_weight: float = 2.0,
        var_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.grad_weight = grad_weight
        self.fft_weight = fft_weight
        self.lap_weight = lap_weight
        self.var_weight = var_weight
        self.l1_loss = nn.L1Loss()
        self.gradient_loss = SobelGradientLoss(channels=channels)
        self.fft_loss = FFTSpectralLoss()
        self.laplacian_loss = LaplacianSharpnessLoss(channels=channels)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute compound sharpness-preserving loss."""
        l1 = self.l1_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        lap, var = self.laplacian_loss(pred, target)
        fft = self.fft_loss(pred, target)

        total = (
            l1
            + self.grad_weight * grad
            + self.lap_weight * lap
            + self.var_weight * var
            + self.fft_weight * fft
        )

        loss_dict: Dict[str, float] = {
            "l1": float(l1.detach().item()),
            "grad": float(grad.detach().item()),
            "lap": float(lap.detach().item()),
            "var": float(var.detach().item()),
            "fft": float(fft.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, loss_dict
