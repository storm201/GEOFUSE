"""PyTorch Dataset for Sentinel-2 Super-Resolution with Geographic Partitioning.

Enforces strict geographic hold-out splitting to prevent spatial autocorrelation
and data leakage between training and validation sets.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.degrade import synthesize_pseudo_lr


class SentinelSRDataset(Dataset):
    """Dataset yielding paired (pseudo-LR, pseudo-HR) Sentinel-2 multi-spectral patches.

    Args:
        full_image: Multi-band array of shape (H, W, C) in normalized reflectance [0, 1].
        patch_size_hr: Dimension of the high-resolution patch (e.g., 128 for 64x64 LR at 2x).
        stride: Extraction step size between adjacent patches.
        split: Mode, either 'train' or 'val'.
        val_quadrant: Bounding box (row_min, row_max, col_min, col_max) defining the
                      independent geographic validation hold-out zone.
        downsample_factor: Spatial scale factor (default: 2).
        blur_kernel_size: Gaussian optical PSF blur size.
        noise_std: Additive sensor radiometric noise standard deviation.
        seed: Base random seed for reproducible noise degradation.
    """

    def __init__(
        self,
        full_image: np.ndarray,
        patch_size_hr: int = 128,
        stride: int = 64,
        split: str = "train",
        val_quadrant: Tuple[int, int, int, int] = (256, 512, 256, 512),
        downsample_factor: Union[int, float] = 2.0,
        blur_kernel_size: int = 3,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert split in ("train", "val"), f"Invalid split: {split}. Must be 'train' or 'val'."
        self.split = split
        self.patch_size_hr = patch_size_hr
        self.downsample_factor = downsample_factor
        self.blur_kernel_size = blur_kernel_size
        self.noise_std = noise_std
        self.seed = seed

        val_r_min, val_r_max, val_c_min, val_c_max = val_quadrant
        h, w, c = full_image.shape

        # Extract all candidate patch coordinates
        y_coords = list(range(0, h - patch_size_hr + 1, stride))
        x_coords = list(range(0, w - patch_size_hr + 1, stride))
        if y_coords[-1] != h - patch_size_hr:
            y_coords.append(h - patch_size_hr)
        if x_coords[-1] != w - patch_size_hr:
            x_coords.append(w - patch_size_hr)

        self.patches: List[Dict[str, Any]] = []

        for y in y_coords:
            for x in x_coords:
                patch_center_y = y + patch_size_hr // 2
                patch_center_x = x + patch_size_hr // 2

                # Check if patch center lies within the geographic validation hold-out
                in_val_zone = (
                    val_r_min <= patch_center_y < val_r_max
                    and val_c_min <= patch_center_x < val_c_max
                )

                if (split == "val" and in_val_zone) or (split == "train" and not in_val_zone):
                    hr_patch = full_image[y : y + patch_size_hr, x : x + patch_size_hr, :].copy()
                    self.patches.append({
                        "hr": hr_patch,
                        "y": y,
                        "x": x,
                        "size": patch_size_hr,
                    })

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.patches[idx]
        hr_patch = record["hr"].copy()

        # Data augmentation on training partition (dihedral symmetry group D4)
        if self.split == "train":
            if np.random.rand() > 0.5:
                hr_patch = np.fliplr(hr_patch).copy()
            if np.random.rand() > 0.5:
                hr_patch = np.flipud(hr_patch).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                hr_patch = np.rot90(hr_patch, k=k).copy()

        # Synthesize degraded pseudo-LR tile
        lr_patch = synthesize_pseudo_lr(
            hr_tile=hr_patch,
            downsample_factor=self.downsample_factor,
            blur_kernel_size=self.blur_kernel_size,
            noise_std=self.noise_std,
            seed=self.seed + idx if self.split == "val" else None,
        )

        # Convert to PyTorch format (Channels, Height, Width)
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).contiguous().float()
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).contiguous().float()

        meta = {
            "y": record["y"],
            "x": record["x"],
            "split": self.split,
        }
        return lr_tensor, hr_tensor, meta
