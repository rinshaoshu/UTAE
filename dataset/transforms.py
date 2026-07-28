import json
import torch
import numpy as np
from typing import Optional, Tuple


def _flip_h(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (width dimension)."""
    return x.flip(-1)


def _flip_v(x: torch.Tensor) -> torch.Tensor:
    """Vertical flip (height dimension)."""
    return x.flip(-2)


def _rot90k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate by k*90 degrees."""
    if k % 4 == 0:
        return x
    return torch.rot90(x, k, dims=(-2, -1))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() >= self.p:
            return img, msk
        img = _flip_h(img)
        msk = _flip_h(msk)
        return img, msk


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() >= self.p:
            return img, msk
        img = _flip_v(img)
        msk = _flip_v(msk)
        return img, msk


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() >= self.p:
            return img, msk
        k = int(torch.randint(1, 4, (1,)).item())
        img = _rot90k(img, k)
        msk = _rot90k(msk, k)
        return img, msk



class Normalize:
    """Normalize image tensor using mean and std from JSON file.
    Supports any number of channels, just needs matching statistics.
    """
    def __init__(self, norm_json: str, clip_data: bool = False, clip_range: Tuple[float, float] = None):
        """
        Args:
            norm_json: Path to JSON file containing mean and std lists
            clip_data: Whether to clip data before normalization
            clip_range: (min, max) tuple for clipping if clip_data is True
        """
        with open(norm_json) as f:
            self.norm_stats = json.load(f)
        
        # Convert to tensors
        self.mean = torch.tensor(self.norm_stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(self.norm_stats["std"], dtype=torch.float32)
        
        self.clip_data = clip_data
        self.clip_range = clip_range

    def __call__(self, img: torch.Tensor, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img: [T, C, H, W] tensor
            msk: [T, C, H, W] tensor (or any shape, will be returned unchanged)
        Returns:
            Normalized img and unchanged msk
        """
        C = img.shape[1]
        
        # Check if channel count matches
        if C != len(self.mean) or C != len(self.std):
            raise ValueError(
                f"Channel mismatch: img has {C} channels, "
                f"mean has {len(self.mean)} channels, "
                f"std has {len(self.std)} channels"
            )
        
        # Clip if enabled
        if self.clip_data and self.clip_range is not None:
            clip_min, clip_max = self.clip_range
            img = torch.clamp(img, min=clip_min, max=clip_max)
        
        # Reshape for broadcasting: [C] -> [1, C, 1, 1]
        mean = self.mean.view(1, C, 1, 1)
        std = self.std.view(1, C, 1, 1)
        
        # Normalize
        img = (img - mean) / std
        
        return img, msk



class Compose:
    """Compose several transforms that all take/return img and msk."""

    def __init__(self, transforms: list):
        self.transforms = list(transforms)

    def __call__(self, img: torch.Tensor, msk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            img, msk = t(img, msk)
        return img, msk

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {inner}\n])"