"""Test-time augmentation (TTA) for validation predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.training.inference import sliding_window_inference


@dataclass(frozen=True)
class FlipAugmentation:
    """Single or combined spatial flip augmentation."""

    name: str
    flip_z: bool
    flip_y: bool
    flip_x: bool


# Standard 8-fold flip TTA.
STANDARD_TTA_AUGMENTATIONS: tuple[FlipAugmentation, ...] = (
    FlipAugmentation("identity", False, False, False),
    FlipAugmentation("flip_x", False, False, True),
    FlipAugmentation("flip_y", False, True, False),
    FlipAugmentation("flip_z", True, False, False),
    FlipAugmentation("flip_xy", False, True, True),
    FlipAugmentation("flip_xz", True, False, True),
    FlipAugmentation("flip_yz", True, True, False),
    FlipAugmentation("flip_xyz", True, True, True),
)


def apply_spatial_flip(
    tensor: torch.Tensor,
    flip_z: bool,
    flip_y: bool,
    flip_x: bool,
) -> torch.Tensor:
    """
    Flip the last three spatial dimensions of a tensor.

    Supports (C, D, H, W) image/probability tensors where:
      - z = depth (D)
      - y = height (H)
      - x = width (W)
    """
    spatial_start = tensor.ndim - 3
    dims: list[int] = []
    if flip_z:
        dims.append(spatial_start)
    if flip_y:
        dims.append(spatial_start + 1)
    if flip_x:
        dims.append(spatial_start + 2)
    if not dims:
        return tensor
    return torch.flip(tensor, dims=dims)


def predictive_entropy(
    probabilities: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-voxel predictive entropy from class probabilities.

    Args:
        probabilities: (num_classes, D, H, W) softmax probabilities.

    Returns:
        (D, H, W) entropy map.
    """
    return -(probabilities * torch.log(probabilities + eps)).sum(dim=0)


@torch.no_grad()
def tta_sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    device: torch.device,
    overlap: float = 0.5,
    augmentations: Sequence[FlipAugmentation] = STANDARD_TTA_AUGMENTATIONS,
    max_augmentations: int | None = None,
) -> torch.Tensor:
    """
    Run sliding-window inference under multiple flip augmentations and average
    the reversed softmax probability maps.

    Args:
        model: Segmentation model.
        image: (C, D, H, W) input volume.
        patch_size: Patch spatial dimensions.
        num_classes: Number of segmentation classes.
        device: Torch device.
        overlap: Sliding-window overlap fraction.
        augmentations: Flip augmentations to apply.

    Returns:
        Mean softmax probabilities with shape (num_classes, D, H, W).
    """
    model.eval()
    probability_maps: list[torch.Tensor] = []

    if max_augmentations is not None:
        augmentations = augmentations[:max_augmentations]

    for augmentation in tqdm(
        augmentations,
        desc="TTA augmentations",
        leave=False,
        disable=len(augmentations) <= 1,
    ):
        augmented_image = apply_spatial_flip(
            image,
            flip_z=augmentation.flip_z,
            flip_y=augmentation.flip_y,
            flip_x=augmentation.flip_x,
        )
        logits, _, _ = sliding_window_inference(
            model=model,
            image=augmented_image,
            patch_size=patch_size,
            num_classes=num_classes,
            overlap=overlap,
            device=device,
        )
        probabilities = F.softmax(logits, dim=0)
        # Reverse the augmentation so all maps align in original orientation.
        aligned_probabilities = apply_spatial_flip(
            probabilities,
            flip_z=augmentation.flip_z,
            flip_y=augmentation.flip_y,
            flip_x=augmentation.flip_x,
        )
        probability_maps.append(aligned_probabilities)

    return torch.stack(probability_maps, dim=0).mean(dim=0)


def tta_predict_segmentation(mean_probabilities: torch.Tensor) -> np.ndarray:
    """Derive the final hard segmentation from TTA-averaged probabilities."""
    return torch.argmax(mean_probabilities, dim=0).cpu().numpy().astype(np.uint8)


def tta_predict_probabilities(mean_probabilities: torch.Tensor) -> np.ndarray:
    """Convert TTA-averaged probabilities to NumPy."""
    return mean_probabilities.cpu().numpy().astype(np.float32)


def tta_predict_entropy(mean_probabilities: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Compute and return the voxelwise predictive entropy map."""
    entropy = predictive_entropy(mean_probabilities, eps=eps)
    return entropy.cpu().numpy().astype(np.float32)
