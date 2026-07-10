"""Sliding-window inference with cached decoder1 activations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.models.unet3d import UNet3D
from src.training.inference import (
    _axis_steps,
    _extract_patch,
    _patch_indices,
    predict_segmentation,
)


def capture_decoder1_patches(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
) -> tuple[list[tuple[int, int, int]], list[torch.Tensor], tuple[int, int, int]]:
    """
    One encoder pass: cache decoder1 activation tensor per sliding-window patch.

    Returns patch origins, decoder1 maps (B,C,D,H,W), and volume shape (D,H,W).
    """
    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image.shape

    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    z_steps = _axis_steps(depth, patch_size[0], stride[0])
    y_steps = _axis_steps(height, patch_size[1], stride[1])
    x_steps = _axis_steps(width, patch_size[2], stride[2])
    origins = list(_patch_indices(z_steps, y_steps, x_steps))

    d1_patches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for z0, y0, x0 in tqdm(origins, desc="Cache decoder1", leave=False):
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            d1_patches.append(model.forward_to_decoder1(patch))

    return origins, d1_patches, (depth, height, width)


def downstream_from_decoder1_cache(
    model: UNet3D,
    origins: list[tuple[int, int, int]],
    d1_patches: list[torch.Tensor],
    volume_shape: tuple[int, int, int],
    patch_size: Sequence[int],
    num_classes: int,
    channel_delta: np.ndarray | torch.Tensor,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Apply a decoder1 perturbation and run seg_head only on cached activations."""
    patch_size = tuple(int(s) for s in patch_size)
    depth, height, width = volume_shape
    delta = torch.as_tensor(channel_delta, dtype=torch.float32, device=device).flatten()

    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)

    model.eval()
    with torch.no_grad():
        for (z0, y0, x0), d1 in zip(origins, d1_patches):
            d1 = d1.to(device)
            edited = model.apply_channel_perturbation(d1, delta, alpha)
            patch_logits = model.forward_from_decoder1(edited)

            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    return logits_sum / count_map.clamp_min(1e-8)


def downstream_prediction_from_cache(
    model: UNet3D,
    origins: list[tuple[int, int, int]],
    d1_patches: list[torch.Tensor],
    volume_shape: tuple[int, int, int],
    patch_size: Sequence[int],
    num_classes: int,
    channel_delta: np.ndarray,
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    logits = downstream_from_decoder1_cache(
        model=model,
        origins=origins,
        d1_patches=d1_patches,
        volume_shape=volume_shape,
        patch_size=patch_size,
        num_classes=num_classes,
        channel_delta=channel_delta,
        alpha=alpha,
        device=device,
    )
    return predict_segmentation(logits).astype(np.int16)
