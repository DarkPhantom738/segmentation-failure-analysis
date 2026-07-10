"""Sliding-window inference with cached decoder2 activations."""

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


def capture_decoder2_patches(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
) -> tuple[
    list[tuple[int, int, int]],
    list[torch.Tensor],
    list[torch.Tensor],
    tuple[int, int, int],
]:
    """
    One encoder pass: cache decoder2 maps and encoder1 skip tensors per patch.

    Returns origins, d2 patches, s1 skip patches, volume shape (D,H,W).
    """
    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image.shape

    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    z_steps = _axis_steps(depth, patch_size[0], stride[0])
    y_steps = _axis_steps(height, patch_size[1], stride[1])
    x_steps = _axis_steps(width, patch_size[2], stride[2])
    origins = list(_patch_indices(z_steps, y_steps, x_steps))

    d2_patches: list[torch.Tensor] = []
    s1_patches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for z0, y0, x0 in tqdm(origins, desc="Cache decoder2", leave=False):
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            d2, s1 = model.forward_to_decoder2(patch)
            d2_patches.append(d2)
            s1_patches.append(s1)

    return origins, d2_patches, s1_patches, (depth, height, width)


def downstream_from_decoder2_cache(
    model: UNet3D,
    origins: list[tuple[int, int, int]],
    d2_patches: list[torch.Tensor],
    s1_patches: list[torch.Tensor],
    volume_shape: tuple[int, int, int],
    patch_size: Sequence[int],
    num_classes: int,
    channel_delta: np.ndarray | torch.Tensor,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Apply decoder2 perturbation and run decoder1 + seg_head on cached activations."""
    patch_size = tuple(int(s) for s in patch_size)
    depth, height, width = volume_shape
    delta = torch.as_tensor(channel_delta, dtype=torch.float32, device=device).flatten()

    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)

    model.eval()
    with torch.no_grad():
        for (z0, y0, x0), d2, s1 in zip(origins, d2_patches, s1_patches):
            d2 = d2.to(device)
            s1 = s1.to(device)
            edited = model.apply_channel_perturbation(d2, delta, alpha)
            patch_logits = model.forward_from_decoder2(edited, s1)

            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    return logits_sum / count_map.clamp_min(1e-8)


def downstream_prediction_from_decoder2_cache(
    model: UNet3D,
    origins: list[tuple[int, int, int]],
    d2_patches: list[torch.Tensor],
    s1_patches: list[torch.Tensor],
    volume_shape: tuple[int, int, int],
    patch_size: Sequence[int],
    num_classes: int,
    channel_delta: np.ndarray,
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    logits = downstream_from_decoder2_cache(
        model=model,
        origins=origins,
        d2_patches=d2_patches,
        s1_patches=s1_patches,
        volume_shape=volume_shape,
        patch_size=patch_size,
        num_classes=num_classes,
        channel_delta=channel_delta,
        alpha=alpha,
        device=device,
    )
    return predict_segmentation(logits).astype(np.int16)
