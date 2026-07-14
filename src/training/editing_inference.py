"""Sliding-window inference with additive representation editing."""

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


def sliding_window_editing_inference(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    edit_layer: str,
    channel_delta: np.ndarray | torch.Tensor,
    alpha: float,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Patch-based inference with semantic activation editing at one layer."""
    if device is None:
        device = image.device

    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image.shape

    delta = torch.as_tensor(channel_delta, dtype=torch.float32, device=device).flatten()

    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)

    z_steps = _axis_steps(depth, patch_size[0], stride[0])
    y_steps = _axis_steps(height, patch_size[1], stride[1])
    x_steps = _axis_steps(width, patch_size[2], stride[2])
    total_patches = len(z_steps) * len(y_steps) * len(x_steps)

    model.eval()
    with torch.no_grad():
        patch_iter = tqdm(
            _patch_indices(z_steps, y_steps, x_steps),
            total=total_patches,
            desc=f"Edit {edit_layer} a={alpha:+.1f}",
            leave=False,
        )
        for z0, y0, x0 in patch_iter:
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            patch_logits = model.forward_with_representation_edit(
                patch,
                edit_layer=edit_layer,
                channel_delta=delta,
                alpha=alpha,
            )

            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    return logits_sum / count_map.clamp_min(1.0)


def editing_prediction(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    edit_layer: str,
    channel_delta: np.ndarray,
    alpha: float,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return hard segmentation from an edited forward pass."""
    logits = sliding_window_editing_inference(
        model=model,
        image=image,
        patch_size=patch_size,
        num_classes=num_classes,
        edit_layer=edit_layer,
        channel_delta=channel_delta,
        alpha=alpha,
        overlap=overlap,
        device=device,
    )
    return predict_segmentation(logits).astype(np.int16)
