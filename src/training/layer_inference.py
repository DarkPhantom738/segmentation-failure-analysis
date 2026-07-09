"""Sliding-window inference with multi-layer activation extraction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.models.unet3d import LAYER_NAMES, UNet3D
from src.training.inference import (
    _axis_steps,
    _extract_patch,
    _patch_indices,
    predict_probabilities,
    predict_segmentation,
)


def sliding_window_layer_inference(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Run patch-based inference and aggregate per-layer globally pooled embeddings.

    Each layer vector is the mean of patch-level vectors across the sliding window,
    matching how single bottleneck embeddings are exported elsewhere in the pipeline.
    """
    if device is None:
        device = image.device

    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image.shape

    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)

    layer_sums: dict[str, list[torch.Tensor]] = {name: [] for name in LAYER_NAMES}

    z_steps = _axis_steps(depth, patch_size[0], stride[0])
    y_steps = _axis_steps(height, patch_size[1], stride[1])
    x_steps = _axis_steps(width, patch_size[2], stride[2])
    total_patches = len(z_steps) * len(y_steps) * len(x_steps)

    model.eval()
    with torch.no_grad():
        patch_iter = tqdm(
            _patch_indices(z_steps, y_steps, x_steps),
            total=total_patches,
            desc="Sliding-window layer inference",
            leave=False,
        )
        for z0, y0, x0 in patch_iter:
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            patch_logits, patch_layers = model.forward_with_layer_embeddings(patch)

            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0

            for name in LAYER_NAMES:
                layer_sums[name].append(patch_layers[name][0].cpu())

    logits = logits_sum / count_map.clamp_min(1.0)
    layer_embeddings = {
        name: torch.stack(layer_sums[name], dim=0).mean(dim=0) for name in LAYER_NAMES
    }
    return logits, layer_embeddings


def layer_inference_outputs(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convenience wrapper returning numpy prediction, probabilities, and layer vectors."""
    logits, layer_embeddings = sliding_window_layer_inference(
        model=model,
        image=image,
        patch_size=patch_size,
        num_classes=num_classes,
        overlap=overlap,
        device=device,
    )
    prediction = predict_segmentation(logits)
    probabilities = predict_probabilities(logits)
    layer_np = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in layer_embeddings.items()
    }
    return prediction, probabilities, layer_np
