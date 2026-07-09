"""Sliding-window inference for full 3D volumes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run patch-based inference over a full volume and aggregate outputs.

    Args:
        model: Segmentation model returning (logits, embedding, bottleneck).
        image: (C, D, H, W) input tensor.
        patch_size: Patch spatial dimensions (D, H, W).
        num_classes: Number of segmentation classes.
        overlap: Fractional overlap between neighboring patches.
        device: Torch device.

    Returns:
        logits: (num_classes, D, H, W) aggregated logits
        embedding: (embedding_dim,) case embedding approximated as the mean of
            sliding-window patch embeddings (not a single full-volume GAP pass)
        bottleneck: (C, d, h, w) mean bottleneck map (for debugging; not saved by default)
    """
    if device is None:
        device = image.device

    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)  # (1, C, D, H, W)
    _, _, depth, height, width = image.shape

    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)

    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)

    embeddings: list[torch.Tensor] = []
    bottleneck_maps: list[torch.Tensor] = []

    z_steps = _axis_steps(depth, patch_size[0], stride[0])
    y_steps = _axis_steps(height, patch_size[1], stride[1])
    x_steps = _axis_steps(width, patch_size[2], stride[2])

    # Collect patch predictions; tqdm gives progress on long validation runs.
    total_patches = len(z_steps) * len(y_steps) * len(x_steps)
    patch_iter = tqdm(
        _patch_indices(z_steps, y_steps, x_steps),
        total=total_patches,
        desc="Sliding-window inference",
        leave=False,
    )

    model.eval()
    with torch.no_grad():
        for z0, y0, x0 in patch_iter:
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            patch_logits, patch_embedding, patch_bottleneck = model(patch)

            # Valid region inside the volume (handles volumes smaller than patch).
            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0
            embeddings.append(patch_embedding[0].cpu())
            bottleneck_maps.append(patch_bottleneck[0].cpu())

    logits = logits_sum / count_map.clamp_min(1.0)
    embedding = torch.stack(embeddings, dim=0).mean(dim=0)
    bottleneck = torch.stack(bottleneck_maps, dim=0).mean(dim=0)
    return logits, embedding, bottleneck


def predict_segmentation(logits: torch.Tensor) -> np.ndarray:
    """Convert logits to hard class predictions."""
    return torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)


def predict_probabilities(logits: torch.Tensor) -> np.ndarray:
    """Convert logits to softmax class probabilities."""
    probs = F.softmax(logits, dim=0)
    return probs.cpu().numpy().astype(np.float32)


def _axis_steps(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    steps = list(range(0, size - patch + 1, stride))
    if steps[-1] != size - patch:
        steps.append(size - patch)
    return steps


def _patch_indices(
    z_steps: list[int],
    y_steps: list[int],
    x_steps: list[int],
) -> list[tuple[int, int, int]]:
    return [(z, y, x) for z in z_steps for y in y_steps for x in x_steps]


def _extract_patch(
    image: torch.Tensor,
    z0: int,
    y0: int,
    x0: int,
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Extract a patch, zero-padding when the volume is smaller than patch_size."""
    _, channels, depth, height, width = image.shape
    patch = torch.zeros(
        (1, channels, patch_size[0], patch_size[1], patch_size[2]),
        device=image.device,
        dtype=image.dtype,
    )

    z1 = min(z0 + patch_size[0], depth)
    y1 = min(y0 + patch_size[1], height)
    x1 = min(x0 + patch_size[2], width)
    # Copy valid region; remainder stays zero-padded.
    patch[:, :, : z1 - z0, : y1 - y0, : x1 - x0] = image[:, :, z0:z1, y0:y1, x0:x1]
    return patch
