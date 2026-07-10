"""Sliding-window inference with single-layer activation ablations."""

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


def sliding_window_intervention_inference(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    ablate_layer: str,
    intervention: str,
    overlap: float = 0.5,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, float, float]:
    """
    Run patch-based inference with one layer activation ablated.

    Returns:
        logits, mean_rho, std_rho (rho aggregated across patches)
    """
    if device is None:
        device = image.device

    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image.shape

    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    logits_sum = torch.zeros((num_classes, depth, height, width), device=device)
    count_map = torch.zeros((1, depth, height, width), device=device)
    rhos: list[float] = []

    z_steps = _axis_steps(depth, patch_size[0], stride[0])
    y_steps = _axis_steps(height, patch_size[1], stride[1])
    x_steps = _axis_steps(width, patch_size[2], stride[2])
    total_patches = len(z_steps) * len(y_steps) * len(x_steps)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        patch_iter = tqdm(
            _patch_indices(z_steps, y_steps, x_steps),
            total=total_patches,
            desc=f"Ablate {ablate_layer}/{intervention}",
            leave=False,
        )
        for z0, y0, x0 in patch_iter:
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            patch_logits, patch_rho = model.forward_with_intervention(
                patch,
                ablate_layer=ablate_layer,
                intervention=intervention,
                generator=generator,
            )
            rhos.append(patch_rho)

            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    rho_arr = np.asarray(rhos, dtype=np.float64)
    mean_rho = float(rho_arr.mean()) if len(rho_arr) else 0.0
    std_rho = float(rho_arr.std()) if len(rho_arr) else 0.0
    return logits_sum / count_map.clamp_min(1.0), mean_rho, std_rho


def sliding_window_baseline_inference(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Sliding-window inference without layer ablation.

    Uses the same patch grid and overlap as ``sliding_window_intervention_inference``
    so ablated outputs are compared to a matched non-TTA baseline.
    """
    if device is None:
        device = image.device

    patch_size = tuple(int(s) for s in patch_size)
    image = image.unsqueeze(0).to(device)
    _, _, depth, height, width = image.shape

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
            desc="Matched baseline",
            leave=False,
        )
        for z0, y0, x0 in patch_iter:
            patch = _extract_patch(image, z0, y0, x0, patch_size)
            patch_logits, _, _ = model(patch)

            valid_d = min(patch_size[0], depth - z0)
            valid_h = min(patch_size[1], height - y0)
            valid_w = min(patch_size[2], width - x0)
            z1, y1, x1 = z0 + valid_d, y0 + valid_h, x0 + valid_w

            logits_sum[:, z0:z1, y0:y1, x0:x1] += patch_logits[
                0, :, :valid_d, :valid_h, :valid_w
            ]
            count_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    return logits_sum / count_map.clamp_min(1.0)


def baseline_prediction(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> np.ndarray:
    """Hard segmentation from matched sliding-window inference (no ablation)."""
    logits = sliding_window_baseline_inference(
        model=model,
        image=image,
        patch_size=patch_size,
        num_classes=num_classes,
        overlap=overlap,
        device=device,
    )
    return predict_segmentation(logits).astype(np.int16)


def intervention_prediction(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    ablate_layer: str,
    intervention: str,
    overlap: float = 0.5,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Return hard segmentation and patch-aggregated rho from ablated inference."""
    logits, mean_rho, std_rho = sliding_window_intervention_inference(
        model=model,
        image=image,
        patch_size=patch_size,
        num_classes=num_classes,
        ablate_layer=ablate_layer,
        intervention=intervention,
        overlap=overlap,
        device=device,
        seed=seed,
    )
    return predict_segmentation(logits).astype(np.int16), mean_rho, std_rho
