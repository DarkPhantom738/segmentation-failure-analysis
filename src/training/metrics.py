"""Segmentation evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch


def whole_tumor_mask(segmentation: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Build the BraTS whole-tumor (WT) binary mask from multi-class labels.

    WT includes necrosis (1), edema (2), and enhancing tumor (3).
    """
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.detach().cpu().numpy()
    return (segmentation > 0).astype(np.uint8)


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + target.sum() + eps))


def whole_tumor_dice(
    prediction: np.ndarray | torch.Tensor,
    ground_truth: np.ndarray | torch.Tensor,
) -> float:
    """Dice score on the BraTS whole-tumor region."""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    pred_wt = whole_tumor_mask(prediction)
    gt_wt = whole_tumor_mask(ground_truth)
    return dice_score(pred_wt, gt_wt)
