"""Uncertainty-error relationship metrics."""

from __future__ import annotations

import numpy as np


def mean_entropy_inside_mask(entropy: np.ndarray, mask: np.ndarray) -> float:
    """Mean entropy over voxels where mask is True."""
    if not mask.any():
        return float("nan")
    return float(entropy[mask].mean())


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Area under the ROC curve for binary labels and continuous scores.

    Higher scores are assumed to indicate the positive class (error voxels).
    """
    labels = y_true.astype(bool).ravel()
    values = scores.ravel()

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank voxels by descending score and sweep the classification threshold.
    order = np.argsort(-values, kind="mergesort")
    labels_sorted = labels[order]

    true_positives = np.cumsum(labels_sorted)
    false_positives = np.cumsum(~labels_sorted)

    tpr = np.concatenate([[0.0], true_positives / n_pos])
    fpr = np.concatenate([[0.0], false_positives / n_neg])
    return float(np.trapezoid(tpr, fpr))


def binary_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve for binary labels and scores."""
    labels = y_true.astype(bool).ravel()
    values = scores.ravel()

    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-values, kind="mergesort")
    labels_sorted = labels[order]

    true_positives = np.cumsum(labels_sorted)
    false_positives = np.cumsum(~labels_sorted)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / n_pos

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


def uncertainty_error_overlap(
    error_mask: np.ndarray,
    entropy: np.ndarray,
    top_fraction: float,
    analysis_region: np.ndarray | None = None,
) -> float:
    """
    Fraction of error voxels captured in the top `top_fraction` uncertain voxels.

    Uncertainty is ranked by entropy inside the analysis region. For example,
    top_fraction=0.05 uses the 5% highest-entropy voxels.
    """
    error_count = int(error_mask.sum())
    if error_count == 0:
        return float("nan")

    if analysis_region is None:
        analysis_region = np.ones_like(error_mask, dtype=bool)

    region_entropy = entropy[analysis_region]
    if region_entropy.size == 0:
        return float("nan")

    percentile = 100.0 * (1.0 - top_fraction)
    threshold = np.percentile(region_entropy, percentile)
    top_uncertain_mask = analysis_region & (entropy >= threshold)
    captured_errors = int((error_mask & top_uncertain_mask).sum())
    return captured_errors / error_count


def false_negative_low_entropy_fraction(
    false_negative_mask: np.ndarray,
    entropy: np.ndarray,
    analysis_region: np.ndarray,
    low_percentile: float = 25.0,
) -> float:
    """Fraction of false-negative voxels with entropy below the low percentile."""
    false_negative_count = int(false_negative_mask.sum())
    if false_negative_count == 0:
        return float("nan")

    threshold = np.percentile(entropy[analysis_region], low_percentile)
    low_entropy_false_negatives = false_negative_mask & (entropy < threshold)
    return int(low_entropy_false_negatives.sum()) / false_negative_count


def false_positive_high_entropy_fraction(
    false_positive_mask: np.ndarray,
    entropy: np.ndarray,
    analysis_region: np.ndarray,
    high_percentile: float = 75.0,
) -> float:
    """Fraction of false-positive voxels with entropy above the high percentile."""
    false_positive_count = int(false_positive_mask.sum())
    if false_positive_count == 0:
        return float("nan")

    threshold = np.percentile(entropy[analysis_region], high_percentile)
    high_entropy_false_positives = false_positive_mask & (entropy > threshold)
    return int(high_entropy_false_positives.sum()) / false_positive_count
