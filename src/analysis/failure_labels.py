"""Segmentation failure label computation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.training.metrics import whole_tumor_mask


def compute_error_mask(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute whole-tumor (WT) binary error decomposition.

    Errors are defined on the WT mask (any tumor vs background), not on
    multiclass confusion among necrosis/edema/enhancing labels.

    Returns:
        error_mask: WT false positives or false negatives
        false_positive_mask: predicted tumor but background in GT
        false_negative_mask: GT tumor but predicted background
    """
    gt_tumor = whole_tumor_mask(ground_truth).astype(bool)
    pred_tumor = whole_tumor_mask(prediction).astype(bool)

    false_positive_mask = pred_tumor & ~gt_tumor
    false_negative_mask = gt_tumor & ~pred_tumor
    error_mask = false_positive_mask | false_negative_mask
    return error_mask, false_positive_mask, false_negative_mask


def tumor_boundary_region(
    ground_truth: np.ndarray,
    structure_iterations: int = 2,
) -> np.ndarray:
    """
    Build a boundary band around the ground-truth tumor mask.

    The band is created by subtracting an eroded tumor mask from a dilated
    tumor mask. This captures voxels near the lesion surface.
    """
    gt_tumor = whole_tumor_mask(ground_truth).astype(bool)
    if not gt_tumor.any():
        return np.zeros_like(gt_tumor, dtype=bool)

    structure = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(
        gt_tumor, structure=structure, iterations=structure_iterations
    )
    eroded = ndimage.binary_erosion(
        gt_tumor, structure=structure, iterations=structure_iterations
    )
    return dilated & ~eroded


def boundary_error_fraction(
    error_mask: np.ndarray,
    ground_truth: np.ndarray,
    structure_iterations: int = 2,
) -> float:
    """
    Fraction of all error voxels that fall inside the GT tumor boundary band.

    High values indicate errors concentrated near lesion borders.
    """
    total_errors = int(error_mask.sum())
    if total_errors == 0:
        return 0.0

    boundary = tumor_boundary_region(
        ground_truth, structure_iterations=structure_iterations
    )
    boundary_errors = int((error_mask & boundary).sum())
    return boundary_errors / total_errors


def confident_false_negative_fraction(
    false_negative_mask: np.ndarray,
    entropy: np.ndarray,
    analysis_region: np.ndarray | None = None,
    low_percentile: float = 25.0,
) -> float:
    """
    Fraction of false-negative voxels with low entropy.

    Low entropy is defined as below the given percentile of entropy values
    inside the analysis region (brain/tumor voxels).
    """
    false_negative_count = int(false_negative_mask.sum())
    if false_negative_count == 0:
        return 0.0

    if analysis_region is None:
        analysis_region = np.ones_like(entropy, dtype=bool)

    region_entropy = entropy[analysis_region]
    if region_entropy.size == 0:
        return 0.0

    low_entropy_threshold = np.percentile(region_entropy, low_percentile)
    confident_false_negatives = false_negative_mask & (entropy < low_entropy_threshold)
    return int(confident_false_negatives.sum()) / false_negative_count


def missed_small_lesion_count(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    small_lesion_voxel_threshold: int = 200,
    miss_fraction_threshold: float = 0.5,
) -> int:
    """
    Count small connected GT tumor components that are mostly missed.

    A component is considered missed when less than `miss_fraction_threshold`
    of its voxels are predicted as tumor. Only components with at most
    `small_lesion_voxel_threshold` voxels are counted as small lesions.
    """
    gt_tumor = whole_tumor_mask(ground_truth).astype(np.uint8)
    pred_tumor = whole_tumor_mask(prediction).astype(bool)

    if gt_tumor.sum() == 0:
        return 0

    labeled, num_components = ndimage.label(gt_tumor)
    missed_small_lesions = 0

    for component_id in range(1, num_components + 1):
        component_mask = labeled == component_id
        component_size = int(component_mask.sum())
        if component_size > small_lesion_voxel_threshold:
            continue

        predicted_fraction = pred_tumor[component_mask].mean()
        if predicted_fraction < miss_fraction_threshold:
            missed_small_lesions += 1

    return missed_small_lesions


def build_analysis_region(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """
    Region used for entropy percentile calculations.

    Uses the union of GT and predicted tumor voxels, which approximates the
    brain/tumor region in cropped BraTS volumes.
    """
    gt_tumor = whole_tumor_mask(ground_truth).astype(bool)
    pred_tumor = whole_tumor_mask(prediction).astype(bool)
    return gt_tumor | pred_tumor
