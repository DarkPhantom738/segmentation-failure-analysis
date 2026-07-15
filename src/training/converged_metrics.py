"""Development metrics for converged U-Net training."""

from __future__ import annotations

import numpy as np

from src.training.metrics import dice_score, whole_tumor_dice

# BraTS remapped labels used throughout this repository.
CLS_NECROTIC = 1
CLS_EDEMA = 2
CLS_ENHANCING = 3


def class_dice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_id: int,
) -> float:
    """
    Binary Dice for a single class.

    Absent-class rule (shared by all seeds):
      - Both empty (class absent in GT and prediction): Dice = 1.0
        (implemented by dice_score eps → (eps)/(eps) = 1).
      - GT empty, prediction non-empty: Dice → 0 (false-positive-only).
      - GT non-empty, prediction empty: Dice → 0 (complete miss).

    These cases are always included in the mean; we never drop missing classes.
    """
    pred = prediction == class_id
    gt = ground_truth == class_id
    return dice_score(pred, gt)


def necrotic_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return class_dice(prediction, ground_truth, CLS_NECROTIC)


def edema_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return class_dice(prediction, ground_truth, CLS_EDEMA)


def enhancing_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    return class_dice(prediction, ground_truth, CLS_ENHANCING)


def mean_foreground_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Mean of necrotic, edema, and enhancing Dice.

    This is the early-stopping / checkpoint-selection metric.
    """
    return float(
        np.mean(
            [
                necrotic_dice(prediction, ground_truth),
                edema_dice(prediction, ground_truth),
                enhancing_dice(prediction, ground_truth),
            ]
        )
    )


def development_dice_bundle(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    return {
        "mean_foreground_dice": mean_foreground_dice(prediction, ground_truth),
        "necrotic_dice": necrotic_dice(prediction, ground_truth),
        "edema_dice": edema_dice(prediction, ground_truth),
        "enhancing_dice": enhancing_dice(prediction, ground_truth),
        "whole_tumor_dice": whole_tumor_dice(prediction, ground_truth),
    }
