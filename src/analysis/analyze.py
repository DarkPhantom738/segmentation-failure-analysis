"""Per-case failure labels and entropy–error metrics for the validation table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.failure_labels import (
    boundary_error_fraction,
    build_analysis_region,
    compute_error_mask,
    confident_false_negative_fraction,
    missed_small_lesion_count,
)
from src.analysis.uncertainty_metrics import (
    binary_auprc,
    binary_auroc,
    mean_entropy_inside_mask,
    uncertainty_error_overlap,
)
from src.training.metrics import whole_tumor_dice
from src.utils.io import ensure_dir


def analyze_case(
    case_id: str,
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    entropy_path: str | Path,
    embedding_path: str | Path,
    boundary_iterations: int = 2,
    small_lesion_voxel_threshold: int = 200,
    miss_fraction_threshold: float = 0.5,
    low_entropy_percentile: float = 25.0,
) -> dict[str, float | int | str]:
    """Compute failure labels and uncertainty-error metrics for one case."""
    ground_truth = np.load(ground_truth_path)
    prediction = np.load(prediction_path)
    entropy = np.load(entropy_path)

    error_mask, false_positive_mask, false_negative_mask = compute_error_mask(
        ground_truth, prediction
    )
    analysis_region = build_analysis_region(ground_truth, prediction)

    dice = whole_tumor_dice(prediction, ground_truth)
    total_error_voxels = int(error_mask.sum())

    row: dict[str, float | int | str] = {
        "case_id": case_id,
        "dice": dice,
        "boundary_error_fraction": boundary_error_fraction(
            error_mask,
            ground_truth,
            structure_iterations=boundary_iterations,
        ),
        "false_positive_voxels": int(false_positive_mask.sum()),
        "false_negative_voxels": int(false_negative_mask.sum()),
        "confident_false_negative_fraction": confident_false_negative_fraction(
            false_negative_mask,
            entropy,
            analysis_region=analysis_region,
            low_percentile=low_entropy_percentile,
        ),
        "missed_small_lesion_count": missed_small_lesion_count(
            ground_truth,
            prediction,
            small_lesion_voxel_threshold=small_lesion_voxel_threshold,
            miss_fraction_threshold=miss_fraction_threshold,
        ),
        # Entropy-error metrics are restricted to the analysis region
        # (GT tumor | predicted tumor) so background-dominated voxels do not
        # inflate AUROC / non-error entropy.
        "mean_entropy_error": mean_entropy_inside_mask(
            entropy, error_mask & analysis_region
        ),
        "mean_entropy_nonerror": mean_entropy_inside_mask(
            entropy, (~error_mask) & analysis_region
        ),
        "entropy_error_auroc": binary_auroc(
            error_mask[analysis_region], entropy[analysis_region]
        ),
        "entropy_error_auprc": binary_auprc(
            error_mask[analysis_region], entropy[analysis_region]
        ),
        "overlap_top_5_uncertainty": uncertainty_error_overlap(
            error_mask, entropy, top_fraction=0.05, analysis_region=analysis_region
        ),
        "overlap_top_10_uncertainty": uncertainty_error_overlap(
            error_mask, entropy, top_fraction=0.10, analysis_region=analysis_region
        ),
        "overlap_top_20_uncertainty": uncertainty_error_overlap(
            error_mask, entropy, top_fraction=0.20, analysis_region=analysis_region
        ),
        "path_embedding": str(embedding_path),
        "path_entropy": str(entropy_path),
        "path_prediction": str(prediction_path),
        "path_ground_truth": str(ground_truth_path),
    }

    # Keep extra diagnostics out of the CSV but useful during development.
    row["_total_error_voxels"] = total_error_voxels
    return row


def analyze_cases_from_metrics(
    metrics_path: str | Path,
    output_path: str | Path,
    boundary_iterations: int = 2,
    small_lesion_voxel_threshold: int = 200,
    miss_fraction_threshold: float = 0.5,
    low_entropy_percentile: float = 25.0,
) -> pd.DataFrame:
    """
    Analyze all cases listed in a TTA metrics CSV (from ``train.py --export-tta``).

    If multiple epochs are present, the latest row per case_id is used.
    """
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    metrics_df = pd.read_csv(metrics_path)

    required_columns = {
        "case_id",
        "path_ground_truth",
        "path_prediction",
        "path_entropy",
        "path_embedding",
    }
    missing = required_columns - set(metrics_df.columns)
    if missing:
        raise ValueError(f"Metrics file missing required columns: {sorted(missing)}")

    if "epoch" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("epoch").groupby("case_id", as_index=False).tail(1)

    rows: list[dict[str, float | int | str]] = []
    for record in metrics_df.to_dict(orient="records"):
        row = analyze_case(
            case_id=str(record["case_id"]),
            ground_truth_path=record["path_ground_truth"],
            prediction_path=record["path_prediction"],
            entropy_path=record["path_entropy"],
            embedding_path=record["path_embedding"],
            boundary_iterations=boundary_iterations,
            small_lesion_voxel_threshold=small_lesion_voxel_threshold,
            miss_fraction_threshold=miss_fraction_threshold,
            low_entropy_percentile=low_entropy_percentile,
        )
        row.pop("_total_error_voxels", None)
        rows.append(row)

    result = pd.DataFrame(rows)
    ensure_dir(output_path.parent)
    result.to_csv(output_path, index=False)
    return result
