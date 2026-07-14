"""Shared layer-embedding I/O and probe helpers for recoverability / editing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from src.models.unet3d import LAYER_NAMES
from src.training.metrics import whole_tumor_mask

LAYER_ORDER = list(LAYER_NAMES)

ANATOMY_TARGET_SPECS: dict[str, str] = {
    "centroid_x": "Centroid X",
    "centroid_y": "Centroid Y",
    "centroid_z": "Centroid Z",
    "gt_wt_voxels": "Whole-tumor volume",
    "log_wt_volume": "log(WT volume)",
    "gt_edema_voxels": "Edema volume",
    "gt_enhancing_voxels": "Enhancing volume",
    "gt_necrosis_voxels": "Necrosis volume",
    "gt_edema_frac": "Edema fraction",
    "gt_enhancing_frac": "Enhancing fraction",
    "gt_necrosis_frac": "Necrosis fraction",
    "gt_compactness": "Boundary complexity",
    "boundary_to_volume_ratio": "Boundary-to-volume ratio",
    "gt_elongation": "Elongation",
    "gt_n_components": "Multifocality (# components)",
    "false_positive_voxels": "False-positive voxels",
    "false_negative_voxels": "False-negative voxels",
    "boundary_error_fraction": "Boundary error fraction",
    "dice": "Dice",
}

DEPTH_LABELS = {
    "encoder1": "Enc 1",
    "encoder2": "Enc 2",
    "encoder3": "Enc 3",
    "encoder4": "Enc 4",
    "bottleneck": "Bottleneck",
    "decoder4": "Dec 4",
    "decoder3": "Dec 3",
    "decoder2": "Dec 2",
    "decoder1": "Dec 1",
}


def load_layer_index(index_path: str | Path) -> pd.DataFrame:
    """Load the layer embedding index table."""
    return pd.read_csv(index_path)


def load_layer_matrix(index_df: pd.DataFrame, layer_name: str) -> np.ndarray:
    """Stack embeddings for one layer across cases."""
    col = f"path_{layer_name}"
    if col not in index_df.columns:
        raise ValueError(f"Missing column {col} in layer index")
    return np.stack(
        [np.load(path).astype(np.float32) for path in index_df[col]],
        axis=0,
    )


def build_anatomy_table(
    index_df: pd.DataFrame,
    failure_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive anatomical and failure-morphology targets from GT masks and failure metrics."""
    merged = index_df.copy()
    if failure_df is not None:
        failure_cols = [
            c
            for c in failure_df.columns
            if c
            in {
                "case_id",
                "boundary_error_fraction",
                "false_positive_voxels",
                "false_negative_voxels",
                "dice",
            }
        ]
        merged = merged.merge(
            failure_df[failure_cols].drop_duplicates("case_id"),
            on="case_id",
            how="left",
            suffixes=("", "_fail"),
        )
        if "dice_fail" in merged.columns:
            merged["dice"] = merged["dice"].fillna(merged["dice_fail"])
            merged = merged.drop(columns=["dice_fail"])

    rows: list[dict[str, float]] = []
    for record in merged.to_dict(orient="records"):
        gt = np.load(record["path_ground_truth"]).astype(np.int16)
        wt_mask = whole_tumor_mask(gt).astype(bool)
        wt_vol = int(wt_mask.sum())
        nec = int((gt == 1).sum())
        edema = int((gt == 2).sum())
        enh = int((gt == 3).sum())

        if wt_vol > 0:
            eroded = ndimage.binary_erosion(wt_mask)
            surface = int((wt_mask & ~eroded).sum())
            compactness = surface / (wt_vol ** (2.0 / 3.0) + 1e-8)
            boundary_to_volume_ratio = surface / (wt_vol + 1e-8)
            coords = np.argwhere(wt_mask)
            centroid = coords.mean(axis=0) / np.array(gt.shape, dtype=np.float64)
            if len(coords) > 3:
                evals = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]
                elongation = float(evals[0] / (evals[-1] + 1e-8))
            else:
                elongation = 1.0
            _, n_comp = ndimage.label(wt_mask.astype(np.uint8))
        else:
            surface = 0
            compactness = 0.0
            boundary_to_volume_ratio = 0.0
            centroid = np.array([0.5, 0.5, 0.5])
            elongation = 1.0
            n_comp = 0

        rows.append(
            {
                "case_id": record["case_id"],
                "centroid_x": float(centroid[2]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[0]),
                "gt_wt_voxels": float(wt_vol),
                "log_wt_volume": float(np.log1p(wt_vol)),
                "gt_edema_voxels": float(edema),
                "gt_enhancing_voxels": float(enh),
                "gt_necrosis_voxels": float(nec),
                "gt_edema_frac": float(edema / (wt_vol + 1e-8)),
                "gt_enhancing_frac": float(enh / (wt_vol + 1e-8)),
                "gt_necrosis_frac": float(nec / (wt_vol + 1e-8)),
                "gt_compactness": float(compactness),
                "boundary_to_volume_ratio": float(boundary_to_volume_ratio),
                "gt_elongation": float(elongation),
                "gt_n_components": float(n_comp),
                "gt_surface_voxels": float(surface),
                "boundary_error_fraction": float(
                    record.get("boundary_error_fraction", np.nan)
                ),
                "false_positive_voxels": float(
                    record.get("false_positive_voxels", np.nan)
                ),
                "false_negative_voxels": float(
                    record.get("false_negative_voxels", np.nan)
                ),
                "dice": float(record.get("dice", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def prepare_merged(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
) -> pd.DataFrame:
    """Join layer index with failure-table columns (excluding bulky path/dice dupes)."""
    index_df = load_layer_index(layer_index_path)
    failure_df = pd.read_csv(failure_table_path)
    failure_cols = [
        c
        for c in failure_df.columns
        if c
        not in {
            "path_prediction",
            "path_ground_truth",
            "path_entropy",
            "path_embedding",
            "dice",
        }
    ]
    return index_df.merge(
        failure_df[failure_cols].drop_duplicates("case_id"),
        on="case_id",
        how="inner",
    )


def create_bad_case_label(
    dice: np.ndarray,
    mode: str = "threshold",
    threshold: float = 0.80,
    bad_quantile: float | None = None,
    bottom_percentile: float | None = None,
) -> tuple[np.ndarray, float]:
    """Binary label for poor segmentation performance."""
    dice = np.asarray(dice, dtype=np.float64)

    if mode == "bottom_percentile":
        mode = "quantile"
        if bad_quantile is None:
            pct = 25.0 if bottom_percentile is None else float(bottom_percentile)
            bad_quantile = pct / 100.0

    if mode == "threshold":
        effective_threshold = float(threshold)
    elif mode == "quantile":
        if bad_quantile is None:
            bad_quantile = 0.25
        if not 0.0 < float(bad_quantile) < 1.0:
            raise ValueError(f"bad_quantile must be in (0, 1), got {bad_quantile}")
        effective_threshold = float(np.quantile(dice, float(bad_quantile)))
    else:
        raise ValueError(
            f"Unknown bad-case mode: {mode}. Use 'threshold' or 'quantile'."
        )

    labels = (dice < effective_threshold).astype(int)
    if labels.sum() == 0 and len(dice) > 0:
        labels[int(np.argmin(dice))] = 1
        effective_threshold = float(dice[labels.astype(bool)].max())
    return labels, effective_threshold


def choose_cv_folds(labels: np.ndarray, max_folds: int = 5) -> int:
    """Choose a CV fold count valid for the label/sample distribution."""
    n_cases = len(labels)
    if n_cases < 4:
        return 0

    _, class_counts = np.unique(labels, return_counts=True)
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return 0

    return min(max_folds, n_cases, min_class_count)


def impute_nan(features: np.ndarray) -> np.ndarray:
    """Replace NaNs with column means (0 if a column is all-NaN)."""
    cleaned = features.astype(np.float64).copy()
    for col in range(cleaned.shape[1]):
        column = cleaned[:, col]
        if np.isnan(column).all():
            cleaned[:, col] = 0.0
        elif np.isnan(column).any():
            column[np.isnan(column)] = np.nanmean(column)
            cleaned[:, col] = column
    return cleaned.astype(np.float32)
