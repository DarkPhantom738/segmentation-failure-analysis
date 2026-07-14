"""Shared helpers for matched-random probe screens."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.analysis.semantic_directions import SemanticDirection
from src.models.unet3d import UNet3D

PUBLICATION_RC = {"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10}
N_PER_BIN = 10


def select_stratified_cases(
    failure_df: pd.DataFrame, n_per_bin: int = N_PER_BIN, seed: int = 42
) -> pd.DataFrame:
    """Pick low / medium / high baseline-Dice cases (tertiles)."""
    ranked = failure_df.sort_values("dice").reset_index(drop=True)
    n = len(ranked)
    bins = {
        "low": ranked.iloc[: n // 3],
        "medium": ranked.iloc[n // 3 : 2 * n // 3],
        "high": ranked.iloc[2 * n // 3 :],
    }
    rng = np.random.default_rng(seed)
    chosen: list[pd.DataFrame] = []
    for label, sub in bins.items():
        idx = rng.choice(len(sub), size=min(n_per_bin, len(sub)), replace=False)
        part = sub.iloc[idx].copy()
        part["dice_bin"] = label
        chosen.append(part)
    return pd.concat(chosen, ignore_index=True)


def screening_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def unit_random_direction(n_channels: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(n_channels).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def mean_gap(patches: list[torch.Tensor]) -> np.ndarray:
    gaps = [
        torch.nn.functional.adaptive_avg_pool3d(p.float(), 1).flatten().cpu().numpy()
        for p in patches
    ]
    return np.mean(np.stack(gaps, axis=0), axis=0)


def mean_perturbation_ratio(
    model: UNet3D,
    patches: list[torch.Tensor],
    channel_delta: np.ndarray,
    alpha: float,
) -> float:
    ratios: list[float] = []
    delta = torch.as_tensor(channel_delta, dtype=torch.float32)
    for patch in patches:
        edited = model.apply_channel_perturbation(patch, delta, alpha)
        ratios.append(float(model.rms_perturbation_ratio(patch, edited).item()))
    return float(np.mean(ratios))


def analytical_probe_predict(gap: np.ndarray, direction: SemanticDirection) -> float:
    gap = np.asarray(gap, dtype=np.float64).reshape(-1)
    x_scaled = (gap - direction.scaler_mean) / direction.scaler_scale
    return float(np.dot(direction.probe_coef_scaled, x_scaled) + direction.probe_intercept)


def analytical_probe_delta(
    gap_before: np.ndarray,
    direction: SemanticDirection,
    alpha: float,
    channel_delta: np.ndarray,
) -> float:
    """Expected Δ probe prediction from pooled activation + applied perturbation."""
    gap_before = np.asarray(gap_before, dtype=np.float64).reshape(-1)
    gap_after = gap_before + float(alpha) * np.asarray(channel_delta, dtype=np.float64).reshape(
        -1
    )
    return analytical_probe_predict(gap_after, direction) - analytical_probe_predict(
        gap_before, direction
    )


def analytical_probe_before_after(
    gap_before: np.ndarray,
    direction: SemanticDirection,
    alpha: float,
    channel_delta: np.ndarray,
) -> tuple[float, float, float]:
    gap_before = np.asarray(gap_before, dtype=np.float64).reshape(-1)
    gap_after = gap_before + float(alpha) * np.asarray(channel_delta, dtype=np.float64).reshape(
        -1
    )
    pred_before = analytical_probe_predict(gap_before, direction)
    pred_after = analytical_probe_predict(gap_after, direction)
    return pred_before, pred_after, pred_after - pred_before


def load_existing_probe_rows(
    editing_df: pd.DataFrame,
    case_ids: list[str],
    direction_id: str,
    metric: str,
    alpha: float,
) -> pd.DataFrame:
    """Join probe metric rows with Dice rows from a prior full editing run."""
    sub = editing_df[
        (editing_df["direction_id"] == direction_id)
        & (editing_df["metric"] == metric)
        & (editing_df["alpha"].isin([-alpha, alpha]))
        & (editing_df["case_id"].isin(case_ids))
    ].copy()
    dice = editing_df[
        (editing_df["direction_id"] == direction_id)
        & (editing_df["metric"] == "dice")
        & (editing_df["alpha"].isin([-alpha, alpha]))
        & (editing_df["case_id"].isin(case_ids))
    ][["case_id", "alpha", "baseline_value", "edited_value", "delta"]].rename(
        columns={
            "baseline_value": "dice_before",
            "edited_value": "dice_after",
            "delta": "dice_delta",
        }
    )
    return sub.merge(dice, on=["case_id", "alpha"], how="left")
