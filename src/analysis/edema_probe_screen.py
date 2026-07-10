"""Minimal screening: decoder1 edema probe vs matched random perturbations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.analysis.layer_interventions import compute_prediction_quantities
from src.analysis.semantic_directions import SemanticDirection
from src.data.brats_dataset import BraTSCase
from src.models.unet3d import UNet3D, build_model
from src.training.decoder1_cache_inference import (
    capture_decoder1_patches,
    downstream_prediction_from_cache,
)
from src.utils.io import ensure_dir

PUBLICATION_RC = {"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10}

PROBE_DIRECTION_ID = "decoder1::gt_edema_frac"
PROBE_ALPHA = 1.0
N_RANDOM = 3
N_PER_BIN = 10
EDITING_RESULTS = Path("outputs_10hour/representation_editing/editing_results.csv")
DIRECTION_PATH = Path(
    "outputs_10hour/semantic_directions/directions/decoder1__gt_edema_frac.npz"
)


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


def _screening_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _unit_random_direction(n_channels: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(n_channels).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def _mean_gap(d1_patches: list[torch.Tensor]) -> np.ndarray:
    gaps = [
        torch.nn.functional.adaptive_avg_pool3d(d1.float(), 1).flatten().cpu().numpy()
        for d1 in d1_patches
    ]
    return np.mean(np.stack(gaps, axis=0), axis=0)


def _mean_perturbation_ratio(
    model: UNet3D,
    d1_patches: list[torch.Tensor],
    channel_delta: np.ndarray,
    alpha: float,
) -> float:
    ratios: list[float] = []
    delta = torch.as_tensor(channel_delta, dtype=torch.float32)
    for d1 in d1_patches:
        edited = model.apply_channel_perturbation(d1, delta, alpha)
        ratios.append(float(model.rms_perturbation_ratio(d1, edited).item()))
    return float(np.mean(ratios))


def analytical_probe_delta(
    gap_before: np.ndarray,
    direction: SemanticDirection,
    alpha: float,
    channel_delta: np.ndarray,
) -> float:
    """Expected Δ probe prediction from pooled activation and applied perturbation."""
    gap_before = np.asarray(gap_before, dtype=np.float64).reshape(-1)
    gap_after = gap_before + float(alpha) * np.asarray(channel_delta, dtype=np.float64).reshape(-1)
    x_before = (gap_before - direction.scaler_mean) / direction.scaler_scale
    x_after = (gap_after - direction.scaler_mean) / direction.scaler_scale
    pred_before = float(np.dot(direction.probe_coef_scaled, x_before) + direction.probe_intercept)
    pred_after = float(np.dot(direction.probe_coef_scaled, x_after) + direction.probe_intercept)
    return pred_after - pred_before


def _load_existing_probe_rows(editing_df: pd.DataFrame, case_ids: list[str]) -> pd.DataFrame:
    sub = editing_df[
        (editing_df["direction_id"] == PROBE_DIRECTION_ID)
        & (editing_df["metric"] == "edema_fraction")
        & (editing_df["alpha"].isin([-PROBE_ALPHA, PROBE_ALPHA]))
        & (editing_df["case_id"].isin(case_ids))
    ].copy()
    dice = editing_df[
        (editing_df["direction_id"] == PROBE_DIRECTION_ID)
        & (editing_df["metric"] == "dice")
        & (editing_df["alpha"].isin([-PROBE_ALPHA, PROBE_ALPHA]))
        & (editing_df["case_id"].isin(case_ids))
    ][["case_id", "alpha", "baseline_value", "edited_value", "delta"]].rename(
        columns={
            "baseline_value": "dice_before",
            "edited_value": "dice_after",
            "delta": "dice_delta",
        }
    )
    return sub.merge(dice, on=["case_id", "alpha"], how="left")


def run_edema_probe_screen(
    config: dict,
    checkpoint_path: Path,
    failure_table_path: Path,
    output_dir: Path,
    device: torch.device,
    editing_results_path: Path = EDITING_RESULTS,
    direction_path: Path = DIRECTION_PATH,
    n_per_bin: int = N_PER_BIN,
    n_random: int = N_RANDOM,
    random_seed: int = 42,
    overlap: float | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    failure_df = pd.read_csv(failure_table_path)
    selected = select_stratified_cases(failure_df, n_per_bin=n_per_bin, seed=random_seed)
    case_ids = selected["case_id"].tolist()

    editing_df = pd.read_csv(editing_results_path)
    existing_probe = _load_existing_probe_rows(editing_df, case_ids)

    probe_direction = SemanticDirection.load(direction_path)
    probe_delta_vec = probe_direction.activation_direction.astype(np.float64)

    rng = _screening_rng(random_seed)
    random_directions = [_unit_random_direction(probe_direction.activation_channels, rng) for _ in range(n_random)]

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    patch_size = config["data"]["patch_size"]
    num_classes = config["model"]["num_classes"]
    data_cfg = config["data"]
    if overlap is None:
        overlap = float(config.get("uncertainty", {}).get("overlap", 0.25))

    rows: list[dict[str, Any]] = []

    for record in tqdm(selected.to_dict(orient="records"), desc="Edema probe screen"):
        case_id = str(record["case_id"])
        dice_bin = str(record["dice_bin"])
        baseline_dice = float(record["dice"])
        gt = np.load(record["path_ground_truth"]).astype(np.int16)

        loader = BraTSCase(
            case_id=case_id,
            data_root=Path(data_cfg["root"]),
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
        )
        image, _ = loader.load()
        image_tensor = torch.from_numpy(image)

        origins, d1_patches, vol_shape = capture_decoder1_patches(
            model, image_tensor, patch_size, overlap, device
        )
        gap_before = _mean_gap(d1_patches)
        perturbation_ratio = _mean_perturbation_ratio(
            model, d1_patches, probe_delta_vec, PROBE_ALPHA
        )

        baseline_pred = downstream_prediction_from_cache(
            model, origins, d1_patches, vol_shape, patch_size, num_classes,
            probe_delta_vec, 0.0, device,
        )
        baseline_metrics = compute_prediction_quantities(baseline_pred, gt)
        baseline_edema = float(baseline_metrics["edema_fraction"])

        for alpha in (-PROBE_ALPHA, PROBE_ALPHA):
            sign = int(np.sign(alpha))
            ex = existing_probe[
                (existing_probe["case_id"] == case_id)
                & (np.isclose(existing_probe["alpha"], alpha))
            ]
            if not ex.empty:
                edema_before = float(ex["baseline_value"].iloc[0])
                edema_after = float(ex["edited_value"].iloc[0])
                edema_delta = float(ex["delta"].iloc[0])
                dice_before = float(ex["dice_before"].iloc[0])
                dice_after = float(ex["dice_after"].iloc[0])
                dice_delta = float(ex["dice_delta"].iloc[0])
            else:
                edema_before = baseline_edema
                pred = downstream_prediction_from_cache(
                    model, origins, d1_patches, vol_shape, patch_size, num_classes,
                    probe_delta_vec, alpha, device,
                )
                m = compute_prediction_quantities(pred, gt)
                edema_after = float(m["edema_fraction"])
                edema_delta = edema_after - edema_before
                dice_before = baseline_dice
                dice_after = float(m["dice"])
                dice_delta = dice_after - dice_before

            probe_pred = downstream_prediction_from_cache(
                model, origins, d1_patches, vol_shape, patch_size, num_classes,
                probe_delta_vec, alpha, device,
            )
            changed_voxels = int(np.sum(probe_pred != baseline_pred))

            rows.append(
                {
                    "case_id": case_id,
                    "dice_bin": dice_bin,
                    "baseline_dice": baseline_dice,
                    "direction_type": "probe",
                    "random_direction_id": "",
                    "sign": sign,
                    "alpha": alpha,
                    "perturbation_rms_ratio": perturbation_ratio,
                    "edema_fraction_before": edema_before,
                    "edema_fraction_after": edema_after,
                    "edema_fraction_delta": edema_delta,
                    "analytical_probe_pred_delta": analytical_probe_delta(
                        gap_before, probe_direction, alpha, probe_delta_vec
                    ),
                    "dice_before": dice_before,
                    "dice_after": dice_after,
                    "dice_delta": dice_delta,
                    "changed_voxels": changed_voxels,
                    "source": "existing_editing_results",
                }
            )

        for ridx, rnd_dir in enumerate(random_directions):
            for alpha in (-PROBE_ALPHA, PROBE_ALPHA):
                sign = int(np.sign(alpha))
                pred = downstream_prediction_from_cache(
                    model, origins, d1_patches, vol_shape, patch_size, num_classes,
                    rnd_dir, alpha, device,
                )
                m = compute_prediction_quantities(pred, gt)
                edema_after = float(m["edema_fraction"])
                rows.append(
                    {
                        "case_id": case_id,
                        "dice_bin": dice_bin,
                        "baseline_dice": baseline_dice,
                        "direction_type": "random",
                        "random_direction_id": f"random_{ridx:02d}",
                        "sign": sign,
                        "alpha": alpha,
                        "perturbation_rms_ratio": perturbation_ratio,
                        "edema_fraction_before": baseline_edema,
                        "edema_fraction_after": edema_after,
                        "edema_fraction_delta": edema_after - baseline_edema,
                        "analytical_probe_pred_delta": analytical_probe_delta(
                            gap_before, probe_direction, alpha, rnd_dir
                        ),
                        "dice_before": baseline_dice,
                        "dice_after": float(m["dice"]),
                        "dice_delta": float(m["dice"]) - baseline_dice,
                        "changed_voxels": int(np.sum(pred != baseline_pred)),
                        "source": "downstream_from_cached_decoder1",
                    }
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "edema_probe_screen_cases.csv", index=False)

    probe_abs = results_df[results_df["direction_type"] == "probe"]["edema_fraction_delta"].abs().mean()
    random_abs = results_df[results_df["direction_type"] == "random"]["edema_fraction_delta"].abs().mean()
    ratio = probe_abs / max(random_abs, 1e-12)

    summary = pd.DataFrame(
        [
            {
                "n_cases": len(case_ids),
                "n_random_directions": n_random,
                "probe_alpha": PROBE_ALPHA,
                "mean_abs_edema_delta_probe": probe_abs,
                "mean_abs_edema_delta_random": random_abs,
                "probe_to_random_abs_edema_ratio": ratio,
            }
        ]
    )
    summary.to_csv(output_dir / "edema_probe_screen_summary.csv", index=False)

    _plot_abs_edema_comparison(results_df, figures_dir / "abs_edema_delta_probe_vs_random.png")
    _plot_by_dice_bin(results_df, figures_dir / "abs_edema_delta_by_dice_bin.png")

    conclusion = _build_conclusion(results_df, ratio)
    (output_dir / "edema_probe_screen_conclusion.md").write_text(conclusion)

    return {"results": results_df, "summary": summary, "conclusion": conclusion}


def _plot_abs_edema_comparison(results_df: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    probe = results_df[results_df["direction_type"] == "probe"]["edema_fraction_delta"].abs()
    random = results_df[results_df["direction_type"] == "random"]["edema_fraction_delta"].abs()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(
        [probe.values, random.values],
        tick_labels=["probe", "random"],
        patch_artist=True,
    )
    ax.set_ylabel("|Δ edema fraction|")
    ax.set_title("Probe vs random (matched |α|=1 RMS budget)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_by_dice_bin(results_df: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    bins = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(bins))
    probe_means = []
    random_means = []
    for b in bins:
        sub = results_df[results_df["dice_bin"] == b]
        probe_means.append(sub[sub["direction_type"] == "probe"]["edema_fraction_delta"].abs().mean())
        random_means.append(sub[sub["direction_type"] == "random"]["edema_fraction_delta"].abs().mean())
    ax.bar(x - width / 2, probe_means, width, label="probe", color="steelblue")
    ax.bar(x + width / 2, random_means, width, label="random", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("mean |Δ edema fraction|")
    ax.set_title("By baseline Dice bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_conclusion(results_df: pd.DataFrame, ratio: float) -> str:
    probe = results_df[results_df["direction_type"] == "probe"]
    random = results_df[results_df["direction_type"] == "random"]
    probe_abs = probe["edema_fraction_delta"].abs().mean()
    random_abs = random["edema_fraction_delta"].abs().mean()

    clearly = ratio > 1.25 and probe_abs > random_abs * 1.1
    verdict = (
        "**The probe direction clearly outperforms matched random perturbations** "
        if clearly
        else "**The probe direction does not clearly outperform matched random perturbations** "
    )

    return f"""# Edema Probe Screening Conclusion

{verdict}on mean |Δ edema fraction| at |α|=1 with RMS-matched unit-norm directions.

| Metric | Probe | Random (mean) | Ratio |
|--------|------:|--------------:|------:|
| mean \\|Δ edema fraction\\| | {probe_abs:.5f} | {random_abs:.5f} | {ratio:.2f} |

- Probe edema effects loaded from `outputs_10hour/representation_editing/editing_results.csv` (|α|=1).
- Random directions: {N_RANDOM} fixed unit vectors, same |α| magnitude as probe, decoder1→head only from cached activations.
- Analytical probe-prediction Δ uses pooled GAP + scaler + Ridge coef/intercept (no segmentation).

**Interpretation:** A ratio near 1 means the edema shift is generic to any same-sized perturbation; a ratio >1.25 suggests the learned direction is somewhat specific.
"""
