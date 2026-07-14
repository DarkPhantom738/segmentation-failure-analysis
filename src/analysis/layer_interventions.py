"""Layer ablation interventions for causal probing of U-Net representations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

from src.data.brats_dataset import BraTSCase
from src.models.unet3d import LAYER_NAMES, build_model
from src.training.intervention_inference import (
    baseline_prediction,
    intervention_prediction,
)
from src.training.metrics import whole_tumor_dice, whole_tumor_mask
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}

INTERVENTIONS = ("mean",)

METRIC_SPECS: dict[str, str] = {
    "dice": "Dice",
    "tumor_volume": "Tumor volume (pred WT voxels)",
    "edema_fraction": "Edema fraction",
    "enhancing_fraction": "Enhancing fraction",
    "necrosis_fraction": "Necrosis fraction",
    "centroid_z": "Centroid Z (normalized)",
    "boundary_complexity": "Boundary complexity",
    "false_positive_voxels": "False-positive voxels",
    "false_negative_voxels": "False-negative voxels",
}


def compute_prediction_quantities(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Derive segmentation quantities from prediction compared to GT."""
    pred = prediction.astype(np.int16)
    gt = ground_truth.astype(np.int16)
    pred_wt = whole_tumor_mask(pred).astype(bool)
    gt_wt = whole_tumor_mask(gt).astype(bool)
    pred_vol = int(pred_wt.sum())

    nec = int((pred == 1).sum())
    edema = int((pred == 2).sum())
    enh = int((pred == 3).sum())

    if pred_vol > 0:
        eroded = ndimage.binary_erosion(pred_wt)
        surface = int((pred_wt & ~eroded).sum())
        compactness = surface / (pred_vol ** (2.0 / 3.0) + 1e-8)
        coords = np.argwhere(pred_wt)
        centroid = coords.mean(axis=0) / np.array(pred.shape, dtype=np.float64)
    else:
        compactness = 0.0
        centroid = np.array([0.5, 0.5, 0.5])

    fp = int(np.logical_and(pred_wt, ~gt_wt).sum())
    fn = int(np.logical_and(~pred_wt, gt_wt).sum())

    return {
        "dice": float(whole_tumor_dice(pred, gt)),
        "tumor_volume": float(pred_vol),
        "edema_fraction": float(edema / (pred_vol + 1e-8)),
        "enhancing_fraction": float(enh / (pred_vol + 1e-8)),
        "necrosis_fraction": float(nec / (pred_vol + 1e-8)),
        "centroid_z": float(centroid[0]),
        "boundary_complexity": float(compactness),
        "false_positive_voxels": float(fp),
        "false_negative_voxels": float(fn),
    }


def _degradation_delta(metric: str, baseline: float, ablated: float) -> float:
    """Degradation score: larger positive values mean worse outcome after ablation."""
    if metric == "dice":
        return float(baseline - ablated)
    if metric in ("false_positive_voxels", "false_negative_voxels"):
        return float(ablated - baseline)
    return float(abs(ablated - baseline))


def _case_seed(case_id: str, ablate_layer: str, intervention: str) -> int:
    key = f"{case_id}:{ablate_layer}:{intervention}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest[:8], 16)


def _rho_log_path(output_dir: Path) -> Path:
    return output_dir / "rho_log.csv"


def load_rho_log(output_dir: Path) -> pd.DataFrame:
    path = _rho_log_path(output_dir)
    if not path.exists():
        return pd.DataFrame(
            columns=["case_id", "layer", "intervention", "rho_mean", "rho_std"]
        )
    return pd.read_csv(path)


def _rho_is_logged(rho_df: pd.DataFrame, case_id: str, layer: str, intervention: str) -> bool:
    if rho_df.empty:
        return False
    mask = (
        (rho_df["case_id"] == case_id)
        & (rho_df["layer"] == layer)
        & (rho_df["intervention"] == intervention)
    )
    return bool(mask.any())


def _append_rho_log(
    output_dir: Path,
    rho_df: pd.DataFrame,
    case_id: str,
    layer: str,
    intervention: str,
    rho_mean: float,
    rho_std: float,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "layer": layer,
                "intervention": intervention,
                "rho_mean": rho_mean,
                "rho_std": rho_std,
            }
        ]
    )
    rho_df = pd.concat([rho_df, row], ignore_index=True)
    rho_df = rho_df.drop_duplicates(
        subset=["case_id", "layer", "intervention"], keep="last"
    )
    rho_df.to_csv(_rho_log_path(output_dir), index=False)
    return rho_df


def build_intervention_strength_summary(
    rho_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-layer table with perturbation magnitude rho and normalized effect sizes.

    dice_degradation_per_rho = mean Dice degradation / mean rho (higher = more
    output-sensitive per unit activation perturbation).
    """
    if rho_df.empty or summary_df.empty:
        return pd.DataFrame()

    rho_summary = (
        rho_df.groupby(["layer", "intervention"], as_index=False)
        .agg(mean_rho=("rho_mean", "mean"), std_rho=("rho_mean", "std"), n_cases=("case_id", "nunique"))
    )
    key_metrics = [
        "dice",
        "tumor_volume",
        "edema_fraction",
        "boundary_complexity",
        "false_negative_voxels",
    ]
    rows: list[dict[str, Any]] = []
    for _, rho_row in rho_summary.iterrows():
        layer = str(rho_row["layer"])
        intervention = str(rho_row["intervention"])
        mean_rho = float(rho_row["mean_rho"])
        entry: dict[str, Any] = {
            "layer": layer,
            "intervention": intervention,
            "mean_rho": mean_rho,
            "std_rho": float(rho_row["std_rho"]),
            "n_cases": int(rho_row["n_cases"]),
        }
        for metric in key_metrics:
            sub = summary_df[
                (summary_df["layer"] == layer)
                & (summary_df["intervention"] == intervention)
                & (summary_df["metric"] == metric)
            ]
            if sub.empty:
                continue
            deg = float(sub["mean_degradation"].iloc[0])
            entry[f"mean_{metric}_degradation"] = deg
            entry[f"{metric}_degradation_per_rho"] = deg / max(mean_rho, 1e-6)
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(["intervention", "mean_rho"])


def _build_ablation_artifacts(
    results_df: pd.DataFrame,
    rho_df: pd.DataFrame,
    output_dir: Path,
    n_cases: int,
    baseline_description: str,
) -> dict[str, pd.DataFrame | str]:
    """Aggregate per-case ablation rows into summary tables, figures, and report."""
    figures_dir = ensure_dir(output_dir / "figures")

    ablated_df = results_df[results_df["intervention"] != "baseline"].copy()
    summary_df = (
        ablated_df.groupby(["layer", "intervention", "metric", "metric_label"], as_index=False)
        .agg(
            mean_baseline=("baseline_value", "mean"),
            mean_ablated=("value", "mean"),
            mean_delta=("delta", "mean"),
            mean_degradation=("degradation", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
            std_degradation=("degradation", "std"),
            n_cases=("case_id", "nunique"),
        )
        .sort_values(["layer", "intervention", "mean_degradation"], ascending=[True, True, False])
    )
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    ranking_rows: list[dict[str, Any]] = []
    for (layer, intervention), group in summary_df.groupby(["layer", "intervention"]):
        ranked = group.sort_values("mean_degradation", ascending=False)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            ranking_rows.append(
                {
                    "layer": layer,
                    "intervention": intervention,
                    "metric": row["metric"],
                    "metric_label": row["metric_label"],
                    "rank": rank,
                    "mean_degradation": row["mean_degradation"],
                    "mean_delta": row["mean_delta"],
                    "mean_baseline": row["mean_baseline"],
                    "mean_ablated": row["mean_ablated"],
                }
            )
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(output_dir / "degradation_ranking.csv", index=False)

    strength_df = build_intervention_strength_summary(rho_df, summary_df)
    strength_df.to_csv(output_dir / "intervention_strength_summary.csv", index=False)

    _plot_degradation_heatmap(summary_df, figures_dir / "heatmap_mean_degradation.png")
    _plot_layer_metric_bars(
        summary_df, intervention="mean", output_path=figures_dir / "bar_degradation_mean.png"
    )

    report = _generate_ablation_report(
        summary_df,
        ranking_df,
        strength_df,
        n_cases=n_cases,
        baseline_description=baseline_description,
    )
    (output_dir / "layer_interventions_report.md").write_text(report)

    return {
        "summary": summary_df,
        "ranking": ranking_df,
        "strength": strength_df,
        "report": report,
    }


def compare_dice_rankings(
    old_summary: pd.DataFrame,
    new_summary: pd.DataFrame,
    intervention: str = "mean",
) -> pd.DataFrame:
    """Compare per-layer mean Dice degradation between two ablation summaries."""
    rows: list[dict[str, Any]] = []
    for layer in LAYER_NAMES:
        old_row = old_summary[
            (old_summary["layer"] == layer)
            & (old_summary["intervention"] == intervention)
            & (old_summary["metric"] == "dice")
        ]
        new_row = new_summary[
            (new_summary["layer"] == layer)
            & (new_summary["intervention"] == intervention)
            & (new_summary["metric"] == "dice")
        ]
        if old_row.empty or new_row.empty:
            continue
        old_deg = float(old_row["mean_degradation"].iloc[0])
        new_deg = float(new_row["mean_degradation"].iloc[0])
        rows.append(
            {
                "layer": layer,
                "intervention": intervention,
                "tta_baseline_dice_degradation": old_deg,
                "matched_baseline_dice_degradation": new_deg,
                "delta_degradation": new_deg - old_deg,
            }
        )
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp
    comp["tta_rank"] = comp["tta_baseline_dice_degradation"].rank(ascending=False, method="min")
    comp["matched_rank"] = comp["matched_baseline_dice_degradation"].rank(
        ascending=False, method="min"
    )
    return comp.sort_values("matched_rank")


def recompute_ablation_with_matched_baseline(
    config: dict,
    checkpoint_path: Path,
    failure_table_path: Path,
    output_dir: Path,
    device: torch.device,
    overlap: float | None = None,
    max_cases: int | None = None,
    layers: tuple[str, ...] = LAYER_NAMES,
    interventions: tuple[str, ...] = INTERVENTIONS,
) -> dict[str, Any]:
    """
    Re-score cached ablated predictions against a matched sliding-window baseline.

    Baseline: same overlap and patch grid as ablation inference, no TTA, no ablation.
    Ablated masks are read from ``output_dir/predictions/{layer}/{intervention}/``.
    """
    output_dir = ensure_dir(output_dir)
    matched_dir = ensure_dir(output_dir / "matched_baseline")
    pred_dir = output_dir / "predictions"
    matched_baseline_dir = ensure_dir(pred_dir / "matched_baseline")

    failure_df = pd.read_csv(failure_table_path)
    if max_cases is not None and max_cases > 0:
        failure_df = failure_df.head(max_cases)

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    patch_size = config["data"]["patch_size"]
    num_classes = config["model"]["num_classes"]
    data_cfg = config["data"]
    if overlap is None:
        overlap = float(config.get("inference", config.get("uncertainty", {})).get("overlap", 0.25))

    rho_df = load_rho_log(output_dir)
    result_rows: list[dict[str, Any]] = []

    for record in tqdm(failure_df.to_dict(orient="records"), desc="Matched-baseline rescore"):
        case_id = str(record["case_id"])
        gt = np.load(record["path_ground_truth"]).astype(np.int16)

        baseline_path = matched_baseline_dir / f"{case_id}.npy"
        if baseline_path.exists():
            baseline_pred = np.load(baseline_path).astype(np.int16)
        else:
            loader = BraTSCase(
                case_id=case_id,
                data_root=Path(data_cfg["root"]),
                modalities=data_cfg["modalities"],
                target_spacing=data_cfg["target_spacing"],
                percentile_clip=data_cfg["percentile_clip"],
            )
            image, _ = loader.load()
            baseline_pred = baseline_prediction(
                model=model,
                image=torch.from_numpy(image),
                patch_size=patch_size,
                num_classes=num_classes,
                overlap=overlap,
                device=device,
            )
            np.save(baseline_path, baseline_pred)

        baseline_metrics = compute_prediction_quantities(baseline_pred, gt)
        for metric, value in baseline_metrics.items():
            result_rows.append(
                {
                    "case_id": case_id,
                    "layer": "none",
                    "intervention": "baseline",
                    "metric": metric,
                    "metric_label": METRIC_SPECS[metric],
                    "value": value,
                    "baseline_value": value,
                    "delta": 0.0,
                    "degradation": 0.0,
                    "abs_delta": 0.0,
                }
            )

        for layer_name in layers:
            for intervention in interventions:
                pred_path = pred_dir / layer_name / intervention / f"{case_id}.npy"
                if not pred_path.exists():
                    continue
                ablated_pred = np.load(pred_path).astype(np.int16)
                rho_row = rho_df[
                    (rho_df["case_id"] == case_id)
                    & (rho_df["layer"] == layer_name)
                    & (rho_df["intervention"] == intervention)
                ]
                rho_mean = float(rho_row["rho_mean"].iloc[0]) if not rho_row.empty else float("nan")

                ablated_metrics = compute_prediction_quantities(ablated_pred, gt)
                for metric, ablated_value in ablated_metrics.items():
                    baseline_value = baseline_metrics[metric]
                    delta = float(ablated_value - baseline_value)
                    degradation = _degradation_delta(metric, baseline_value, ablated_value)
                    result_rows.append(
                        {
                            "case_id": case_id,
                            "layer": layer_name,
                            "intervention": intervention,
                            "metric": metric,
                            "metric_label": METRIC_SPECS[metric],
                            "value": ablated_value,
                            "baseline_value": baseline_value,
                            "delta": delta,
                            "degradation": degradation,
                            "abs_delta": abs(delta),
                            "rho_mean": rho_mean,
                        }
                    )

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(matched_dir / "ablation_results.csv", index=False)

    artifacts = _build_ablation_artifacts(
        results_df=results_df,
        rho_df=rho_df,
        output_dir=matched_dir,
        n_cases=failure_df["case_id"].nunique(),
        baseline_description=(
            "matched sliding-window inference (same overlap/patch grid as ablations; "
            "no TTA, no ablation)"
        ),
    )

    old_summary_path = output_dir / "ablation_summary.csv"
    comparison_df = pd.DataFrame()
    comparison_md = ""
    if old_summary_path.exists():
        old_summary = pd.read_csv(old_summary_path)
        comparison_df = compare_dice_rankings(old_summary, artifacts["summary"])
        comparison_df.to_csv(matched_dir / "baseline_comparison.csv", index=False)
        comparison_md = _generate_baseline_comparison_report(
            comparison_df, old_summary, artifacts["summary"]
        )
        (matched_dir / "baseline_comparison.md").write_text(comparison_md)

    return {
        "results": results_df,
        "comparison": comparison_df,
        "comparison_report": comparison_md,
        **artifacts,
    }


def run_layer_ablations(
    config: dict,
    checkpoint_path: Path,
    failure_table_path: Path,
    output_dir: Path,
    device: torch.device,
    overlap: float | None = None,
    max_cases: int | None = None,
    layers: tuple[str, ...] = LAYER_NAMES,
    interventions: tuple[str, ...] = INTERVENTIONS,
) -> dict[str, pd.DataFrame]:
    """
    Mean-ablate each layer and score against existing validation predictions.

    Caches ablated masks under ``predictions/`` so ``--recompute-matched-baseline``
    can rescore without re-running inference. Skips cases already cached with ρ.
    """
    output_dir = ensure_dir(output_dir)
    pred_dir = ensure_dir(output_dir / "predictions")

    failure_df = pd.read_csv(failure_table_path)
    if max_cases is not None and max_cases > 0:
        failure_df = failure_df.head(max_cases)

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    patch_size = config["data"]["patch_size"]
    num_classes = config["model"]["num_classes"]
    data_cfg = config["data"]
    if overlap is None:
        overlap = float(config.get("inference", config.get("uncertainty", {})).get("overlap", 0.25))

    result_rows: list[dict[str, Any]] = []
    rho_df = load_rho_log(output_dir)

    for record in tqdm(failure_df.to_dict(orient="records"), desc="Layer ablations"):
        case_id = str(record["case_id"])
        gt = np.load(record["path_ground_truth"]).astype(np.int16)
        baseline_pred = np.load(record["path_prediction"]).astype(np.int16)
        baseline_metrics = compute_prediction_quantities(baseline_pred, gt)

        for metric, value in baseline_metrics.items():
            result_rows.append(
                {
                    "case_id": case_id,
                    "layer": "none",
                    "intervention": "baseline",
                    "metric": metric,
                    "metric_label": METRIC_SPECS[metric],
                    "value": value,
                    "baseline_value": value,
                    "delta": 0.0,
                    "degradation": 0.0,
                    "abs_delta": 0.0,
                }
            )

        loader = BraTSCase(
            case_id=case_id,
            data_root=Path(data_cfg["root"]),
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
        )
        image, _ = loader.load()
        image_tensor = torch.from_numpy(image)

        for layer_name in layers:
            for intervention in interventions:
                pred_path = pred_dir / layer_name / intervention / f"{case_id}.npy"
                has_pred = pred_path.exists()
                has_rho = _rho_is_logged(rho_df, case_id, layer_name, intervention)

                if has_pred and has_rho:
                    ablated_pred = np.load(pred_path).astype(np.int16)
                    rho_row = rho_df[
                        (rho_df["case_id"] == case_id)
                        & (rho_df["layer"] == layer_name)
                        & (rho_df["intervention"] == intervention)
                    ].iloc[0]
                    rho_mean = float(rho_row["rho_mean"])
                else:
                    seed = _case_seed(case_id, layer_name, intervention)
                    ablated_pred, rho_mean, rho_std = intervention_prediction(
                        model=model,
                        image=image_tensor,
                        patch_size=patch_size,
                        num_classes=num_classes,
                        ablate_layer=layer_name,
                        intervention=intervention,
                        overlap=overlap,
                        device=device,
                        seed=seed,
                    )
                    if not has_pred:
                        ensure_dir(pred_path.parent)
                        np.save(pred_path, ablated_pred)
                    else:
                        ablated_pred = np.load(pred_path).astype(np.int16)
                    if not has_rho:
                        rho_df = _append_rho_log(
                            output_dir,
                            rho_df,
                            case_id,
                            layer_name,
                            intervention,
                            rho_mean,
                            rho_std,
                        )

                ablated_metrics = compute_prediction_quantities(ablated_pred, gt)
                for metric, ablated_value in ablated_metrics.items():
                    baseline_value = baseline_metrics[metric]
                    delta = float(ablated_value - baseline_value)
                    degradation = _degradation_delta(metric, baseline_value, ablated_value)
                    result_rows.append(
                        {
                            "case_id": case_id,
                            "layer": layer_name,
                            "intervention": intervention,
                            "metric": metric,
                            "metric_label": METRIC_SPECS[metric],
                            "value": ablated_value,
                            "baseline_value": baseline_value,
                            "delta": delta,
                            "degradation": degradation,
                            "abs_delta": abs(delta),
                            "rho_mean": rho_mean,
                        }
                    )

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    artifacts = _build_ablation_artifacts(
        results_df=results_df,
        rho_df=rho_df,
        output_dir=output_dir,
        n_cases=failure_df["case_id"].nunique(),
        baseline_description="existing validation predictions (TTA; `failure_metrics.csv`)",
    )

    return {
        "results": results_df,
        "rho_log": rho_df,
        **artifacts,
    }


def _plot_degradation_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    plot_df = summary_df[summary_df["intervention"] == "mean"].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="layer", columns="metric_label", values="mean_degradation")
    pivot = pivot.reindex(list(LAYER_NAMES))
    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = max(float(np.nanmax(np.abs(pivot.values))), 1e-6)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean degradation after mean ablation (higher = worse)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_metric_bars(
    summary_df: pd.DataFrame,
    intervention: str,
    output_path: Path,
    top_metrics: tuple[str, ...] = ("dice", "tumor_volume", "boundary_complexity", "false_negative_voxels"),
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    plot_df = summary_df[
        (summary_df["intervention"] == intervention) & (summary_df["metric"].isin(top_metrics))
    ]
    if plot_df.empty:
        return
    layers = list(LAYER_NAMES)
    metrics = list(top_metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4.5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = plot_df[plot_df["metric"] == metric].set_index("layer").reindex(layers)
        ax.bar(np.arange(len(layers)), sub["mean_degradation"].values, color="steelblue")
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=7)
        ax.set_title(METRIC_SPECS[metric], fontsize=8)
        ax.set_ylabel("Mean degradation")
    fig.suptitle(f"Metric degradation by layer ({intervention} ablation)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _generate_ablation_report(
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    n_cases: int,
    baseline_description: str = "existing validation predictions (no ablation)",
) -> str:
    report = f"""# Layer Ablation Report

**Cases:** {n_cases}  
**Intervention:** mean ablation (replace activations with per-channel spatial mean)  
**Baseline:** {baseline_description}

## Ablation semantics

| Layer | What is replaced |
|-------|------------------|
| encoder1–encoder4 | Skip tensor at decode fusion (encoder still runs) |
| bottleneck | Bottleneck map, then decode |
| decoder4–decoder1 | Decoder block output, continue downstream |

Mean ablation removes case-specific spatial detail while preserving channel scale. Results measure **functional dependence on intact spatial activations**, not necessity of any single probe direction.

Positive `mean_degradation` means the metric got worse after ablation.
"""
    if "matched" in baseline_description.lower():
        report += """
**Primary baseline:** matched sliding-window inference. Scoring against TTA validation predictions is a robustness check (`baseline_comparison.md` when available).
"""
    elif "TTA" in baseline_description or "failure_metrics" in baseline_description:
        report += """
**Note:** Prefer `matched_baseline/` for manuscript tables when available.
"""
    report += """
## Perturbation strength (ρ = ||A - A'|| / ||A||)

| Layer | mean ρ | Dice deg/ρ |
|-------|--------|------------|
"""
    if not strength_df.empty:
        for layer in LAYER_NAMES:
            m = strength_df[
                (strength_df["layer"] == layer) & (strength_df["intervention"] == "mean")
            ]
            m_rho = f"{m['mean_rho'].iloc[0]:.3f}" if not m.empty else "n/a"
            deg_per_rho = (
                f"{m['dice_degradation_per_rho'].iloc[0]:.3f}"
                if not m.empty and "dice_degradation_per_rho" in m.columns
                else "n/a"
            )
            report += f"| {layer} | {m_rho} | {deg_per_rho} |\n"

    report += """
## Top degradations — mean ablation

"""
    mean_rank = ranking_df[ranking_df["intervention"] == "mean"].copy()
    for layer in LAYER_NAMES:
        sub = mean_rank[(mean_rank["layer"] == layer) & (mean_rank["rank"] <= 3)]
        if sub.empty:
            continue
        report += f"\n### {layer}\n\n| Rank | Metric | Mean degradation |\n|------|--------|------------------|\n"
        for _, row in sub.iterrows():
            report += f"| {int(row['rank'])} | {row['metric_label']} | {row['mean_degradation']:.4f} |\n"

    report += """
## Cross-layer comparison (mean ablation, mean degradation)

| Layer | Dice | Volume | Edema frac | Enh frac | Boundary | FP | FN |
|-------|------|--------|------------|----------|----------|----|----|
"""
    mean_summary = summary_df[summary_df["intervention"] == "mean"]
    key_metrics = [
        "dice",
        "tumor_volume",
        "edema_fraction",
        "enhancing_fraction",
        "boundary_complexity",
        "false_positive_voxels",
        "false_negative_voxels",
    ]
    for layer in LAYER_NAMES:
        row_vals = []
        for metric in key_metrics:
            val = mean_summary[
                (mean_summary["layer"] == layer) & (mean_summary["metric"] == metric)
            ]["mean_degradation"]
            row_vals.append(f"{val.iloc[0]:.3f}" if not val.empty else "n/a")
        report += f"| {layer} | " + " | ".join(row_vals) + " |\n"

    report += """
## Artifacts

- `ablation_summary.csv` — aggregated means
- `degradation_ranking.csv` — ranked metric disruption per layer
- `rho_log.csv` — per-case perturbation magnitude ρ
- `intervention_strength_summary.csv` — ρ and degradation-per-ρ by layer
- `figures/heatmap_mean_degradation.png`
"""
    return report


def _generate_baseline_comparison_report(
    comparison_df: pd.DataFrame,
    tta_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
) -> str:
    """Summarize whether layer rankings are stable after switching to matched baseline."""
    lines = [
        "# Ablation baseline comparison",
        "",
        "Compares mean Dice degradation (mean ablation) when ablated outputs are scored against:",
        "1. TTA validation predictions from `failure_metrics.csv`",
        "2. Matched sliding-window inference (same path/overlap as ablations, no TTA)",
        "",
        "## Per-layer Dice degradation (mean ablation)",
        "",
        "| Layer | TTA baseline deg. | Matched baseline deg. | Δ deg. | TTA rank | Matched rank |",
        "|-------|------------------:|----------------------:|-------:|---------:|-------------:|",
    ]
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['layer']} | {row['tta_baseline_dice_degradation']:.3f} | "
            f"{row['matched_baseline_dice_degradation']:.3f} | {row['delta_degradation']:+.3f} | "
            f"{int(row['tta_rank'])} | {int(row['matched_rank'])} |"
        )

    tta_top = comparison_df.sort_values("tta_rank").head(3)["layer"].tolist()
    matched_top = comparison_df.sort_values("matched_rank").head(3)["layer"].tolist()
    rank_corr = float(
        comparison_df[["tta_rank", "matched_rank"]].corr(method="spearman").iloc[0, 1]
    )
    lines.extend(
        [
            "",
            "## Ranking stability",
            "",
            f"- Top-3 layers (TTA baseline): {', '.join(tta_top)}",
            f"- Top-3 layers (matched baseline): {', '.join(matched_top)}",
            f"- Spearman correlation of layer ranks: {rank_corr:.3f}",
        ]
    )

    d1_tta = tta_summary[
        (tta_summary["layer"] == "decoder1")
        & (tta_summary["intervention"] == "mean")
        & (tta_summary["metric"] == "dice")
    ]
    d1_matched = matched_summary[
        (matched_summary["layer"] == "decoder1")
        & (matched_summary["intervention"] == "mean")
        & (matched_summary["metric"] == "dice")
    ]
    if not d1_tta.empty and not d1_matched.empty:
        lines.append(
            f"- Reference baseline Dice (cohort mean): "
            f"TTA = {d1_tta['mean_baseline'].iloc[0]:.3f}, "
            f"matched sliding-window = {d1_matched['mean_baseline'].iloc[0]:.3f}"
        )

    return "\n".join(lines)
