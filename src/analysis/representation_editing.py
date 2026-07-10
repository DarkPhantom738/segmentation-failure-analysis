"""Causal representation editing: evaluate selectivity and produce figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

import torch

from src.analysis.failure_labels import boundary_error_fraction, compute_error_mask
from src.analysis.layer_interventions import compute_prediction_quantities
from src.analysis.semantic_directions import (
    DEFAULT_ALPHAS,
    SemanticDirection,
)
from src.data.brats_dataset import BraTSCase
from src.models.unet3d import build_model
from src.training.editing_inference import editing_prediction
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}

# Observed output properties for selectivity matrix columns.
OBSERVED_METRICS: dict[str, str] = {
    "dice": "Dice",
    "tumor_volume": "Tumor volume",
    "centroid_z": "Centroid Z",
    "edema_fraction": "Edema fraction",
    "enhancing_fraction": "Enhancing fraction",
    "necrosis_fraction": "Necrosis fraction",
    "boundary_complexity": "Boundary complexity",
    "boundary_error_fraction": "Boundary error",
    "false_positive_voxels": "FP voxels",
    "false_negative_voxels": "FN voxels",
}


def _direction_id(direction: SemanticDirection) -> str:
    return f"{direction.layer}::{direction.property_key}"


def _prediction_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    metrics = compute_prediction_quantities(pred, gt)
    error_mask, _, _ = compute_error_mask(gt, pred)
    metrics["boundary_error_fraction"] = float(boundary_error_fraction(error_mask, gt))
    return metrics


def _directional_effect_at_unit_alpha(
    summary_df: pd.DataFrame,
    direction_id: str,
    metric: str,
) -> float:
    """
    Symmetric directional effect at |alpha|=1.

    Uses (mean_delta(+1) - mean_delta(-1)) / 2 so opposing signed effects do not
    cancel when averaging +1 and -1.
    """
    sub = summary_df[
        (summary_df["direction_id"] == direction_id) & (summary_df["metric"] == metric)
    ]
    pos = sub[np.isclose(sub["alpha"], 1.0)]
    neg = sub[np.isclose(sub["alpha"], -1.0)]
    if pos.empty or neg.empty:
        return float("nan")
    return (float(pos["mean_delta"].iloc[0]) - float(neg["mean_delta"].iloc[0])) / 2.0


def _inference_baseline_metrics(
    model: Any,
    image_tensor: torch.Tensor,
    gt: np.ndarray,
    patch_size: list[int],
    num_classes: int,
    edit_layer: str,
    channel_delta: np.ndarray,
    overlap: float,
    device: Any,
) -> dict[str, float]:
    """Baseline prediction via the same editing inference path with alpha=0."""
    baseline_pred = editing_prediction(
        model=model,
        image=image_tensor,
        patch_size=patch_size,
        num_classes=num_classes,
        edit_layer=edit_layer,
        channel_delta=channel_delta,
        alpha=0.0,
        overlap=overlap,
        device=device,
    )
    return _prediction_metrics(baseline_pred, gt)


def _effect_delta(metric: str, baseline: float, edited: float) -> float:
    if metric == "dice":
        return float(edited - baseline)
    if metric in ("false_positive_voxels", "false_negative_voxels"):
        return float(edited - baseline)
    return float(edited - baseline)


def _standardized_effect(delta: float, baseline_std: float) -> float:
    if baseline_std < 1e-8:
        return 0.0
    return float(delta / baseline_std)


def run_representation_edits(
    config: dict,
    checkpoint_path: Path,
    failure_table_path: Path,
    directions_dir: Path,
    output_dir: Path,
    device: Any,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    overlap: float | None = None,
    max_cases: int | None = None,
    max_directions: int | None = None,
    save_example_segmentations: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Apply semantic edits at multiple strengths and evaluate output selectivity.
    """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    examples_dir = ensure_dir(output_dir / "examples")

    catalog = pd.read_csv(Path(directions_dir) / "semantic_directions_catalog.csv")
    if catalog.empty:
        raise ValueError(
            "No semantic directions found. Lower --min-r2 or verify the layer index "
            "and failure table before running representation editing."
        )
    best_idx = catalog.groupby("property_key")["oof_r2"].idxmax()
    directions = [
        SemanticDirection.load(Path(row["direction_path"]))
        for _, row in catalog.loc[best_idx].iterrows()
    ]
    if max_directions is not None:
        directions = directions[:max_directions]

    failure_df = pd.read_csv(failure_table_path)
    if max_cases is not None:
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
        overlap = float(config.get("uncertainty", {}).get("overlap", 0.25))

    result_rows: list[dict[str, Any]] = []

    for record in tqdm(failure_df.to_dict(orient="records"), desc="Editing cases"):
        case_id = str(record["case_id"])

        loader = BraTSCase(
            case_id=case_id,
            data_root=Path(data_cfg["root"]),
            modalities=data_cfg["modalities"],
            target_spacing=data_cfg["target_spacing"],
            percentile_clip=data_cfg["percentile_clip"],
        )
        image, _ = loader.load()
        image_tensor = torch.from_numpy(image)
        gt = np.load(record["path_ground_truth"]).astype(np.int16)

        # Baseline: same checkpoint, sliding-window path, alpha=0 (no perturbation).
        ref_direction = directions[0]
        baseline = _inference_baseline_metrics(
            model=model,
            image_tensor=image_tensor,
            gt=gt,
            patch_size=patch_size,
            num_classes=num_classes,
            edit_layer=ref_direction.layer,
            channel_delta=ref_direction.activation_direction,
            overlap=overlap,
            device=device,
        )
        if save_example_segmentations > 0 and case_id == str(failure_df.iloc[0]["case_id"]):
            np.save(examples_dir / f"{case_id}_baseline_a0.npy", 
                    editing_prediction(
                        model=model, image=image_tensor, patch_size=patch_size,
                        num_classes=num_classes, edit_layer=ref_direction.layer,
                        channel_delta=ref_direction.activation_direction,
                        alpha=0.0, overlap=overlap, device=device,
                    ))

        for direction in directions:
            dir_id = _direction_id(direction)
            channel_delta = direction.activation_direction

            for alpha in alphas:
                edited_pred = editing_prediction(
                    model=model,
                    image=image_tensor,
                    patch_size=patch_size,
                    num_classes=num_classes,
                    edit_layer=direction.layer,
                    channel_delta=channel_delta,
                    alpha=alpha,
                    overlap=overlap,
                    device=device,
                )
                edited_metrics = _prediction_metrics(edited_pred, gt)
                for metric in OBSERVED_METRICS:
                    bval = baseline.get(metric, float("nan"))
                    eval_val = edited_metrics.get(metric, float("nan"))
                    if metric == "boundary_error_fraction":
                        eval_val = edited_metrics["boundary_error_fraction"]
                    delta = _effect_delta(metric, bval, eval_val)
                    result_rows.append(
                        {
                            "case_id": case_id,
                            "direction_id": dir_id,
                            "layer": direction.layer,
                            "edited_property": direction.property_key,
                            "edited_property_label": direction.property_label,
                            "on_target_metric": direction.eval_metric,
                            "alpha": alpha,
                            "metric": metric,
                            "baseline_value": bval,
                            "edited_value": eval_val,
                            "delta": delta,
                            "oof_r2": direction.oof_r2,
                        }
                    )

                if (
                    save_example_segmentations > 0
                    and alpha in (1.0, -1.0)
                    and case_id == failure_df.iloc[0]["case_id"]
                ):
                    np.save(
                        examples_dir / f"{case_id}_{dir_id}_a{alpha:+.1f}.npy",
                        edited_pred,
                    )

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(output_dir / "editing_results.csv", index=False)

    summary_df = _summarize_effects(results_df)
    summary_df.to_csv(output_dir / "editing_summary.csv", index=False)

    selectivity_df = _compute_selectivity_matrix(results_df, summary_df)
    selectivity_df.to_csv(output_dir / "selectivity_matrix.csv", index=False)

    monotonicity_df = _compute_monotonicity(results_df)
    monotonicity_df.to_csv(output_dir / "monotonicity.csv", index=False)

    _plot_dose_response(results_df, figures_dir / "dose_response.png")
    _plot_selectivity_heatmap(selectivity_df, figures_dir / "selectivity_heatmap.png")
    _plot_pipeline_diagram(figures_dir / "pipeline_diagram.png")
    _plot_summary_figure(selectivity_df, monotonicity_df, figures_dir / "summary.png")
    if examples_dir.exists() and any(examples_dir.glob("*.npy")):
        _plot_example_segmentations(
            failure_df.iloc[0],
            examples_dir,
            figures_dir / "example_segmentations.png",
        )

    report = _generate_report(summary_df, selectivity_df, monotonicity_df, len(failure_df))
    (output_dir / "representation_editing_report.md").write_text(report)

    return {
        "results": results_df,
        "summary": summary_df,
        "selectivity": selectivity_df,
        "monotonicity": monotonicity_df,
    }


def _summarize_effects(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(
            ["direction_id", "layer", "edited_property", "on_target_metric", "alpha", "metric"],
            as_index=False,
        )
        .agg(
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            mean_baseline=("baseline_value", "mean"),
            mean_edited=("edited_value", "mean"),
            n_cases=("case_id", "nunique"),
        )
    )


def _compute_selectivity_matrix(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rows: edited direction. Columns: observed metrics.

    Effects at |alpha|=1 use the symmetric directional estimator
    (mean_delta(+1) - mean_delta(-1)) / 2 so opposing edits do not cancel.
    """
    rows: list[dict[str, Any]] = []

    for direction_id in summary_df["direction_id"].unique():
        dgroup = summary_df[summary_df["direction_id"] == direction_id]
        meta = dgroup.iloc[0]
        on_target = str(meta["on_target_metric"])
        baseline_std = (
            results_df[results_df["direction_id"] == direction_id]
            .groupby("metric")["baseline_value"]
            .std()
        )

        entry: dict[str, Any] = {
            "direction_id": direction_id,
            "layer": meta["layer"],
            "edited_property": meta["edited_property"],
            "on_target_metric": on_target,
        }
        on_target_effects: list[float] = []
        off_target_effects: list[float] = []

        for metric in OBSERVED_METRICS:
            directional = _directional_effect_at_unit_alpha(summary_df, direction_id, metric)
            if np.isnan(directional):
                entry[f"effect_{metric}"] = float("nan")
                entry[f"std_effect_{metric}"] = float("nan")
                continue
            std_b = float(baseline_std.get(metric, 1.0))
            std_effect = _standardized_effect(directional, std_b)
            entry[f"effect_{metric}"] = directional
            entry[f"std_effect_{metric}"] = std_effect
            if metric == on_target:
                on_target_effects.append(abs(std_effect))
            else:
                off_target_effects.append(abs(std_effect))

        on_sum = sum(on_target_effects) if on_target_effects else 0.0
        off_sum = sum(off_target_effects) if off_target_effects else 1e-8
        entry["selectivity_score"] = on_sum / (on_sum + off_sum)
        entry["off_target_leakage"] = off_sum
        rows.append(entry)

    return pd.DataFrame(rows)


def _compute_monotonicity(results_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between alpha and metric delta per direction × metric."""
    rows: list[dict[str, Any]] = []
    for (direction_id, metric), group in results_df.groupby(["direction_id", "metric"]):
        agg = group.groupby("alpha")["delta"].mean().reset_index()
        if len(agg) < 3:
            continue
        rho, pval = spearmanr(agg["alpha"], agg["delta"])
        meta = group.iloc[0]
        rows.append(
            {
                "direction_id": direction_id,
                "layer": meta["layer"],
                "edited_property": meta["edited_property"],
                "on_target_metric": meta["on_target_metric"],
                "metric": metric,
                "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
                "p_value": float(pval) if not np.isnan(pval) else float("nan"),
                "is_on_target": metric == meta["on_target_metric"],
            }
        )
    return pd.DataFrame(rows)


def _plot_dose_response(results_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    directions = results_df["direction_id"].unique()
    n = len(directions)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for ax, direction_id in zip(axes.flatten(), directions):
        dsub = results_df[results_df["direction_id"] == direction_id]
        on_target = dsub.iloc[0]["on_target_metric"]
        for metric, color, ls in [
            (on_target, "C0", "-"),
            ("dice", "C1", "--"),
            ("tumor_volume", "C2", ":"),
            ("boundary_complexity", "C3", "-."),
        ]:
            msub = dsub[dsub["metric"] == metric]
            if msub.empty:
                continue
            curve = msub.groupby("alpha")["delta"].mean().reset_index().sort_values("alpha")
            label = OBSERVED_METRICS.get(metric, metric)
            if metric == on_target:
                label = f"{label} (on-target)"
            ax.plot(curve["alpha"], curve["delta"], ls=ls, marker="o", label=label, color=color)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(direction_id.replace("::", "\n"), fontsize=8)
        ax.set_xlabel("α")
        ax.set_ylabel("Δ metric")
        ax.legend(fontsize=6)

    for ax in axes.flatten()[n:]:
        ax.axis("off")
    fig.suptitle("Dose–response: semantic activation edits", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_selectivity_heatmap(selectivity_df: pd.DataFrame, output_path: Path) -> None:
    if selectivity_df.empty:
        return
    plt.rcParams.update(PUBLICATION_RC)
    effect_cols = [c for c in selectivity_df.columns if c.startswith("std_effect_")]
    metrics = [c.replace("std_effect_", "") for c in effect_cols]
    matrix = selectivity_df[effect_cols].values
    row_labels = [
        f"{r.edited_property}\n({r.layer})" for r in selectivity_df.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(row_labels))))
    vmax = np.nanmax(np.abs(matrix)) if matrix.size else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([OBSERVED_METRICS.get(m, m) for m in metrics], rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Selectivity matrix (directional effect at |α|=1)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pipeline_diagram(output_path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        (0.05, 0.4, "Linear probe\non GAP(A)"),
        (0.28, 0.4, "Unit direction β̂"),
        (0.48, 0.4, "Adjoint lift\nΔ = GAP*(σ⊙β̂)"),
        (0.68, 0.4, "A' = A + αΔ\n(skip / bottleneck / decoder)"),
        (0.88, 0.4, "Measure\noutputs"),
    ]
    for i, (x, y, text) in enumerate(boxes):
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#e8f0fe", edgecolor="#3367d6"))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0] - 0.04, y), xytext=(x + 0.08, y),
                        arrowprops=dict(arrowstyle="->", color="#3367d6"))
    ax.set_title("Causal representation editing pipeline", fontsize=11)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_figure(
    selectivity_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if selectivity_df.empty:
        return
    plt.rcParams.update(PUBLICATION_RC)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].barh(
        selectivity_df["direction_id"].str.replace("::", " / "),
        selectivity_df["selectivity_score"],
        color="steelblue",
    )
    axes[0].set_xlabel("Selectivity score")
    axes[0].set_title("On-target vs off-target (|α|=1)")
    axes[0].invert_yaxis()

    if not monotonicity_df.empty:
        on = monotonicity_df[monotonicity_df["is_on_target"]]
        off = monotonicity_df[~monotonicity_df["is_on_target"]]
        axes[1].hist(off["spearman_rho"].dropna(), bins=15, alpha=0.6, label="off-target")
        axes[1].hist(on["spearman_rho"].dropna(), bins=15, alpha=0.8, label="on-target")
        axes[1].axvline(0, color="gray", lw=0.5)
        axes[1].set_xlabel("Spearman(α, Δmetric)")
        axes[1].set_title("Dose monotonicity")
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_example_segmentations(
    first_row: pd.Series,
    examples_dir: Path,
    output_path: Path,
) -> None:
    gt = np.load(first_row["path_ground_truth"])
    case_id = str(first_row["case_id"])
    baseline_path = examples_dir / f"{case_id}_baseline_a0.npy"
    baseline = np.load(baseline_path) if baseline_path.exists() else np.load(first_row["path_prediction"])
    example_files = [
        path for path in sorted(examples_dir.glob("*.npy")) if "baseline_a0" not in path.name
    ][:4]
    if not example_files:
        return

    plt.rcParams.update(PUBLICATION_RC)
    n = len(example_files) + 2
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    mid = gt.shape[0] // 2

    def _show(ax, vol, title):
        wt = (vol > 0).astype(float)
        ax.imshow(wt[mid], cmap="hot", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    _show(axes[0], gt, "GT")
    _show(axes[1], baseline, "Baseline")
    for ax, path in zip(axes[2:], example_files):
        pred = np.load(path)
        _show(ax, pred, path.stem[-20:])

    fig.suptitle(f"Edited segmentations — {first_row['case_id']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _generate_report(
    summary_df: pd.DataFrame,
    selectivity_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    n_cases: int,
) -> str:
    lines = [
        "# Causal Representation Editing Report",
        "",
        f"Validation cases: **{n_cases}**",
        "",
        "## Method",
        "",
        "Semantic directions are learned by fold-safe Ridge probes on pooled layer",
        "readouts, converted back to unscaled readout coordinates, and lifted into",
        "activation tensor space with the minimum-norm spatially constant perturbation",
        "associated with global average pooling. Edits:",
        "`A' = A + α·Δ` on the activation map consumed by downstream computation.",
        "Because Δ is unit-normalized in activation channel space, α is an",
        "activation-space dose, not a literal one-standard-deviation probe step.",
        "",
        "**Encoder layers:** edits target the **skip path** at decode fusion (the same",
        "tensor GAP'd for probing), not activations propagated into deeper encoders.",
        "",
        "**Baseline:** alpha=0 through the same sliding-window editing inference path",
        "(not cached failure-table predictions).",
        "",
        "**Selectivity at |α|=1:** directional effect = (Δ(+1) − Δ(−1)) / 2.",
        "",
        "## Selectivity (|α| = 1)",
        "",
    ]
    if not selectivity_df.empty:
        lines.append("| Direction | Selectivity | Off-target leakage |")
        lines.append("|-----------|-------------|-------------------|")
        for row in selectivity_df.sort_values("selectivity_score", ascending=False).itertuples():
            lines.append(
                f"| {row.edited_property} ({row.layer}) | "
                f"{row.selectivity_score:.3f} | {row.off_target_leakage:.3f} |"
            )
    lines.extend(["", "## Monotonicity (on-target)", ""])
    if not monotonicity_df.empty:
        on = monotonicity_df[monotonicity_df["is_on_target"]]
        for row in on.itertuples():
            lines.append(
                f"- **{row.edited_property}** ({row.layer}): "
                f"ρ={row.spearman_rho:.3f}, p={row.p_value:.4f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "High selectivity means editing one semantic direction primarily moves its",
            "intended output property. Off-target effects quantify disentanglement limits.",
            "Compare with ablation results: probes may be recoverable without causal",
            "necessity (bottleneck location), while edits test fine-grained steerability.",
        ]
    )
    return "\n".join(lines)
