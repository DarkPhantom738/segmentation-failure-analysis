"""Minimal screening: decoder2 tumor-volume probe vs matched random perturbations."""

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

from src.analysis.probe_screen_common import (
    N_PER_BIN,
    PUBLICATION_RC,
    analytical_probe_before_after,
    load_existing_probe_rows,
    mean_gap,
    mean_perturbation_ratio,
    screening_rng,
    select_stratified_cases,
    unit_random_direction,
)
from src.analysis.layer_interventions import compute_prediction_quantities
from src.analysis.semantic_directions import SemanticDirection
from src.data.brats_dataset import BraTSCase
from src.models.unet3d import build_model
from src.training.decoder_cache_inference import (
    capture_decoder2_patches,
    downstream_prediction_from_decoder2_cache,
)
from src.utils.io import ensure_dir

PROBE_DIRECTION_ID = "decoder2::gt_wt_voxels"
PROBE_ALPHA = 1.0
N_RANDOM = 5
EDITING_RESULTS = Path("outputs_10hour/representation_editing/editing_results.csv")
DIRECTION_PATH = Path(
    "outputs_10hour/semantic_directions/directions/decoder2__gt_wt_voxels.npz"
)


def _opposite_sign_rate(probe_df: pd.DataFrame, value_col: str) -> float:
    """Fraction of cases where +α and -α deltas have opposite signs."""
    ok = 0
    total = 0
    for case_id in probe_df["case_id"].unique():
        pos = probe_df[(probe_df["case_id"] == case_id) & (probe_df["alpha"] > 0)][value_col]
        neg = probe_df[(probe_df["case_id"] == case_id) & (probe_df["alpha"] < 0)][value_col]
        if pos.empty or neg.empty:
            continue
        total += 1
        if float(pos.iloc[0]) * float(neg.iloc[0]) < 0:
            ok += 1
    return ok / max(total, 1)


def _probe_percentile_among_random(results_df: pd.DataFrame) -> float:
    percentiles: list[float] = []
    for case_id in results_df["case_id"].unique():
        for alpha in (-PROBE_ALPHA, PROBE_ALPHA):
            probe_row = results_df[
                (results_df["case_id"] == case_id)
                & (results_df["direction_type"] == "probe")
                & (np.isclose(results_df["alpha"], alpha))
            ]
            random_rows = results_df[
                (results_df["case_id"] == case_id)
                & (results_df["direction_type"] == "random")
                & (np.isclose(results_df["alpha"], alpha))
            ]
            if probe_row.empty or random_rows.empty:
                continue
            probe_abs = abs(float(probe_row["tumor_volume_delta"].iloc[0]))
            random_abs = random_rows["tumor_volume_delta"].abs().values
            percentiles.append(float(np.mean(random_abs <= probe_abs)))
    return float(np.mean(percentiles)) if percentiles else float("nan")


def run_volume_probe_screen(
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
    existing_probe = load_existing_probe_rows(
        editing_df, case_ids, PROBE_DIRECTION_ID, "tumor_volume", PROBE_ALPHA
    )

    probe_direction = SemanticDirection.load(direction_path)
    probe_delta_vec = probe_direction.activation_direction.astype(np.float64)

    rng = screening_rng(random_seed + 1)
    random_directions = [
        unit_random_direction(probe_direction.activation_channels, rng)
        for _ in range(n_random)
    ]

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

    rows: list[dict[str, Any]] = []

    for record in tqdm(selected.to_dict(orient="records"), desc="Volume probe screen"):
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

        origins, d2_patches, s1_patches, vol_shape = capture_decoder2_patches(
            model, image_tensor, patch_size, overlap, device
        )
        gap_before = mean_gap(d2_patches)
        perturbation_ratio = mean_perturbation_ratio(
            model, d2_patches, probe_delta_vec, PROBE_ALPHA
        )

        baseline_pred = downstream_prediction_from_decoder2_cache(
            model, origins, d2_patches, s1_patches, vol_shape, patch_size, num_classes,
            probe_delta_vec, 0.0, device,
        )
        baseline_metrics = compute_prediction_quantities(baseline_pred, gt)
        baseline_volume = float(baseline_metrics["tumor_volume"])

        for alpha in (-PROBE_ALPHA, PROBE_ALPHA):
            sign = int(np.sign(alpha))
            ex = existing_probe[
                (existing_probe["case_id"] == case_id)
                & (np.isclose(existing_probe["alpha"], alpha))
            ]
            if not ex.empty:
                vol_before = float(ex["baseline_value"].iloc[0])
                vol_after = float(ex["edited_value"].iloc[0])
                vol_delta = float(ex["delta"].iloc[0])
                dice_before = float(ex["dice_before"].iloc[0])
                dice_after = float(ex["dice_after"].iloc[0])
                dice_delta = float(ex["dice_delta"].iloc[0])
            else:
                vol_before = baseline_volume
                pred = downstream_prediction_from_decoder2_cache(
                    model, origins, d2_patches, s1_patches, vol_shape, patch_size,
                    num_classes, probe_delta_vec, alpha, device,
                )
                m = compute_prediction_quantities(pred, gt)
                vol_after = float(m["tumor_volume"])
                vol_delta = vol_after - vol_before
                dice_before = baseline_dice
                dice_after = float(m["dice"])
                dice_delta = dice_after - dice_before

            probe_pred = downstream_prediction_from_decoder2_cache(
                model, origins, d2_patches, s1_patches, vol_shape, patch_size, num_classes,
                probe_delta_vec, alpha, device,
            )
            changed_voxels = int(np.sum(probe_pred != baseline_pred))
            pred_b, pred_a, pred_d = analytical_probe_before_after(
                gap_before, probe_direction, alpha, probe_delta_vec
            )

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
                    "tumor_volume_before": vol_before,
                    "tumor_volume_after": vol_after,
                    "tumor_volume_delta": vol_delta,
                    "tumor_volume_delta_abs": abs(vol_delta),
                    "analytical_probe_pred_before": pred_b,
                    "analytical_probe_pred_after": pred_a,
                    "analytical_probe_pred_delta": pred_d,
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
                pred = downstream_prediction_from_decoder2_cache(
                    model, origins, d2_patches, s1_patches, vol_shape, patch_size,
                    num_classes, rnd_dir, alpha, device,
                )
                m = compute_prediction_quantities(pred, gt)
                vol_after = float(m["tumor_volume"])
                vol_delta = vol_after - baseline_volume
                pred_b, pred_a, pred_d = analytical_probe_before_after(
                    gap_before, probe_direction, alpha, rnd_dir
                )
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
                        "tumor_volume_before": baseline_volume,
                        "tumor_volume_after": vol_after,
                        "tumor_volume_delta": vol_delta,
                        "tumor_volume_delta_abs": abs(vol_delta),
                        "analytical_probe_pred_before": pred_b,
                        "analytical_probe_pred_after": pred_a,
                        "analytical_probe_pred_delta": pred_d,
                        "dice_before": baseline_dice,
                        "dice_after": float(m["dice"]),
                        "dice_delta": float(m["dice"]) - baseline_dice,
                        "changed_voxels": int(np.sum(pred != baseline_pred)),
                        "source": "downstream_from_cached_decoder2",
                    }
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "volume_probe_screen_cases.csv", index=False)

    probe_df = results_df[results_df["direction_type"] == "probe"]
    random_df = results_df[results_df["direction_type"] == "random"]
    probe_abs = float(probe_df["tumor_volume_delta_abs"].mean())
    random_abs = float(random_df["tumor_volume_delta_abs"].mean())
    ratio = probe_abs / max(random_abs, 1e-12)
    percentile = _probe_percentile_among_random(results_df)
    vol_opposite = _opposite_sign_rate(probe_df, "tumor_volume_delta")
    anal_opposite = _opposite_sign_rate(probe_df, "analytical_probe_pred_delta")

    summary_rows: list[dict[str, Any]] = [
        {
            "scope": "all",
            "dice_bin": "all",
            "n_cases": len(case_ids),
            "n_random_directions": n_random,
            "mean_abs_volume_delta_probe": probe_abs,
            "mean_abs_volume_delta_random": random_abs,
            "probe_to_random_abs_volume_ratio": ratio,
            "probe_volume_percentile_vs_random": percentile,
            "fraction_opposite_sign_actual_volume": vol_opposite,
            "fraction_opposite_sign_analytical_probe": anal_opposite,
        }
    ]
    for dice_bin in ("low", "medium", "high"):
        psub = probe_df[probe_df["dice_bin"] == dice_bin]
        rsub = random_df[random_df["dice_bin"] == dice_bin]
        r_all = results_df[results_df["dice_bin"] == dice_bin]
        p_abs = float(psub["tumor_volume_delta_abs"].mean()) if not psub.empty else float("nan")
        r_abs = float(rsub["tumor_volume_delta_abs"].mean()) if not rsub.empty else float("nan")
        summary_rows.append(
            {
                "scope": "dice_bin",
                "dice_bin": dice_bin,
                "n_cases": int(psub["case_id"].nunique()),
                "n_random_directions": n_random,
                "mean_abs_volume_delta_probe": p_abs,
                "mean_abs_volume_delta_random": r_abs,
                "probe_to_random_abs_volume_ratio": p_abs / max(r_abs, 1e-12),
                "probe_volume_percentile_vs_random": _probe_percentile_among_random(r_all),
                "fraction_opposite_sign_actual_volume": _opposite_sign_rate(psub, "tumor_volume_delta"),
                "fraction_opposite_sign_analytical_probe": _opposite_sign_rate(
                    psub, "analytical_probe_pred_delta"
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "volume_probe_screen_summary.csv", index=False)

    _plot_volume_probe_vs_random(results_df, figures_dir / "abs_volume_delta_probe_vs_random.png")
    _plot_analytical_vs_actual(probe_df, figures_dir / "analytical_vs_actual_volume.png")

    conclusion = _build_conclusion(
        summary_df, probe_abs, random_abs, ratio, percentile, vol_opposite, anal_opposite
    )
    (output_dir / "volume_probe_screen_conclusion.md").write_text(conclusion)

    return {"results": results_df, "summary": summary_df, "conclusion": conclusion}


def _plot_volume_probe_vs_random(results_df: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    probe = results_df[results_df["direction_type"] == "probe"]["tumor_volume_delta_abs"]
    random = results_df[results_df["direction_type"] == "random"]["tumor_volume_delta_abs"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([probe.values, random.values], tick_labels=["probe", "random"])
    ax.set_ylabel("|Δ tumor volume| (voxels)")
    ax.set_title("decoder2 volume: probe vs random (|α|=1)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_analytical_vs_actual(probe_df: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(
        probe_df["analytical_probe_pred_delta"],
        probe_df["tumor_volume_delta"],
        alpha=0.7,
        c=probe_df["alpha"],
        cmap="coolwarm",
    )
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Analytical Δ probe prediction (volume)")
    ax.set_ylabel("Actual Δ tumor volume (voxels)")
    ax.set_title("Probe direction: representation vs segmentation")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_conclusion(
    summary_df: pd.DataFrame,
    probe_abs: float,
    random_abs: float,
    ratio: float,
    percentile: float,
    vol_opposite: float,
    anal_opposite: float,
) -> str:
    all_row = summary_df[summary_df["scope"] == "all"].iloc[0]
    clearly = ratio > 1.25 and probe_abs > random_abs
    verdict = (
        "clearly outperforms"
        if clearly
        else "does not clearly outperform"
    )

    lines = [
        "# Decoder2 Volume Probe Screening Conclusion",
        "",
        f"The probe direction **{verdict}** matched random perturbations on mean |Δ tumor volume|.",
        "",
        "## Overall (30 cases, |α|=1)",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| mean \\|Δ volume\\| probe | {probe_abs:.2f} vox |",
        f"| mean \\|Δ volume\\| random | {random_abs:.2f} vox |",
        f"| probe / random ratio | {ratio:.2f} |",
        f"| probe percentile vs random | {percentile:.2f} |",
        f"| +α/−α opposite (actual volume) | {vol_opposite:.0%} of cases |",
        f"| +α/−α opposite (analytical probe) | {anal_opposite:.0%} of cases |",
        "",
        "## By Dice bin",
        "",
        "| bin | probe | random | ratio | percentile | opp. vol | opp. anal |",
        "|-----|------:|-------:|------:|-----------:|---------:|----------:|",
    ]
    for _, row in summary_df[summary_df["scope"] == "dice_bin"].iterrows():
        lines.append(
            f"| {row['dice_bin']} | {row['mean_abs_volume_delta_probe']:.1f} | "
            f"{row['mean_abs_volume_delta_random']:.1f} | {row['probe_to_random_abs_volume_ratio']:.2f} | "
            f"{row['probe_volume_percentile_vs_random']:.2f} | "
            f"{row['fraction_opposite_sign_actual_volume']:.0%} | "
            f"{row['fraction_opposite_sign_analytical_probe']:.0%} |"
        )

    lines.extend(
        [
            "",
            "**Interpretation:** High analytical opposite-sign rate with near-zero actual volume "
            "change indicates the probe axis moves in representation space but decoder2 edits "
            "do not propagate to segmentation volume.",
        ]
    )
    return "\n".join(lines)
