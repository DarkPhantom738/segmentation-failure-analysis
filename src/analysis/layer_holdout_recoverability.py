"""Locked holdout validation for anatomical recoverability (R²) by layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import _impute_nan, create_bad_case_label
from src.analysis.layer_analysis import (
    ANATOMY_TARGET_SPECS,
    DEPTH_LABELS,
    LAYER_ORDER,
    build_anatomy_table,
    load_layer_index,
    load_layer_matrix,
)
from src.analysis.layer_holdout import _prepare_merged
from src.models.unet3d import LAYER_NAMES
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}


def _oof_ridge_metrics(
    x: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int = 42,
) -> dict[str, float]:
    """Fold-safe OOF Ridge regression metrics on a subset."""
    y = np.asarray(y, dtype=np.float64)
    x = _impute_nan(x)
    if len(y) < 4 or np.std(y) < 1e-12:
        return {"r2": float("nan"), "mae": float("nan"), "spearman": float("nan")}

    cv_folds = min(cv_folds, len(y))
    cv_folds = max(2, cv_folds)
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    preds = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in splitter.split(x):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 20))),
            ]
        )
        pipe.fit(x[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(x[test_idx])

    rho, _ = spearmanr(y, preds)
    return {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "spearman": float(rho) if not np.isnan(rho) else float("nan"),
    }


def _fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 20))),
        ]
    )
    pipe.fit(_impute_nan(x_train), y_train)
    return pipe.predict(_impute_nan(x_test))


def _locked_ridge_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if np.std(y_train) < 1e-12 or np.std(y_test) < 1e-12:
        return {"r2": float("nan"), "mae": float("nan"), "spearman": float("nan")}
    preds = _fit_ridge_predict(x_train, y_train, x_test, random_state=random_state)
    rho, _ = spearmanr(y_test, preds)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "spearman": float(rho) if not np.isnan(rho) else float("nan"),
    }


def _plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    vmin: float = -0.2,
    vmax: float = 1.0,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    pivot = df.pivot(index="layer", columns="target_label", values=value_col)
    pivot = pivot.reindex(LAYER_ORDER)
    col_order = [ANATOMY_TARGET_SPECS[k] for k in ANATOMY_TARGET_SPECS if k in df["target"].values]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([DEPTH_LABELS.get(l, l) for l in pivot.index])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, label=value_col)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_depth_lines(
    df: pd.DataFrame,
    targets: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LAYER_ORDER))
    for target_key, label in targets.items():
        sub = df[df["target"] == target_key].set_index("layer").reindex(LAYER_ORDER)
        ax.plot(x, sub["r2"].values, marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([DEPTH_LABELS[l] for l in LAYER_ORDER], rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_layer_holdout_recoverability(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
    output_dir: str | Path,
    dice_threshold: float = 0.80,
    selection_fraction: float = 0.5,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    50/50 holdout for anatomical R² probes.

    Selection set: OOF Ridge R² for every layer × anatomical target.
    Locked test: fit Ridge on selection set, evaluate R² on locked half.
  """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    merged = _prepare_merged(layer_index_path, failure_table_path)
    failure_df = pd.read_csv(failure_table_path)
    anatomy_df = build_anatomy_table(merged, failure_df)

    dice = merged["dice"].values.astype(np.float64)
    bad_labels, effective_threshold = create_bad_case_label(
        dice, mode="threshold", threshold=dice_threshold
    )

    selection_idx, locked_idx = train_test_split(
        np.arange(len(merged)),
        test_size=1.0 - selection_fraction,
        random_state=random_state,
        stratify=bad_labels,
    )

    split_df = pd.DataFrame(
        {
            "case_id": merged["case_id"].values,
            "dice": dice,
            "label_bad": bad_labels,
            "split": "selection",
        }
    )
    split_df.loc[locked_idx, "split"] = "locked_test"
    split_df.to_csv(output_dir / "case_split.csv", index=False)

    layer_features = {name: load_layer_matrix(merged, name) for name in LAYER_NAMES}
    sel_cv_folds = min(cv_folds, len(selection_idx))

    selection_rows: list[dict[str, Any]] = []
    locked_rows: list[dict[str, Any]] = []

    for layer_name in LAYER_NAMES:
        x_all = layer_features[layer_name]
        x_sel = x_all[selection_idx]
        x_lock = x_all[locked_idx]

        for target_key, target_label in ANATOMY_TARGET_SPECS.items():
            y_all = anatomy_df[target_key].values.astype(np.float64)
            y_sel = y_all[selection_idx]
            y_lock = y_all[locked_idx]

            sel_metrics = _oof_ridge_metrics(
                x_sel, y_sel, cv_folds=sel_cv_folds, random_state=random_state
            )
            lock_metrics = _locked_ridge_metrics(
                x_sel, y_sel, x_lock, y_lock, random_state=random_state
            )

            selection_rows.append(
                {
                    "layer": layer_name,
                    "target": target_key,
                    "target_label": target_label,
                    "split": "selection_oof",
                    "n_cases": len(selection_idx),
                    "cv_folds": sel_cv_folds,
                    **sel_metrics,
                }
            )
            locked_rows.append(
                {
                    "layer": layer_name,
                    "target": target_key,
                    "target_label": target_label,
                    "split": "locked_test",
                    "n_cases": len(locked_idx),
                    **lock_metrics,
                }
            )

    selection_df = pd.DataFrame(selection_rows)
    locked_df = pd.DataFrame(locked_rows)
    selection_df.to_csv(output_dir / "selection_recoverability.csv", index=False)
    locked_df.to_csv(output_dir / "locked_test_recoverability.csv", index=False)

    # Best layer per target on selection set; record locked R² for winner vs bottleneck
    best_rows: list[dict[str, Any]] = []
    for target_key, target_label in ANATOMY_TARGET_SPECS.items():
        sel_sub = selection_df[selection_df["target"] == target_key]
        best_row = sel_sub.loc[sel_sub["r2"].idxmax()]
        best_layer = str(best_row["layer"])
        lock_best = locked_df[
            (locked_df["target"] == target_key) & (locked_df["layer"] == best_layer)
        ].iloc[0]
        lock_bn = locked_df[
            (locked_df["target"] == target_key) & (locked_df["layer"] == "bottleneck")
        ].iloc[0]
        best_rows.append(
            {
                "target": target_key,
                "target_label": target_label,
                "best_layer_selection": best_layer,
                "selection_r2_best": float(best_row["r2"]),
                "locked_r2_best": float(lock_best["r2"]),
                "locked_r2_bottleneck": float(lock_bn["r2"]),
                "locked_r2_delta_best_minus_bottleneck": float(lock_best["r2"] - lock_bn["r2"]),
            }
        )
    best_per_target_df = pd.DataFrame(best_rows)
    best_per_target_df.to_csv(output_dir / "best_layer_per_target.csv", index=False)

    combined_df = pd.concat(
        [
            selection_df.assign(split_phase="selection_oof"),
            locked_df.assign(split_phase="locked_test"),
        ],
        ignore_index=True,
    )
    combined_df.to_csv(output_dir / "recoverability_holdout_all.csv", index=False)

    depth_targets = {
        "centroid_z": "Centroid Z",
        "log_wt_volume": "log(WT volume)",
        "gt_compactness": "Boundary complexity",
        "dice": "Dice",
        "gt_edema_frac": "Edema fraction",
    }
    _plot_heatmap(
        selection_df,
        "r2",
        f"Selection-set OOF R² (seed={random_state})",
        figures_dir / "heatmap_selection_r2.png",
    )
    _plot_heatmap(
        locked_df,
        "r2",
        f"Locked-test R² (seed={random_state})",
        figures_dir / "heatmap_locked_r2.png",
    )
    _plot_depth_lines(
        selection_df,
        depth_targets,
        "Selection-set OOF R² across depth",
        figures_dir / "line_selection_r2_depth.png",
    )
    _plot_depth_lines(
        locked_df,
        depth_targets,
        "Locked-test R² across depth",
        figures_dir / "line_locked_r2_depth.png",
    )

    report = _generate_report(
        n_total=len(merged),
        n_selection=len(selection_idx),
        n_locked=len(locked_idx),
        random_state=random_state,
        dice_threshold=dice_threshold,
        effective_threshold=effective_threshold,
        selection_df=selection_df,
        locked_df=locked_df,
        best_per_target_df=best_per_target_df,
    )
    (output_dir / "layer_holdout_recoverability_report.md").write_text(report)

    return {
        "selection": selection_df,
        "locked": locked_df,
        "best_per_target": best_per_target_df,
        "split": split_df,
        "report": report,
    }


def _generate_report(
    n_total: int,
    n_selection: int,
    n_locked: int,
    random_state: int,
    dice_threshold: float,
    effective_threshold: float,
    selection_df: pd.DataFrame,
    locked_df: pd.DataFrame,
    best_per_target_df: pd.DataFrame,
) -> str:
    key_targets = [
        "centroid_z",
        "log_wt_volume",
        "gt_compactness",
        "gt_edema_frac",
        "dice",
    ]

    report = f"""# Layer Holdout Recoverability Report

**Cases:** {n_total} ({n_selection} selection / {n_locked} locked)  
**Split seed:** {random_state} (stratified by Dice < {dice_threshold})  
**Probe:** Ridge regression, scaler fit per fold (selection) or on selection only (locked)

## Key targets — selection OOF R² vs locked test R²

| Target | Best layer (selection) | Sel R² | Locked R² (best) | Locked R² (bottleneck) | Δ locked |
|--------|------------------------|--------|--------------------|-------------------------|----------|
"""
    for target in key_targets:
        row = best_per_target_df[best_per_target_df["target"] == target].iloc[0]
        report += (
            f"| {row['target_label']} | {row['best_layer_selection']} | "
            f"{row['selection_r2_best']:.3f} | {row['locked_r2_best']:.3f} | "
            f"{row['locked_r2_bottleneck']:.3f} | {row['locked_r2_delta_best_minus_bottleneck']:+.3f} |\n"
        )

    report += """
## All layers × targets (locked test R²)

See `locked_test_recoverability.csv` and `figures/heatmap_locked_r2.png`.

## Notes

- Selection-set R² is OOF within the selection half (used to rank layers).
- Locked-test R² is a single evaluation: train on full selection half, test on locked half.
- Layer choice per target uses selection OOF only; locked R² for the chosen layer is not peeked during selection.
"""
    return report
