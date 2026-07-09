"""Locked holdout validation for layer selection and bad-case detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import (
    _impute_nan,
    choose_cv_folds,
    compute_deployable_uncertainty_features,
    create_bad_case_label,
)
from src.analysis.layer_analysis import load_layer_index, load_layer_matrix
from src.analysis.layer_significance import (
    bootstrap_auroc_ci,
    bootstrap_auroc_difference_ci,
    delong_test,
)
from src.models.unet3d import LAYER_NAMES
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}


def _prepare_merged(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
) -> pd.DataFrame:
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


def _oof_probabilities(
    x: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    x = _impute_nan(x)
    cv_folds = choose_cv_folds(labels)
    proba = np.zeros(len(labels), dtype=np.float64)
    if cv_folds < 2 or len(np.unique(labels)) < 2:
        return proba

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(x, labels):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=random_state)),
            ]
        )
        pipe.fit(x[train_idx], labels[train_idx])
        proba[test_idx] = pipe.predict_proba(x[test_idx])[:, 1]
    return proba


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]
    )
    pipe.fit(_impute_nan(x_train), y_train)
    return pipe.predict_proba(_impute_nan(x_test))[:, 1]


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _metrics_at_threshold(labels: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "auroc": _safe_auroc(labels, proba),
        "auprc": float(average_precision_score(labels, proba)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
    }


def run_layer_holdout(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
    output_dir: str | Path,
    dice_threshold: float = 0.80,
    selection_fraction: float = 0.5,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Split validation cases into layer-selection and locked-test halves.

    Phase 1 (selection set only): OOF AUROC for every layer; pick best layer.
    Phase 2 (locked test set): train on selection set, evaluate uncertainty,
    chosen layer, and combined — no further tuning after layer choice.
    """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    merged = _prepare_merged(layer_index_path, failure_table_path)
    dice = merged["dice"].values.astype(np.float64)
    labels, effective_threshold = create_bad_case_label(
        dice, mode="threshold", threshold=dice_threshold
    )

    selection_idx, locked_idx = train_test_split(
        np.arange(len(merged)),
        test_size=1.0 - selection_fraction,
        random_state=random_state,
        stratify=labels,
    )

    split_df = pd.DataFrame(
        {
            "case_id": merged["case_id"].values,
            "dice": dice,
            "label_bad": labels,
            "split": "selection",
        }
    )
    split_df.loc[locked_idx, "split"] = "locked_test"
    split_df.to_csv(output_dir / "case_split.csv", index=False)

    uncertainty, _ = compute_deployable_uncertainty_features(merged)
    layer_features = {name: load_layer_matrix(merged, name) for name in LAYER_NAMES}

    # Phase 1: layer selection on selection set only (OOF within selection set)
    selection_rows: list[dict[str, Any]] = []
    sel_labels = labels[selection_idx]
    for layer_name in LAYER_NAMES:
        x_sel = layer_features[layer_name][selection_idx]
        proba_oof = _oof_probabilities(x_sel, sel_labels, random_state=random_state)
        metrics = _metrics_at_threshold(sel_labels, proba_oof)
        selection_rows.append(
            {
                "layer": layer_name,
                "split": "selection",
                "n_cases": len(selection_idx),
                "n_bad_cases": int(sel_labels.sum()),
                "dice_threshold": dice_threshold,
                **metrics,
            }
        )
    selection_df = pd.DataFrame(selection_rows).sort_values("auroc", ascending=False)
    selection_df.to_csv(output_dir / "layer_selection_results.csv", index=False)

    chosen_layer = str(selection_df.iloc[0]["layer"])
    chosen_auroc_selection = float(selection_df.iloc[0]["auroc"])

    # Phase 2: locked test — fit on full selection set, evaluate on locked test
    sel_unc = uncertainty[selection_idx]
    sel_layer = layer_features[chosen_layer][selection_idx]
    sel_combined = np.concatenate([sel_unc, sel_layer], axis=1).astype(np.float32)

    lock_unc = uncertainty[locked_idx]
    lock_layer = layer_features[chosen_layer][locked_idx]
    lock_combined = np.concatenate([lock_unc, lock_layer], axis=1).astype(np.float32)
    lock_labels = labels[locked_idx]

    proba_map = {
        "uncertainty": _fit_predict(sel_unc, sel_labels, lock_unc, random_state=random_state),
        "chosen_layer": _fit_predict(sel_layer, sel_labels, lock_layer, random_state=random_state),
        "combined": _fit_predict(sel_combined, sel_labels, lock_combined, random_state=random_state),
    }

    locked_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for model_name, proba in proba_map.items():
        metrics = _metrics_at_threshold(lock_labels, proba)
        ci = bootstrap_auroc_ci(
            lock_labels, proba, n_bootstrap=n_bootstrap, random_state=random_state
        )
        locked_rows.append(
            {
                "model": model_name,
                "chosen_layer": chosen_layer,
                "split": "locked_test",
                "n_cases": len(locked_idx),
                "n_bad_cases": int(lock_labels.sum()),
                "dice_threshold": dice_threshold,
                "auroc_ci_lower": ci["ci_lower"],
                "auroc_ci_upper": ci["ci_upper"],
                **metrics,
            }
        )
        boot_rows.append(
            {
                "model": model_name,
                "auroc": ci["auroc"],
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
            }
        )

    diff = bootstrap_auroc_difference_ci(
        lock_labels,
        proba_map["uncertainty"],
        proba_map["combined"],
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    delong = delong_test(
        lock_labels, proba_map["uncertainty"], proba_map["combined"]
    )

    locked_df = pd.DataFrame(locked_rows)
    locked_df.to_csv(output_dir / "locked_test_results.csv", index=False)

    comparison_df = pd.DataFrame(
        [
            {
                "comparison": "combined_minus_uncertainty",
                "auroc_diff": diff["auroc_diff"],
                "ci_lower": diff["ci_lower"],
                "ci_upper": diff["ci_upper"],
                "delong_z": delong["z_stat"],
                "delong_p_value": delong["p_value"],
            }
        ]
    )
    comparison_df.to_csv(output_dir / "locked_test_comparison.csv", index=False)

    # Per-case locked predictions for audit
    pred_df = pd.DataFrame(
        {
            "case_id": merged.iloc[locked_idx]["case_id"].values,
            "dice": dice[locked_idx],
            "label_bad": lock_labels,
            "P_uncertainty": proba_map["uncertainty"],
            f"P_{chosen_layer}": proba_map["chosen_layer"],
            "P_combined": proba_map["combined"],
        }
    )
    pred_df.to_csv(output_dir / "locked_test_predictions.csv", index=False)

    _plot_locked_roc(lock_labels, proba_map, boot_rows, chosen_layer, figures_dir / "locked_test_roc.png")
    _plot_selection_bar(selection_df, figures_dir / "layer_selection_auroc.png")

    report = _generate_holdout_report(
        n_total=len(merged),
        n_selection=len(selection_idx),
        n_locked=len(locked_idx),
        dice_threshold=dice_threshold,
        effective_threshold=effective_threshold,
        chosen_layer=chosen_layer,
        chosen_auroc_selection=chosen_auroc_selection,
        selection_df=selection_df,
        locked_df=locked_df,
        comparison_df=comparison_df,
        random_state=random_state,
    )
    (output_dir / "layer_holdout_report.md").write_text(report)

    return {
        "chosen_layer": chosen_layer,
        "selection_results": selection_df,
        "locked_results": locked_df,
        "comparison": comparison_df,
        "split": split_df,
        "predictions": pred_df,
        "report": report,
    }


def _plot_locked_roc(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    boot_rows: list[dict[str, Any]],
    chosen_layer: str,
    output_path: Path,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "uncertainty": "crimson",
        "chosen_layer": "steelblue",
        "combined": "seagreen",
    }
    labels_display = {
        "uncertainty": "uncertainty",
        "chosen_layer": chosen_layer,
        "combined": f"uncertainty + {chosen_layer}",
    }
    boot_lookup = {row["model"]: row for row in boot_rows}
    for model, proba in proba_map.items():
        fpr, tpr, _ = roc_curve(labels, proba)
        row = boot_lookup[model]
        ax.plot(
            fpr,
            tpr,
            color=colors[model],
            lw=2,
            label=(
                f"{labels_display[model]} "
                f"AUROC={row['auroc']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            ),
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Locked test set ROC (trained on selection set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_selection_bar(selection_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_df = selection_df.sort_values("auroc", ascending=True)
    colors = ["seagreen" if i == len(plot_df) - 1 else "steelblue" for i in range(len(plot_df))]
    ax.barh(plot_df["layer"], plot_df["auroc"], color=colors)
    ax.set_xlabel("OOF AUROC (selection set)")
    ax.set_title("Layer selection on 50% holdout")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _generate_holdout_report(
    n_total: int,
    n_selection: int,
    n_locked: int,
    dice_threshold: float,
    effective_threshold: float,
    chosen_layer: str,
    chosen_auroc_selection: float,
    selection_df: pd.DataFrame,
    locked_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    random_state: int,
) -> str:
    unc = locked_df[locked_df["model"] == "uncertainty"].iloc[0]
    layer = locked_df[locked_df["model"] == "chosen_layer"].iloc[0]
    comb = locked_df[locked_df["model"] == "combined"].iloc[0]
    comp = comparison_df.iloc[0]

    sig = comp["ci_lower"] > 0 and comp["delong_p_value"] < 0.05
    beats_unc = comb["auroc"] > unc["auroc"]

    top3 = selection_df.head(3)[["layer", "auroc"]]

    report = f"""# Layer Holdout Validation Report

**Total validation cases:** {n_total}  
**Layer-selection set:** {n_selection} cases (50%)  
**Locked test set:** {n_locked} cases (50%)  
**Split seed:** {random_state} (stratified by Dice < {dice_threshold})  
**Bad-case threshold:** Dice < {effective_threshold:.4f}

## Protocol

1. On the **selection set only**, every layer was ranked by fold-safe OOF logistic AUROC.
2. **Chosen layer:** `{chosen_layer}` (selection AUROC = {chosen_auroc_selection:.3f}).
3. On the **locked test set**, models were fit on the full selection set and evaluated once:
   - uncertainty only
   - `{chosen_layer}` only
   - uncertainty + `{chosen_layer}`
4. No hyperparameters or layer choice were changed after viewing locked-test results.

## Layer selection (selection set OOF AUROC)

| Rank | Layer | AUROC |
|------|-------|-------|
"""
    for i, (_, row) in enumerate(selection_df.iterrows(), start=1):
        report += f"| {i} | {row['layer']} | {row['auroc']:.3f} |\n"

    report += f"""
Top 3 on selection set: {", ".join(f"{r['layer']} ({r['auroc']:.3f})" for _, r in top3.iterrows())}.

## Locked test results

| Model | AUROC | 95% bootstrap CI | AUPRC | F1 |
|-------|-------|----------------|-------|-----|
| Uncertainty | {unc['auroc']:.3f} | [{unc['auroc_ci_lower']:.3f}, {unc['auroc_ci_upper']:.3f}] | {unc['auprc']:.3f} | {unc['f1']:.3f} |
| {chosen_layer} | {layer['auroc']:.3f} | [{layer['auroc_ci_lower']:.3f}, {layer['auroc_ci_upper']:.3f}] | {layer['auprc']:.3f} | {layer['f1']:.3f} |
| Combined | {comb['auroc']:.3f} | [{comb['auroc_ci_lower']:.3f}, {comb['auroc_ci_upper']:.3f}] | {comb['auprc']:.3f} | {comb['f1']:.3f} |

**Combined − uncertainty:** {comp['auroc_diff']:.3f} [{comp['ci_lower']:.3f}, {comp['ci_upper']:.3f}]  
**DeLong p-value (locked test):** {comp['delong_p_value']:.4f}

## Conservative conclusion

- Combined beats uncertainty on locked test: **{'yes' if beats_unc else 'no'}** ({comb['auroc']:.3f} vs {unc['auroc']:.3f}).
- Statistically significant at α=0.05 (bootstrap CI excludes 0 **and** DeLong p < 0.05): **{'yes' if sig else 'no'}**.
- Layer choice was blind to locked test; this is stronger evidence than full-cohort layer picking.

{'The locked holdout supports deploying uncertainty + ' + chosen_layer + ' over uncertainty alone, pending external validation.' if beats_unc and comp['ci_lower'] > 0 else 'The locked holdout does not provide strong support for combined over uncertainty alone.'}
"""
    return report
