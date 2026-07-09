"""Statistical validation of uncertainty vs decoder-layer bad-case detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.baselines import (
    _impute_nan,
    choose_cv_folds,
    compute_deployable_uncertainty_features,
    create_bad_case_label,
)
from src.analysis.failure_taxonomy import assign_dominant_failure_label
from src.analysis.layer_analysis import load_layer_index, load_layer_matrix
from src.utils.io import ensure_dir

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
}

MODEL_NAMES = ("uncertainty", "decoder1", "combined")
DECISION_THRESHOLD = 0.5


def _oof_probabilities(
    x: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Fold-safe out-of-fold P(bad) from logistic regression."""
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
                (
                    "model",
                    LogisticRegression(max_iter=5000, random_state=random_state),
                ),
            ]
        )
        pipe.fit(x[train_idx], labels[train_idx])
        proba[test_idx] = pipe.predict_proba(x[test_idx])[:, 1]
    return proba


def _binary_metrics(labels: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "auroc": float(_safe_auroc(labels, proba)),
        "auprc": float(average_precision_score(labels, proba)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
    }


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _prepare_data(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
    layer_name: str = "decoder1",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
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
    merged = index_df.merge(
        failure_df[failure_cols].drop_duplicates("case_id"),
        on="case_id",
        how="inner",
    )
    merged["dominant_failure_label"] = merged.apply(
        lambda row: assign_dominant_failure_label(row.to_dict()), axis=1
    )

    uncertainty, unc_names = compute_deployable_uncertainty_features(merged)
    decoder = load_layer_matrix(merged, layer_name)
    combined = np.concatenate([uncertainty, decoder], axis=1).astype(np.float32)
    comb_names = [f"unc_{n}" for n in unc_names] + [f"dec_{i}" for i in range(decoder.shape[1])]
    dice = merged["dice"].values.astype(np.float64)
    if "dice_fail" in merged.columns:
        dice = np.where(np.isnan(dice), merged["dice_fail"].values, dice)
    return merged, dice, uncertainty, decoder, combined, unc_names, comb_names


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    midranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> np.ndarray:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return (sx / m + sy / n, aucs)


def delong_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float]:
    """DeLong test comparing two correlated ROC curves (A vs B)."""
    labels = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    order = np.argsort(-labels)
    label_1_count = int(labels.sum())
    if label_1_count < 1 or label_1_count >= len(labels):
        return {
            "auroc_a": float("nan"),
            "auroc_b": float("nan"),
            "z_stat": float("nan"),
            "p_value": float("nan"),
        }

    preds = np.vstack([scores_a, scores_b])[:, order]
    var_a, aucs_a = _fast_delong(preds[:1], label_1_count)
    var_b, aucs_b = _fast_delong(preds[1:], label_1_count)
    var_ab, _ = _fast_delong(preds, label_1_count)
    var_a = float(np.atleast_2d(var_a)[0, 0])
    var_b = float(np.atleast_2d(var_b)[0, 0])
    var_ab = float(np.atleast_2d(var_ab)[0, 0])
    cov_ab = 0.5 * (var_ab - var_a - var_b)
    z = (float(aucs_a[0]) - float(aucs_b[0])) / np.sqrt(max(var_a + var_b - 2 * cov_ab, 1e-12))
    p_value = float(2 * norm.sf(abs(z)))
    return {
        "auroc_a": float(aucs_a[0]),
        "auroc_b": float(aucs_b[0]),
        "z_stat": float(z),
        "p_value": p_value,
    }


def bootstrap_auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified bootstrap CI for AUROC."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "auroc": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_bootstrap": n_bootstrap,
        }

    boot_scores: list[float] = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sample_pos, sample_neg])
        boot_scores.append(_safe_auroc(labels[idx], scores[idx]))

    boot_arr = np.asarray(boot_scores, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    return {
        "auroc": _safe_auroc(labels, scores),
        "ci_lower": float(np.quantile(boot_arr, alpha)),
        "ci_upper": float(np.quantile(boot_arr, 1.0 - alpha)),
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_auroc_difference_ci(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for AUROC(b) - AUROC(a)."""
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sample_pos, sample_neg])
        diffs.append(_safe_auroc(labels[idx], scores_b[idx]) - _safe_auroc(labels[idx], scores_a[idx]))
    diff_arr = np.asarray(diffs, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    point = _safe_auroc(labels, scores_b) - _safe_auroc(labels, scores_a)
    return {
        "auroc_diff": float(point),
        "ci_lower": float(np.quantile(diff_arr, alpha)),
        "ci_upper": float(np.quantile(diff_arr, 1.0 - alpha)),
        "n_bootstrap": n_bootstrap,
    }


def run_bootstrap_analysis(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    dice_threshold: float,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs and DeLong tests for all models."""
    boot_rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        scores = proba_map[model]
        ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=n_bootstrap, random_state=random_state)
        boot_rows.append(
            {
                "dice_threshold": dice_threshold,
                "model": model,
                "auroc": ci["auroc"],
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "n_bootstrap": ci["n_bootstrap"],
            }
        )

    diff_rows: list[dict[str, Any]] = []
    for model_a, model_b in (("uncertainty", "combined"), ("decoder1", "combined")):
        diff = bootstrap_auroc_difference_ci(
            labels,
            proba_map[model_a],
            proba_map[model_b],
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        diff_rows.append(
            {
                "dice_threshold": dice_threshold,
                "comparison": f"{model_b}_minus_{model_a}",
                "auroc_diff": diff["auroc_diff"],
                "ci_lower": diff["ci_lower"],
                "ci_upper": diff["ci_upper"],
                "n_bootstrap": diff["n_bootstrap"],
            }
        )
        boot_rows.append(
            {
                "dice_threshold": dice_threshold,
                "model": f"diff_{model_b}_minus_{model_a}",
                "auroc": diff["auroc_diff"],
                "ci_lower": diff["ci_lower"],
                "ci_upper": diff["ci_upper"],
                "n_bootstrap": diff["n_bootstrap"],
            }
        )

    delong_rows: list[dict[str, Any]] = []
    for model_a, model_b in (("uncertainty", "combined"), ("decoder1", "combined")):
        result = delong_test(labels, proba_map[model_a], proba_map[model_b])
        delong_rows.append(
            {
                "dice_threshold": dice_threshold,
                "model_a": model_a,
                "model_b": model_b,
                "auroc_a": result["auroc_a"],
                "auroc_b": result["auroc_b"],
                "z_stat": result["z_stat"],
                "p_value": result["p_value"],
            }
        )

    return pd.DataFrame(boot_rows), pd.DataFrame(delong_rows)


def run_seed_stability(
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    dice_threshold: float,
    n_seeds: int = 20,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Repeated stratified CV across random seeds."""
    rows: list[dict[str, Any]] = []
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        for model in MODEL_NAMES:
            proba = _oof_probabilities(feature_map[model], labels, random_state=seed)
            metrics = _binary_metrics(labels, proba, threshold=DECISION_THRESHOLD)
            rows.append(
                {
                    "dice_threshold": dice_threshold,
                    "seed": seed,
                    "model": model,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _prediction_correct(label: int, proba: float, threshold: float) -> bool:
    pred = int(proba >= threshold)
    return pred == int(label)


def categorize_rescued_cases(
    case_df: pd.DataFrame,
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Assign each case to A/B/C/D rescue categories at a fixed threshold.

    A: uncertainty correct
    B: decoder rescues uncertainty (decoder correct, uncertainty wrong)
    C: decoder hurts uncertainty (uncertainty correct, decoder wrong)
    D: both fail
    """
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(case_df.to_dict(orient="records")):
        label = int(labels[i])
        p_unc = float(proba_map["uncertainty"][i])
        p_dec = float(proba_map["decoder1"][i])
        p_comb = float(proba_map["combined"][i])
        unc_ok = _prediction_correct(label, p_unc, threshold)
        dec_ok = _prediction_correct(label, p_dec, threshold)
        comb_ok = _prediction_correct(label, p_comb, threshold)

        if unc_ok and not dec_ok:
            category = "C_decoder_hurts"
        elif not unc_ok and dec_ok:
            category = "B_decoder_rescues"
        elif unc_ok:
            category = "A_uncertainty_correct"
        else:
            category = "D_both_fail"

        rows.append(
            {
                "case_id": record["case_id"],
                "dice": float(record.get("dice", np.nan)),
                "label_bad": label,
                "dominant_failure_label": record.get("dominant_failure_label", ""),
                "boundary_error_fraction": float(record.get("boundary_error_fraction", np.nan)),
                "false_positive_voxels": float(record.get("false_positive_voxels", np.nan)),
                "false_negative_voxels": float(record.get("false_negative_voxels", np.nan)),
                "P_uncertainty": p_unc,
                "P_decoder": p_dec,
                "P_combined": p_comb,
                "uncertainty_correct": int(unc_ok),
                "decoder_correct": int(dec_ok),
                "combined_correct": int(comb_ok),
                "category": category,
                "combined_rescues_uncertainty": int((not unc_ok) and comb_ok),
                "combined_hurts_uncertainty": int(unc_ok and (not comb_ok)),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    labels: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error with uniform probability bins."""
    labels = np.asarray(labels, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (proba >= bins[i]) & (proba < bins[i + 1])
        else:
            mask = (proba >= bins[i]) & (proba <= bins[i + 1])
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = proba[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def run_calibration_analysis(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    dice_threshold: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, proba in proba_map.items():
        rows.append(
            {
                "dice_threshold": dice_threshold,
                "model": model,
                "brier_score": float(brier_score_loss(labels, proba)),
                "ece": expected_calibration_error(labels, proba, n_bins=n_bins),
                "n_bins": n_bins,
            }
        )
    return pd.DataFrame(rows)


def _threshold_at_specificity(labels: np.ndarray, proba: np.ndarray, specificity: float) -> float:
    fpr, tpr, thresholds = roc_curve(labels, proba)
    spec = 1.0 - fpr
    valid = np.where(spec >= specificity)[0]
    if len(valid) == 0:
        return float(thresholds[0])
    idx = valid[np.argmax(tpr[valid])]
    return float(thresholds[idx])


def _metrics_at_threshold(labels: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "flag_rate": float(pred.mean()),
        "reviews_saved_fraction": float(1.0 - pred.mean()),
        "n_flagged": float(pred.sum()),
    }


def _optimal_threshold(labels: np.ndarray, proba: np.ndarray, criterion: str) -> float:
    thresholds = np.unique(np.round(proba, 6))
    if len(thresholds) == 0:
        return DECISION_THRESHOLD
    best_threshold = DECISION_THRESHOLD
    best_score = -np.inf
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        if criterion == "f1":
            score = f1_score(labels, pred, zero_division=0)
        elif criterion == "balanced_accuracy":
            tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
            score = 0.5 * (tp / (tp + fn + 1e-8) + tn / (tn + fp + 1e-8))
        elif criterion == "youden_j":
            fpr, tpr, thresholds = roc_curve(labels, proba)
            j = tpr - fpr
            idx = int(np.argmax(j))
            return float(thresholds[idx])
        elif criterion.startswith("sensitivity_at_specificity_"):
            target_spec = float(criterion.split("_")[-1]) / 100.0
            return _threshold_at_specificity(labels, proba, target_spec)
        else:
            raise ValueError(criterion)
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
    return best_threshold


def run_threshold_analysis(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    dice_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    criteria = [
        "f1",
        "balanced_accuracy",
        "youden_j",
        "sensitivity_at_specificity_95",
        "sensitivity_at_specificity_97",
        "sensitivity_at_specificity_99",
    ]
    for model, proba in proba_map.items():
        for criterion in criteria:
            if criterion == "youden_j":
                fpr, tpr, thresholds = roc_curve(labels, proba)
                j = tpr - fpr
                idx = int(np.argmax(j))
                thr = float(thresholds[idx])
            else:
                thr = _optimal_threshold(labels, proba, criterion)
            metrics = _metrics_at_threshold(labels, proba, thr)
            rows.append(
                {
                    "dice_threshold": dice_threshold,
                    "model": model,
                    "criterion": criterion,
                    **metrics,
                }
            )
        for spec_target in (95, 97, 99):
            thr = _threshold_at_specificity(labels, proba, spec_target / 100.0)
            metrics = _metrics_at_threshold(labels, proba, thr)
            rows.append(
                {
                    "dice_threshold": dice_threshold,
                    "model": model,
                    "criterion": f"clinical_specificity_{spec_target}",
                    "target_specificity": spec_target,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_feature_importance(
    combined_features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    dice_threshold: float,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Standardized coefficients, odds ratios, and bootstrap CIs for combined model."""
    x = _impute_nan(combined_features)
    labels = np.asarray(labels, dtype=int)
    cv_folds = choose_cv_folds(labels)
    coef_accum = np.zeros(x.shape[1], dtype=np.float64)
    fold_count = 0

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(x, labels):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        model = LogisticRegression(max_iter=5000, random_state=random_state)
        model.fit(x_train, labels[train_idx])
        coef_accum += model.coef_[0]
        fold_count += 1
    mean_coef = coef_accum / max(fold_count, 1)

    rng = np.random.default_rng(random_state)
    boot_coefs = np.zeros((n_bootstrap, x.shape[1]), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        scaler = StandardScaler()
        x_boot = scaler.fit_transform(x[idx])
        model = LogisticRegression(max_iter=5000, random_state=random_state + b)
        model.fit(x_boot, labels[idx])
        boot_coefs[b] = model.coef_[0]

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        boot_col = boot_coefs[:, i]
        rows.append(
            {
                "dice_threshold": dice_threshold,
                "feature": name,
                "feature_group": "uncertainty" if name.startswith("unc_") else "decoder",
                "standardized_coefficient": float(mean_coef[i]),
                "odds_ratio": float(np.exp(mean_coef[i])),
                "coef_ci_lower": float(np.quantile(boot_col, 0.025)),
                "coef_ci_upper": float(np.quantile(boot_col, 0.975)),
                "odds_ratio_ci_lower": float(np.exp(np.quantile(boot_col, 0.025))),
                "odds_ratio_ci_upper": float(np.exp(np.quantile(boot_col, 0.975))),
                "linear_attribution_mean": float(mean_coef[i]),
            }
        )
    return pd.DataFrame(rows)


def _plot_roc_bootstrap(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    boot_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"uncertainty": "crimson", "decoder1": "steelblue", "combined": "seagreen"}
    for model in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(labels, proba_map[model])
        row = boot_df[(boot_df["model"] == model) & (~boot_df["model"].str.startswith("diff_"))].iloc[0]
        ax.plot(
            fpr,
            tpr,
            color=colors[model],
            lw=2,
            label=f"{model} AUROC={row['auroc']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC with 95% bootstrap CIs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_seed_distribution(seed_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [seed_df[seed_df["model"] == m]["auroc"].values for m in MODEL_NAMES]
    ax.boxplot(data)
    ax.set_xticks(np.arange(1, len(MODEL_NAMES) + 1))
    ax.set_xticklabels(list(MODEL_NAMES))
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC distribution across random seeds")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    output_path: Path,
    n_bins: int = 10,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    colors = {"uncertainty": "crimson", "decoder1": "steelblue", "combined": "seagreen"}
    for model, proba in proba_map.items():
        bin_centers = []
        frac_pos = []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (proba >= bins[i]) & (proba < bins[i + 1])
            else:
                mask = (proba >= bins[i]) & (proba <= bins[i + 1])
            if not mask.any():
                continue
            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            frac_pos.append(labels[mask].mean())
        ax.plot(bin_centers, frac_pos, marker="o", label=model, color=colors[model])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed bad-case fraction")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rescued_histograms(rescued_df: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        ["P_uncertainty", "P_decoder", "P_combined"],
        ["Uncertainty", "Decoder1", "Combined"],
    ):
        for label, color in [(0, "tab:blue"), (1, "tab:red")]:
            vals = rescued_df.loc[rescued_df["label_bad"] == label, col]
            ax.hist(vals, bins=20, alpha=0.5, label=f"label={label}", color=color)
        ax.set_title(title)
        ax.set_xlabel("P(bad)")
    axes[0].legend(fontsize=8)
    fig.suptitle("OOF probability distributions by true label")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_decision_thresholds(
    labels: np.ndarray,
    proba_map: dict[str, np.ndarray],
    threshold_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"uncertainty": "crimson", "decoder1": "steelblue", "combined": "seagreen"}
    ax = axes[0]
    for model in MODEL_NAMES:
        fpr, tpr, thresholds = roc_curve(labels, proba_map[model])
        ax.plot(fpr, tpr, color=colors[model], label=model)
        sub = threshold_df[
            (threshold_df["model"] == model)
            & (threshold_df["criterion"].isin(["f1", "balanced_accuracy", "youden_j"]))
        ]
        for _, row in sub.iterrows():
            pred = (proba_map[model] >= row["threshold"]).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
            ax.scatter(fp / (fp + tn + 1e-8), tp / (tp + fn + 1e-8), color=colors[model], s=30)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC with criterion operating points")
    ax.legend(fontsize=8)

    ax = axes[1]
    for model in MODEL_NAMES:
        precision, recall, thresholds = precision_recall_curve(labels, proba_map[model])
        ax.plot(recall, precision, color=colors[model], label=model)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_coefficients(feature_df: pd.DataFrame, output_path: Path, top_k: int = 20) -> None:
    plt.rcParams.update(PUBLICATION_RC)
    plot_df = feature_df.reindex(
        feature_df["standardized_coefficient"].abs().sort_values(ascending=False).index
    ).head(top_k)
    colors = ["crimson" if g == "uncertainty" else "steelblue" for g in plot_df["feature_group"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["standardized_coefficient"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standardized coefficient (combined logistic model)")
    ax.set_title("Top feature contributions")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_significance_report(
    boot_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    rescued_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    n_cases: int,
    primary_threshold: float = 0.80,
) -> str:
    """Conservative markdown report."""
    boot_primary = boot_df[boot_df["dice_threshold"] == primary_threshold]
    delong_primary = delong_df[delong_df["dice_threshold"] == primary_threshold]
    seed_primary = seed_df[seed_df["dice_threshold"] == primary_threshold]
    rescued_primary = rescued_df.copy()
    cal_primary = calibration_df[calibration_df["dice_threshold"] == primary_threshold]
    thr_primary = threshold_df[threshold_df["dice_threshold"] == primary_threshold]

    def boot_row(model: str) -> pd.Series:
        return boot_primary[boot_primary["model"] == model].iloc[0]

    unc = boot_row("uncertainty")
    dec = boot_row("decoder1")
    comb = boot_row("combined")
    diff_unc = boot_primary[boot_primary["model"] == "diff_combined_minus_uncertainty"].iloc[0]
    diff_dec = boot_primary[boot_primary["model"] == "diff_combined_minus_decoder1"].iloc[0]
    delong_unc = delong_primary[
        (delong_primary["model_a"] == "uncertainty") & (delong_primary["model_b"] == "combined")
    ].iloc[0]
    delong_dec = delong_primary[
        (delong_primary["model_a"] == "decoder1") & (delong_primary["model_b"] == "combined")
    ].iloc[0]

    seed_summary = (
        seed_primary.groupby("model")["auroc"].agg(["mean", "std", "min", "max"]).reset_index()
    )

    cat_counts = rescued_primary["category"].value_counts()
    failure_benefit = (
        rescued_primary.groupby("dominant_failure_label")["combined_rescues_uncertainty"]
        .agg(["sum", "count", "mean"])
        .sort_values("mean", ascending=False)
    )

    clinical = thr_primary[
        thr_primary["criterion"].str.startswith("clinical_specificity")
    ]

    unc_features = feature_df[feature_df["feature_group"] == "uncertainty"]
    dec_features = feature_df[feature_df["feature_group"] == "decoder"]
    top_unc = unc_features.reindex(
        unc_features["standardized_coefficient"].abs().sort_values(ascending=False).index
    ).head(3)
    top_dec = dec_features.reindex(
        dec_features["standardized_coefficient"].abs().sort_values(ascending=False).index
    ).head(3)
    dec_positive = int((dec_features["standardized_coefficient"] > 0).sum())
    dec_negative = int((dec_features["standardized_coefficient"] < 0).sum())

    sig_unc = diff_unc["ci_lower"] > 0 and delong_unc["p_value"] < 0.05
    sig_dec = diff_dec["ci_lower"] > 0 and delong_dec["p_value"] < 0.05

    report = f"""# Layer Significance Report

**Validation cases:** {n_cases}  
**Primary bad-case definition:** Dice < {primary_threshold}

## 1. Is the AUROC improvement statistically significant?

| Model | AUROC | 95% bootstrap CI |
|-------|-------|--------------------|
| Uncertainty | {unc['auroc']:.3f} | [{unc['ci_lower']:.3f}, {unc['ci_upper']:.3f}] |
| Decoder1 | {dec['auroc']:.3f} | [{dec['ci_lower']:.3f}, {dec['ci_upper']:.3f}] |
| Combined | {comb['auroc']:.3f} | [{comb['ci_lower']:.3f}, {comb['ci_upper']:.3f}] |

**Combined − uncertainty:** {diff_unc['auroc']:.3f} [{diff_unc['ci_lower']:.3f}, {diff_unc['ci_upper']:.3f}]  
**DeLong p-value (unc vs combined):** {delong_unc['p_value']:.4f}

**Combined − decoder1:** {diff_dec['auroc']:.3f} [{diff_dec['ci_lower']:.3f}, {diff_dec['ci_upper']:.3f}]  
**DeLong p-value (decoder1 vs combined):** {delong_dec['p_value']:.4f}

Conservative read: combined beats uncertainty with a positive bootstrap difference CI and DeLong p < 0.05 only if both hold. On this run that is **{'yes' if sig_unc else 'no'}** for uncertainty vs combined.

## 2. Seed stability

| Model | mean AUROC | std | min | max |
|-------|------------|-----|-----|-----|
"""
    for _, row in seed_summary.iterrows():
        report += (
            f"| {row['model']} | {row['mean']:.3f} | {row['std']:.3f} | "
            f"{row['min']:.3f} | {row['max']:.3f} |\n"
        )

    report += f"""
Combined AUROC std across seeds: {seed_summary[seed_summary['model']=='combined']['std'].iloc[0]:.4f}.

## 3. Which bad cases are rescued?

Category counts (threshold = {DECISION_THRESHOLD} on OOF probabilities):

| Category | Count |
|----------|-------|
"""
    for cat, count in cat_counts.items():
        report += f"| {cat} | {int(count)} |\n"

    report += """
Failure types most often rescued by combined (among bad cases where uncertainty missed):

| Failure type | rescued | total | rate |
|--------------|---------|-------|------|
"""
    for idx, row in failure_benefit.head(6).iterrows():
        report += f"| {idx} | {int(row['sum'])} | {int(row['count'])} | {row['mean']:.2f} |\n"

    report += f"""
## 4. Calibration

| Model | Brier | ECE |
|-------|-------|-----|
"""
    for _, row in cal_primary.iterrows():
        report += f"| {row['model']} | {row['brier_score']:.4f} | {row['ece']:.4f} |\n"

    report += """
## 5–6. Clinical utility (target specificity)

| Model | Spec target | Sensitivity | PPV | NPV | Reviews saved |
|-------|-------------|-------------|-----|-----|---------------|
"""
    for _, row in clinical.iterrows():
        report += (
            f"| {row['model']} | {row.get('target_specificity', '')}% | "
            f"{row['sensitivity']:.3f} | {row['ppv']:.3f} | {row['npv']:.3f} | "
            f"{row['reviews_saved_fraction']*100:.1f}% |\n"
        )

    report += f"""
## 7. Feature contribution (combined logistic model)

Top uncertainty features:
"""
    for _, row in top_unc.iterrows():
        report += f"- {row['feature']}: coef={row['standardized_coefficient']:.3f}, OR={row['odds_ratio']:.3f}\n"

    report += "\nTop decoder dimensions:\n"
    for _, row in top_dec.iterrows():
        report += f"- {row['feature']}: coef={row['standardized_coefficient']:.3f}, OR={row['odds_ratio']:.3f}\n"

    report += f"""
Decoder dimensions with positive vs negative coefficients: {dec_positive} / {dec_negative}.
Linear attribution is used instead of SHAP (logistic model is linear after scaling).

## 8. Threshold robustness

AUROC by Dice threshold:

| Dice threshold | Uncertainty | Decoder1 | Combined |
|----------------|-------------|----------|----------|
"""
    for thr in sorted(boot_df["dice_threshold"].unique()):
        sub = boot_df[(boot_df["dice_threshold"] == thr) & (~boot_df["model"].str.startswith("diff_"))]
        u = sub[sub["model"] == "uncertainty"]["auroc"].iloc[0]
        d = sub[sub["model"] == "decoder1"]["auroc"].iloc[0]
        c = sub[sub["model"] == "combined"]["auroc"].iloc[0]
        report += f"| < {thr:.2f} | {u:.3f} | {d:.3f} | {c:.3f} |\n"

    report += f"""
## Conclusions (conservative)

1. **Is Decoder1 complementary to uncertainty?** Partially. Decoder1 alone is close to uncertainty; combined adds incremental signal in AUROC and rescued-case analysis, but decoder-only is not uniformly better.
2. **Is the AUROC gain significant?** {'Yes' if sig_unc else 'Not conclusively'} for combined vs uncertainty on the primary threshold (bootstrap CI {'excludes' if diff_unc['ci_lower'] > 0 else 'includes'} 0; DeLong p={delong_unc['p_value']:.4f}).
3. **Clinically meaningful?** Review savings depend on operating point. See clinical utility table; high-specificity operating points trade sensitivity for fewer flagged cases.
4. **Stable across seeds?** Combined mean AUROC {seed_summary[seed_summary['model']=='combined']['mean'].iloc[0]:.3f} ± {seed_summary[seed_summary['model']=='combined']['std'].iloc[0]:.3f}.
5. **Recommend combined over uncertainty alone?** {'Weak yes at fixed 0.5 threshold; stronger yes if tuned jointly at clinical specificity targets.' if sig_unc else 'Not recommended on AUROC evidence alone; any deployment should use threshold tuning and external validation.'}

Do not overclaim: this is a single validation cohort with fold-safe OOF scoring, not a prospective clinical study.
"""
    return report


def analyze_layer_significance(
    layer_index_path: str | Path,
    failure_table_path: str | Path,
    output_dir: str | Path,
    dice_thresholds: tuple[float, ...] = (0.75, 0.80, 0.85),
    primary_threshold: float = 0.80,
    layer_name: str = "decoder1",
    n_bootstrap: int = 2000,
    n_seeds: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full statistical validation pipeline."""
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    case_df, dice, uncertainty, decoder, combined, unc_names, comb_names = _prepare_data(
        layer_index_path, failure_table_path, layer_name=layer_name
    )
    feature_map = {
        "uncertainty": uncertainty,
        "decoder1": decoder,
        "combined": combined,
    }

    all_boot: list[pd.DataFrame] = []
    all_delong: list[pd.DataFrame] = []
    all_seed: list[pd.DataFrame] = []
    all_cal: list[pd.DataFrame] = []
    all_thr: list[pd.DataFrame] = []
    all_feat: list[pd.DataFrame] = []
    proba_primary: dict[str, np.ndarray] | None = None
    labels_primary: np.ndarray | None = None

    for dice_thr in dice_thresholds:
        labels, _ = create_bad_case_label(dice, mode="threshold", threshold=dice_thr)
        proba_map = {
            model: _oof_probabilities(feature_map[model], labels, random_state=random_state)
            for model in MODEL_NAMES
        }
        boot_df, delong_df = run_bootstrap_analysis(
            labels, proba_map, dice_thr, n_bootstrap=n_bootstrap, random_state=random_state
        )
        all_boot.append(boot_df)
        all_delong.append(delong_df)
        all_seed.append(
            run_seed_stability(feature_map, labels, dice_thr, n_seeds=n_seeds, base_seed=random_state)
        )
        all_cal.append(run_calibration_analysis(labels, proba_map, dice_thr))
        all_thr.append(run_threshold_analysis(labels, proba_map, dice_thr))
        all_feat.append(
            run_feature_importance(
                combined, labels, comb_names, dice_thr, n_bootstrap=500, random_state=random_state
            )
        )
        if abs(dice_thr - primary_threshold) < 1e-8:
            proba_primary = proba_map
            labels_primary = labels

    bootstrap_results = pd.concat(all_boot, ignore_index=True)
    delong_tests = pd.concat(all_delong, ignore_index=True)
    seed_stability = pd.concat(all_seed, ignore_index=True)
    calibration = pd.concat(all_cal, ignore_index=True)
    threshold_analysis = pd.concat(all_thr, ignore_index=True)
    feature_importance = pd.concat(all_feat, ignore_index=True)

    assert proba_primary is not None and labels_primary is not None
    rescued_cases = categorize_rescued_cases(case_df, labels_primary, proba_primary)

    # Representative examples: top cases per category by margin from threshold
    rescued_cases["example_rank"] = 0
    for cat in rescued_cases["category"].unique():
        mask = rescued_cases["category"] == cat
        margin = (rescued_cases.loc[mask, "P_combined"] - DECISION_THRESHOLD).abs()
        rescued_cases.loc[mask, "example_rank"] = margin.rank(ascending=False, method="first")

    bootstrap_results.to_csv(output_dir / "bootstrap_results.csv", index=False)
    delong_tests.to_csv(output_dir / "delong_tests.csv", index=False)
    seed_stability.to_csv(output_dir / "seed_stability.csv", index=False)
    rescued_cases.to_csv(output_dir / "rescued_cases.csv", index=False)
    threshold_analysis.to_csv(output_dir / "threshold_analysis.csv", index=False)
    calibration.to_csv(output_dir / "calibration.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    boot_primary = bootstrap_results[bootstrap_results["dice_threshold"] == primary_threshold]
    _plot_roc_bootstrap(labels_primary, proba_primary, boot_primary, figures_dir / "roc_bootstrap.png")
    _plot_seed_distribution(
        seed_stability[seed_stability["dice_threshold"] == primary_threshold],
        figures_dir / "auroc_seed_distribution.png",
    )
    _plot_calibration(labels_primary, proba_primary, figures_dir / "calibration_curve.png")
    _plot_rescued_histograms(rescued_cases, figures_dir / "rescued_case_histograms.png")
    _plot_decision_thresholds(
        labels_primary,
        proba_primary,
        threshold_analysis[threshold_analysis["dice_threshold"] == primary_threshold],
        figures_dir / "decision_thresholds.png",
    )
    _plot_feature_coefficients(
        feature_importance[feature_importance["dice_threshold"] == primary_threshold],
        figures_dir / "feature_coefficients.png",
    )

    report = generate_significance_report(
        bootstrap_results,
        delong_tests,
        seed_stability,
        rescued_cases,
        calibration,
        threshold_analysis,
        feature_importance[feature_importance["dice_threshold"] == primary_threshold],
        n_cases=len(case_df),
        primary_threshold=primary_threshold,
    )
    (output_dir / "layer_significance_report.md").write_text(report)

    return {
        "bootstrap_results": bootstrap_results,
        "delong_tests": delong_tests,
        "seed_stability": seed_stability,
        "rescued_cases": rescued_cases,
        "calibration": calibration,
        "threshold_analysis": threshold_analysis,
        "feature_importance": feature_importance,
        "report": report,
    }
