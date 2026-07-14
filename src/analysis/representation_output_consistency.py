"""Representation–output consistency features and nested-CV failure detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis.layer_interventions import compute_prediction_quantities
from src.analysis.layer_io import build_anatomy_table, load_layer_index, load_layer_matrix
from src.training.spatial_repair_trainer import (
    edema_dice,
    mean_foreground_dice,
)
from src.utils.io import ensure_dir

CLS_EDEMA = 2
CLS_ENH = 3
CLS_NEC = 1

# Inference-time confidence requires entropy/probability maps. The columns in
# failure_metrics.csv (mean_entropy_error, entropy_error_auroc, overlap_top_*,
# confident_false_negative_fraction, ...) are computed against GT error masks in
# analyze.py and MUST NOT be used as detector features.
CONFIDENCE_COLS: list[str] = []

# Stored failure-table columns that leak ground truth via error masks / FN labels.
GT_LEAKED_UNCERTAINTY_COLS = (
    "mean_entropy_error",
    "mean_entropy_nonerror",
    "entropy_error_auroc",
    "entropy_error_auprc",
    "overlap_top_5_uncertainty",
    "overlap_top_10_uncertainty",
    "overlap_top_20_uncertainty",
    "confident_false_negative_fraction",
    "false_positive_voxels",
    "false_negative_voxels",
    "missed_small_lesion_count",
    "boundary_error_fraction",
)

MORPHOLOGY_COLS = [
    "pred_tumor_volume",
    "pred_edema_volume",
    "pred_enhancing_volume",
    "pred_necrosis_volume",
    "out_log_wt_volume",
    "out_gt_edema_frac",
    "out_gt_enhancing_frac",
    "out_gt_necrosis_frac",
    "out_gt_compactness",
    "n_components",
    "surface_to_volume",
]

# Ground-truth anatomy / quality columns that must never enter detectors as features.
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "gt_anatomy_",
    "label_",
    "true_",
    "path_",
)

FORBIDDEN_FEATURE_EXACT = frozenset(GT_LEAKED_UNCERTAINTY_COLS)


def pred_mask_anatomy(pred: np.ndarray) -> dict[str, float]:
    """Anatomy measured from predicted mask only (no GT dependence for volumes)."""
    nec = float((pred == CLS_NEC).sum())
    ed = float((pred == CLS_EDEMA).sum())
    enh = float((pred == CLS_ENH).sum())
    wt = nec + ed + enh
    q = compute_prediction_quantities(pred, pred)
    surface_to_vol = float(q["boundary_complexity"] * (wt ** (2.0 / 3.0)) / (wt + 1e-8)) if wt > 0 else 0.0
    return {
        "out_log_wt_volume": float(np.log1p(wt)),
        "out_gt_edema_frac": float(ed / (wt + 1e-8)),
        "out_gt_enhancing_frac": float(enh / (wt + 1e-8)),
        "out_gt_necrosis_frac": float(nec / (wt + 1e-8)),
        "out_gt_compactness": float(q["boundary_complexity"]),
        "pred_tumor_volume": wt,
        "pred_edema_volume": ed,
        "pred_enhancing_volume": enh,
        "pred_necrosis_volume": nec,
        "n_components": float(_n_components(pred > 0)),
        "surface_to_volume": surface_to_vol,
    }


def _n_components(mask: np.ndarray) -> int:
    from scipy import ndimage

    _, n = ndimage.label(mask.astype(np.uint8))
    return int(n)


def quality_labels(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "label_edema_dice": float(edema_dice(pred, gt)),
        "label_mean_fg_dice": float(mean_foreground_dice(pred, gt)),
    }


def oof_ridge_predictions(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    alphas: list[float],
) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    preds = np.full(len(y), np.nan, dtype=np.float64)
    n_splits = min(n_splits, len(y))
    if n_splits < 2:
        scaler = StandardScaler()
        xs = scaler.fit_transform(np.nan_to_num(x, nan=0.0))
        model = RidgeCV(alphas=np.asarray(alphas, dtype=float))
        model.fit(xs, y)
        preds[:] = model.predict(xs)
        return preds, float("nan")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(x):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(np.nan_to_num(x[tr], nan=0.0))
        xte = scaler.transform(np.nan_to_num(x[te], nan=0.0))
        model = RidgeCV(alphas=np.asarray(alphas, dtype=float))
        model.fit(xtr, y[tr])
        preds[te] = model.predict(xte)
    valid = np.isfinite(preds) & np.isfinite(y)
    r2 = float(r2_score(y[valid], preds[valid])) if valid.sum() > 2 else float("nan")
    return preds, r2


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(np.nan_to_num(x_train, nan=0.0))
    xte = scaler.transform(np.nan_to_num(x_test, nan=0.0))
    model = RidgeCV(alphas=np.asarray(alphas, dtype=float))
    model.fit(xtr, y_train)
    return model.predict(xte)


def select_best_layer(
    layer_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    layers: list[str],
    n_splits: int,
    seed: int,
    alphas: list[float],
) -> tuple[str, float, np.ndarray]:
    best_layer, best_r2, best_preds = layers[0], -1e9, None
    for layer in layers:
        preds, r2 = oof_ridge_predictions(
            layer_matrices[layer],
            y,
            n_splits=n_splits,
            seed=seed,
            alphas=alphas,
        )
        score = -1e9 if not np.isfinite(r2) else r2
        if score > best_r2:
            best_layer, best_r2, best_preds = layer, score, preds
    assert best_preds is not None
    return best_layer, float(best_r2), best_preds


def build_gap_features(
    rep: np.ndarray,
    out: np.ndarray,
    train_idx: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    signed = rep - out
    absolute = np.abs(signed)
    std = float(np.nanstd(absolute[train_idx])) if len(train_idx) else 1.0
    std = max(std, 1e-8)
    relative = absolute / (np.abs(out) + 1e-6)
    # Cap relative gaps using training-fold 99th percentile to avoid Ridge blow-ups.
    cap = float(np.nanquantile(relative[train_idx], 0.99)) if len(train_idx) else 10.0
    cap = max(cap, 1e-6)
    relative = np.clip(relative, 0.0, cap)
    return {
        f"signed_gap_{prefix}": signed,
        f"absolute_gap_{prefix}": absolute,
        f"standardized_gap_{prefix}": absolute / std,
        f"relative_gap_{prefix}": relative,
    }


def impute_train_stats(
    x: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaNs using training-fold column means only."""
    out = x.astype(np.float64).copy()
    means = np.zeros(out.shape[1], dtype=np.float64)
    for j in range(out.shape[1]):
        col_tr = out[train_idx, j]
        m = float(np.nanmean(col_tr)) if np.isfinite(col_tr).any() else 0.0
        means[j] = m
        col = out[:, j]
        col[~np.isfinite(col)] = m
        out[:, j] = col
    return out, means


def risk_coverage_curve(
    risk_scores: np.ndarray,
    quality: np.ndarray,
    coverages: list[float] | None = None,
) -> dict[str, float]:
    if coverages is None:
        coverages = [0.5, 0.7, 0.8, 0.9]
    order = np.argsort(risk_scores)
    quality_sorted = quality[order]
    n = len(quality)
    grid = np.linspace(0.05, 1.0, 20)
    risks = []
    for c in grid:
        k = max(1, int(round(c * n)))
        risks.append(1.0 - float(np.mean(quality_sorted[:k])))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    out: dict[str, float] = {"aurc": float(trapz(risks, grid))}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[f"mean_dice_coverage_{int(c * 100)}"] = float(np.mean(quality_sorted[:k]))
    return out


def failure_capture_at_budget(
    risk_scores: np.ndarray,
    is_failure: np.ndarray,
    budgets: list[float] | None = None,
) -> dict[str, float]:
    if budgets is None:
        budgets = [0.1, 0.2, 0.3]
    order = np.argsort(-risk_scores)
    fails = is_failure.astype(bool)[order]
    n_fail = max(int(fails.sum()), 1)
    out: dict[str, float] = {}
    n = len(risk_scores)
    for b in budgets:
        k = max(1, int(round(b * n)))
        out[f"capture_at_{int(b * 100)}"] = float(fails[:k].sum() / n_fail)
    return out


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
        }
    pred = (scores >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    cal = LinearRegression().fit(scores.reshape(-1, 1), y_true.astype(float))
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6))),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "calibration_slope": float(cal.coef_[0]),
        "calibration_intercept": float(cal.intercept_),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def bootstrap_ci(
    values_fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(float(values_fn(idx)))
        except Exception:
            continue
    if not vals:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.asarray(vals)
    return {
        "point": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def tune_logistic(
    x: np.ndarray,
    y: np.ndarray,
    Cs: list[float],
    seed: int,
    n_splits: int = 4,
) -> LogisticRegression:
    y = y.astype(int)
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        model = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced")
        model.fit(x, y)
        return model
    n_splits = min(n_splits, int(y.sum()), int(len(y) - y.sum()))
    if n_splits < 2:
        model = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced")
        model.fit(x, y)
        return model
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c, best_score = Cs[0], -1.0
    for c in Cs:
        scores = []
        for tr, te in splitter.split(x, y):
            if y[tr].sum() == 0 or y[tr].sum() == len(tr):
                continue
            m = LogisticRegression(C=c, max_iter=4000, class_weight="balanced")
            m.fit(x[tr], y[tr])
            s = m.predict_proba(x[te])[:, 1]
            try:
                scores.append(average_precision_score(y[te], s))
            except Exception:
                continue
        mean_s = float(np.mean(scores)) if scores else -1.0
        if mean_s > best_score:
            best_score, best_c = mean_s, c
    model = LogisticRegression(C=best_c, max_iter=4000, class_weight="balanced")
    model.fit(x, y)
    return model


def tune_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    return fit_ridge_predict(x_train, y_train, x_test, alphas)


FEATURE_GROUPS = {
    "morphology": "morphology",
    "dice_probe": "dice_probe",
    "rep_anatomy": "rep_anatomy",
    "inconsistency": "inconsistency",
    "combined": "combined",
    "pooled_repr": "pooled_repr",
}


def _assert_no_forbidden_features(cols: list[str]) -> None:
    for c in cols:
        if c in FORBIDDEN_FEATURE_EXACT:
            raise ValueError(f"Forbidden GT-leaked feature used as model input: {c}")
        for bad in FORBIDDEN_FEATURE_SUBSTRINGS:
            if bad in c:
                raise ValueError(f"Forbidden ground-truth feature leaked into model: {c}")


ARTIFACT_INVENTORY_MD = """# Artifact inventory for Representation–Output Consistency Score

## Available (375 validation cases)

1. **Layer embeddings** — `outputs_10hour/layer_embeddings/` (all 9 layers × 375)
2. **Case-level OOF probe predictions** — **NOT stored**; refit inside nested CV
3. **Ground-truth anatomical targets** — via `build_anatomy_table` (probe training / labels only)
4. **Predicted-segmentation measurements** — from `path_prediction` (375/375)
5. **Dice / boundary error** — `failure_metrics.csv` (evaluation labels only)
6. **Probability / uncertainty maps** — **missing on disk** (`path_entropy` 0/375)
7. **Predicted masks** — 375/375
8. **Best layers** — selected inside each outer-training fold (not globally)
9. **Existing probe predictions OOF?** — aggregate catalog R² only; case-level preds regenerated in CV
10. **Confidence baseline** — **unavailable without leakage**. Cached columns
    (`mean_entropy_error`, `entropy_error_auroc`, `overlap_top_*`,
    `confident_false_negative_fraction`) use GT error masks in `analyze.py`.
    Inference-time confidence would require regenerating entropy/probability maps.
"""


def build_case_table(
    layer_index_path: Path,
    failure_table_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    index_df = load_layer_index(layer_index_path)
    failure_df = pd.read_csv(failure_table_path)
    fail_extra = failure_df.drop(
        columns=[
            c
            for c in (
                "path_prediction",
                "path_ground_truth",
                "path_entropy",
                "path_embedding",
            )
            if c in failure_df.columns
        ],
        errors="ignore",
    )
    # Keep failure-table dice/boundary if index lacks them.
    merged = index_df.merge(fail_extra, on="case_id", how="inner", suffixes=("", "_fail"))
    if "dice" not in merged.columns and "dice_fail" in merged.columns:
        merged["dice"] = merged["dice_fail"]
    if "boundary_error_fraction" not in merged.columns and "boundary_error_fraction_fail" in merged.columns:
        merged["boundary_error_fraction"] = merged["boundary_error_fraction_fail"]

    anatomy_gt = build_anatomy_table(index_df, failure_df)
    anatomy_gt = anatomy_gt.add_prefix("gt_anatomy_")
    anatomy_gt = anatomy_gt.rename(columns={"gt_anatomy_case_id": "case_id"})

    pred_rows = []
    for rec in merged.to_dict(orient="records"):
        pred = np.load(rec["path_prediction"]).astype(np.int16)
        gt = np.load(rec["path_ground_truth"]).astype(np.int16)
        row = {"case_id": rec["case_id"], **pred_mask_anatomy(pred), **quality_labels(pred, gt)}
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    case_df = merged.merge(pred_df, on="case_id", how="inner").merge(
        anatomy_gt, on="case_id", how="inner"
    )
    case_df = case_df.sort_values("case_id").reset_index(drop=True)
    case_df["label_wt_dice"] = case_df["dice"].astype(float)
    case_df["label_boundary_error"] = case_df["boundary_error_fraction"].astype(float)

    layers = [
        c.replace("path_", "")
        for c in index_df.columns
        if c.startswith("path_")
        and c.replace("path_", "")
        not in {"prediction", "ground_truth", "entropy", "embedding"}
    ]
    layer_matrices = {name: load_layer_matrix(case_df, name) for name in layers}
    return case_df, layer_matrices, anatomy_gt


def nested_cv_predictions(
    case_df: pd.DataFrame,
    layer_matrices: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], np.ndarray],
    dict[str, np.ndarray],
]:
    seed = int(config["seed"])
    outer_folds = int(config["outer_folds"])
    inner_folds = int(config["inner_folds"])
    alphas = list(config["ridge_alphas"])
    Cs = list(config["logistic_C"])
    layers = list(config["layers"])
    inconsistency_targets = list(config["targets"]["inconsistency"])
    quality_probes = list(config["targets"]["quality_probes"])

    n = len(case_df)
    case_ids = case_df["case_id"].to_numpy()
    wt_dice = case_df["label_wt_dice"].to_numpy(dtype=float)
    mean_fg = case_df["label_mean_fg_dice"].to_numpy(dtype=float)
    edema_d = case_df["label_edema_dice"].to_numpy(dtype=float)
    boundary = case_df["label_boundary_error"].to_numpy(dtype=float)

    # Output measurements aligned to inconsistency target names.
    out_map = {
        "log_wt_volume": case_df["out_log_wt_volume"].to_numpy(dtype=float),
        "gt_edema_frac": case_df["out_gt_edema_frac"].to_numpy(dtype=float),
        "gt_enhancing_frac": case_df["out_gt_enhancing_frac"].to_numpy(dtype=float),
        "gt_necrosis_frac": case_df["out_gt_necrosis_frac"].to_numpy(dtype=float),
        "gt_compactness": case_df["out_gt_compactness"].to_numpy(dtype=float),
    }
    gt_map = {
        t: case_df[f"gt_anatomy_{t}"].to_numpy(dtype=float) for t in inconsistency_targets
    }
    for qp in quality_probes:
        gt_map[qp] = case_df[f"gt_anatomy_{qp}"].to_numpy(dtype=float)

    morph_mat = case_df[MORPHOLOGY_COLS].to_numpy(dtype=float)
    pooled = np.concatenate([layer_matrices[L] for L in layers], axis=1)

    # Held-out-only storage for case_level_features.csv (never overwrite with train OOF).
    rep_anatomy_heldout = {t: np.full(n, np.nan) for t in inconsistency_targets}
    rep_quality_heldout = {t: np.full(n, np.nan) for t in quality_probes}
    gap_store: dict[str, np.ndarray] = {}
    best_layer_rows = []
    fold_rows = []

    method_names = [
        "morphology",
        "dice_probe",
        "rep_anatomy",
        "inconsistency",
        "combined",
        "pooled_repr",
        "hgb_combined",
    ]
    fail_defs = ["dice_lt_0.80", "dice_lt_0.70", "edema_lt_0.70", "lowest_20pct"]
    score_store = {
        (m, fdef): np.full(n, np.nan) for m in method_names for fdef in fail_defs
    }
    dice_pred_store = {m: np.full(n, np.nan) for m in method_names if m != "hgb_combined"}
    dice_pred_store["edema_ridge_combined"] = np.full(n, np.nan)
    dice_pred_store["boundary_ridge_combined"] = np.full(n, np.nan)
    fail_lowest_oof = np.full(n, np.nan)

    coef_rows = []
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    for fold_i, (tr, te) in enumerate(outer.split(np.arange(n))):
        for idx in tr:
            fold_rows.append(
                {"case_id": case_ids[idx], "outer_fold": fold_i, "split": "train"}
            )
        for idx in te:
            fold_rows.append(
                {"case_id": case_ids[idx], "outer_fold": fold_i, "split": "test"}
            )

        # --- Probe selection + OOF/test predictions (train-only selection) ---
        fold_gaps: dict[str, np.ndarray] = {}
        fold_rep_anat: dict[str, np.ndarray] = {}
        fold_rep_qual: dict[str, np.ndarray] = {}
        for t in inconsistency_targets:
            best_layer, best_r2, oof_tr = select_best_layer(
                {L: layer_matrices[L][tr] for L in layers},
                gt_map[t][tr],
                layers,
                n_splits=inner_folds,
                seed=seed + fold_i,
                alphas=alphas,
            )
            best_layer_rows.append(
                {
                    "outer_fold": fold_i,
                    "target": t,
                    "best_layer": best_layer,
                    "train_oof_r2": best_r2,
                }
            )
            rep_fold = np.full(n, np.nan)
            rep_fold[tr] = oof_tr
            rep_fold[te] = fit_ridge_predict(
                layer_matrices[best_layer][tr],
                gt_map[t][tr],
                layer_matrices[best_layer][te],
                alphas,
            )
            fold_rep_anat[t] = rep_fold
            # Persist only outer-test predictions (one value per case across folds).
            rep_anatomy_heldout[t][te] = rep_fold[te]
            gaps = build_gap_features(rep_fold, out_map[t], tr, t)
            for k, v in gaps.items():
                fold_gaps[k] = v
                if k not in gap_store:
                    gap_store[k] = np.full(n, np.nan)
                gap_store[k][te] = v[te]

        for t in quality_probes:
            best_layer, best_r2, oof_tr = select_best_layer(
                {L: layer_matrices[L][tr] for L in layers},
                gt_map[t][tr],
                layers,
                n_splits=inner_folds,
                seed=seed + 100 + fold_i,
                alphas=alphas,
            )
            best_layer_rows.append(
                {
                    "outer_fold": fold_i,
                    "target": t,
                    "best_layer": best_layer,
                    "train_oof_r2": best_r2,
                }
            )
            rep_fold = np.full(n, np.nan)
            rep_fold[tr] = oof_tr
            rep_fold[te] = fit_ridge_predict(
                layer_matrices[best_layer][tr],
                gt_map[t][tr],
                layer_matrices[best_layer][te],
                alphas,
            )
            fold_rep_qual[t] = rep_fold
            rep_quality_heldout[t][te] = rep_fold[te]

        # Feature matrices for this fold (full length; train stats for impute/scale).
        gap_names = sorted(fold_gaps.keys())
        gap_mat = np.column_stack([fold_gaps[k] for k in gap_names])
        rep_anat_mat = np.column_stack(
            [fold_rep_anat[t] for t in inconsistency_targets]
        )
        dice_probe_mat = fold_rep_qual["dice"].reshape(-1, 1)
        combined_names = gap_names + MORPHOLOGY_COLS
        _assert_no_forbidden_features(combined_names)
        _assert_no_forbidden_features(MORPHOLOGY_COLS)
        _assert_no_forbidden_features(gap_names)

        feature_sets = {
            "morphology": morph_mat,
            "dice_probe": dice_probe_mat,
            "rep_anatomy": rep_anat_mat,
            "inconsistency": gap_mat,
            # Proposed combined model: inconsistency + output morphology only.
            # Confidence omitted: cached uncertainty columns leak GT error masks.
            "combined": np.column_stack([gap_mat, morph_mat]),
            "pooled_repr": pooled,
        }

        # Failure labels (lowest-20% cutoff from train fold only).
        cut20 = float(np.quantile(mean_fg[tr], 0.20))
        fail_lowest_oof[te] = (mean_fg[te] < cut20).astype(float)
        fail_labels = {
            "dice_lt_0.80": (mean_fg < 0.80).astype(int),
            "dice_lt_0.70": (mean_fg < 0.70).astype(int),
            "edema_lt_0.70": (edema_d < 0.70).astype(int),
            "lowest_20pct": (mean_fg < cut20).astype(int),
        }

        for method, raw_x in feature_sets.items():
            x_imp, _ = impute_train_stats(raw_x, tr)
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_imp[tr])
            x_te = scaler.transform(x_imp[te])

            # Continuous Dice prediction (Ridge); clip to [0, 1] for metric stability.
            y_tr = mean_fg[tr]
            dice_pred_store[method][te] = np.clip(
                tune_ridge_predict(x_tr, y_tr, x_te, alphas), 0.0, 1.0
            )
            if method == "combined":
                dice_pred_store["edema_ridge_combined"][te] = np.clip(
                    tune_ridge_predict(x_tr, edema_d[tr], x_te, alphas), 0.0, 1.0
                )
                dice_pred_store["boundary_ridge_combined"][te] = np.clip(
                    tune_ridge_predict(x_tr, boundary[tr], x_te, alphas), 0.0, 1.0
                )

            for fdef, y_all in fail_labels.items():
                y_tr_cls = y_all[tr]
                model = tune_logistic(
                    x_tr, y_tr_cls, Cs=Cs, seed=seed + fold_i, n_splits=inner_folds
                )
                score_store[(method, fdef)][te] = model.predict_proba(x_te)[:, 1]
                if method == "combined" and fdef == "lowest_20pct":
                    for name, coef in zip(
                        combined_names,
                        model.coef_.ravel(),
                        strict=False,
                    ):
                        coef_rows.append(
                            {
                                "outer_fold": fold_i,
                                "failure_def": fdef,
                                "feature": name,
                                "coefficient": float(coef),
                            }
                        )

            # Optional nonlinear secondary on combined features only.
            if method == "combined":
                for fdef, y_all in fail_labels.items():
                    y_tr_cls = y_all[tr]
                    if y_tr_cls.sum() < 2 or (len(y_tr_cls) - y_tr_cls.sum()) < 2:
                        score_store[("hgb_combined", fdef)][te] = score_store[
                            ("combined", fdef)
                        ][te]
                        continue
                    hgb = HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=100,
                        random_state=seed + fold_i,
                        class_weight="balanced",
                    )
                    hgb.fit(x_tr, y_tr_cls)
                    score_store[("hgb_combined", fdef)][te] = hgb.predict_proba(x_te)[:, 1]

    # Assemble case-level feature / prediction tables (held-out probe/gap values only).
    feat_df = case_df[
        ["case_id", "label_wt_dice", "label_mean_fg_dice", "label_edema_dice", "label_boundary_error"]
        + MORPHOLOGY_COLS
    ].copy()
    for t in inconsistency_targets:
        feat_df[f"rep_{t}"] = rep_anatomy_heldout[t]
        feat_df[f"out_{t}"] = out_map[t]
    for t in quality_probes:
        feat_df[f"rep_{t}"] = rep_quality_heldout[t]
    for k, v in gap_store.items():
        feat_df[k] = v
    assert feat_df[[c for c in feat_df.columns if c.startswith(("rep_", "signed_gap_", "absolute_gap_", "standardized_gap_", "relative_gap_"))]].notna().all().all()

    pred_rows = []
    for i, cid in enumerate(case_ids):
        row = {
            "case_id": cid,
            "label_mean_fg_dice": mean_fg[i],
            "label_edema_dice": edema_d[i],
            "label_wt_dice": wt_dice[i],
            "label_boundary_error": boundary[i],
            "fail_dice_lt_0.80": int(mean_fg[i] < 0.80),
            "fail_dice_lt_0.70": int(mean_fg[i] < 0.70),
            "fail_edema_lt_0.70": int(edema_d[i] < 0.70),
            "fail_lowest_20pct": int(fail_lowest_oof[i]),
        }
        for method in dice_pred_store:
            row[f"pred_dice_{method}"] = dice_pred_store[method][i]
        for method, fdef in score_store:
            row[f"score_{method}__{fdef}"] = score_store[(method, fdef)][i]
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    global_cut = float(np.quantile(mean_fg, 0.20))
    pred_df["fail_lowest_20pct_global"] = (mean_fg < global_cut).astype(int)

    fold_df = pd.DataFrame(fold_rows)
    layer_df = pd.DataFrame(best_layer_rows)
    coef_df = pd.DataFrame(coef_rows)
    return feat_df, pred_df, fold_df, layer_df, coef_df, score_store, dice_pred_store


def evaluate_all(
    pred_df: pd.DataFrame,
    score_store: dict[tuple[str, str], np.ndarray],
    dice_pred_store: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_fg = pred_df["label_mean_fg_dice"].to_numpy(dtype=float)
    edema_d = pred_df["label_edema_dice"].to_numpy(dtype=float)
    boundary = pred_df["label_boundary_error"].to_numpy(dtype=float)
    n = len(pred_df)

    fail_y = {
        "dice_lt_0.80": pred_df["fail_dice_lt_0.80"].to_numpy(dtype=int),
        "dice_lt_0.70": pred_df["fail_dice_lt_0.70"].to_numpy(dtype=int),
        "edema_lt_0.70": pred_df["fail_edema_lt_0.70"].to_numpy(dtype=int),
        "lowest_20pct": pred_df["fail_lowest_20pct"].to_numpy(dtype=int),
    }

    # Continuous regression metrics.
    reg_rows = []
    targets = {
        "mean_fg_dice": mean_fg,
        "edema_dice": edema_d,
        "boundary_error": boundary,
    }
    for method, preds in dice_pred_store.items():
        if method.startswith("edema_"):
            y = edema_d
            tname = "edema_dice"
        elif method.startswith("boundary_"):
            y = boundary
            tname = "boundary_error"
        else:
            y = mean_fg
            tname = "mean_fg_dice"
        if not np.isfinite(preds).all():
            continue
        m = regression_metrics(y, preds)
        mae_ci = bootstrap_ci(
            lambda idx, yy=y, pp=preds: mean_absolute_error(yy[idx], pp[idx]),
            n,
            n_bootstrap,
            seed,
        )
        sp_ci = bootstrap_ci(
            lambda idx, yy=y, pp=preds: float(spearmanr(yy[idx], pp[idx]).correlation),
            n,
            n_bootstrap,
            seed + 1,
        )
        reg_rows.append(
            {
                "method": method,
                "target": tname,
                **m,
                "mae_ci_low": mae_ci["ci_low"],
                "mae_ci_high": mae_ci["ci_high"],
                "spearman_ci_low": sp_ci["ci_low"],
                "spearman_ci_high": sp_ci["ci_high"],
            }
        )
    reg_df = pd.DataFrame(reg_rows)

    # Failure detection + risk coverage.
    fail_rows = []
    rc_rows = []
    boot_rows = []
    methods = sorted({m for m, _ in score_store.keys()})

    for fdef, y in fail_y.items():
        # Critical baselines: output / quality-probe families (not pooled_repr).
        # Confidence omitted: GT-leaked. pooled_repr is reported separately.
        baseline_auprc = -1.0
        baseline_method = "morphology"
        for m in ["morphology", "dice_probe", "rep_anatomy"]:
            if (m, fdef) not in score_store:
                continue
            s = score_store[(m, fdef)]
            try:
                a = float(average_precision_score(y, s))
            except Exception:
                a = -1.0
            if a > baseline_auprc:
                baseline_auprc, baseline_method = a, m

        for method in methods:
            scores = score_store[(method, fdef)]
            cm = classification_metrics(y, scores)
            auprc_ci = bootstrap_ci(
                lambda idx, yy=y, ss=scores: average_precision_score(yy[idx], ss[idx]),
                n,
                n_bootstrap,
                seed + 2,
            )
            auroc_ci = bootstrap_ci(
                lambda idx, yy=y, ss=scores: roc_auc_score(yy[idx], ss[idx]),
                n,
                n_bootstrap,
                seed + 3,
            )
            fail_rows.append(
                {
                    "method": method,
                    "failure_def": fdef,
                    **cm,
                    "auprc_ci_low": auprc_ci["ci_low"],
                    "auprc_ci_high": auprc_ci["ci_high"],
                    "auroc_ci_low": auroc_ci["ci_low"],
                    "auroc_ci_high": auroc_ci["ci_high"],
                    "strongest_baseline": baseline_method,
                    "baseline_auprc": baseline_auprc,
                }
            )

            # Risk = failure score; quality = mean_fg dice
            rc = risk_coverage_curve(scores, mean_fg)
            cap = failure_capture_at_budget(scores, y.astype(bool))
            aurc_ci = bootstrap_ci(
                lambda idx, ss=scores, qq=mean_fg: risk_coverage_curve(ss[idx], qq[idx])[
                    "aurc"
                ],
                n,
                n_bootstrap,
                seed + 4,
            )
            cap20_ci = bootstrap_ci(
                lambda idx, ss=scores, yy=y: failure_capture_at_budget(
                    ss[idx], yy[idx].astype(bool)
                )["capture_at_20"],
                n,
                n_bootstrap,
                seed + 5,
            )
            rc_rows.append(
                {
                    "method": method,
                    "failure_def": fdef,
                    **rc,
                    **cap,
                    "aurc_ci_low": aurc_ci["ci_low"],
                    "aurc_ci_high": aurc_ci["ci_high"],
                    "capture20_ci_low": cap20_ci["ci_low"],
                    "capture20_ci_high": cap20_ci["ci_high"],
                    "cohort_mean_dice": float(np.mean(mean_fg)),
                }
            )

            # Paired bootstrap vs strongest baseline
            base_scores = score_store[(baseline_method, fdef)]
            diff = bootstrap_ci(
                lambda idx, yy=y, sa=scores, sb=base_scores: average_precision_score(
                    yy[idx], sa[idx]
                )
                - average_precision_score(yy[idx], sb[idx]),
                n,
                n_bootstrap,
                seed + 6,
            )
            boot_rows.append(
                {
                    "method": method,
                    "failure_def": fdef,
                    "baseline": baseline_method,
                    "metric": "auprc_diff",
                    "diff": diff["point"],
                    "ci_low": diff["ci_low"],
                    "ci_high": diff["ci_high"],
                }
            )

    return (
        reg_df,
        pd.DataFrame(fail_rows),
        pd.DataFrame(rc_rows),
        pd.DataFrame(boot_rows),
    )


def make_figures(
    pred_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    fail_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    score_store: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    mean_fg = pred_df["label_mean_fg_dice"].to_numpy(dtype=float)
    y_low = pred_df["fail_lowest_20pct"].to_numpy(dtype=int)

    # Risk-coverage curves
    plt.figure(figsize=(7, 5))
    for method, color in [
        ("morphology", "#F58518"),
        ("dice_probe", "#54A24B"),
        ("inconsistency", "#E45756"),
        ("combined", "#B279A2"),
        ("pooled_repr", "#4C78A8"),
    ]:
        if (method, "lowest_20pct") not in score_store:
            continue
        scores = score_store[(method, "lowest_20pct")]
        order = np.argsort(scores)
        qs = mean_fg[order]
        cov = np.linspace(0.05, 1.0, 20)
        risks = [1 - float(np.mean(qs[: max(1, int(round(c * len(qs))))])) for c in cov]
        plt.plot(cov, risks, label=method, color=color)
    plt.xlabel("Coverage (retained)")
    plt.ylabel("Risk (1 − mean Dice)")
    plt.title("Risk–coverage (lowest-20% ranking)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "risk_coverage_curves.png")
    plt.close()

    # PR curves for failure detection
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(7, 5))
    for method, color in [
        ("morphology", "#F58518"),
        ("dice_probe", "#54A24B"),
        ("inconsistency", "#E45756"),
        ("combined", "#B279A2"),
        ("pooled_repr", "#4C78A8"),
    ]:
        if (method, "lowest_20pct") not in score_store:
            continue
        scores = score_store[(method, "lowest_20pct")]
        p, r, _ = precision_recall_curve(y_low, scores)
        plt.plot(r, p, label=method, color=color)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Failure detection PR (lowest 20%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_detection_pr_curves.png")
    plt.close()

    # Predicted vs actual Dice
    plt.figure(figsize=(6, 6))
    pred = pred_df["pred_dice_combined"].to_numpy(dtype=float)
    plt.scatter(mean_fg, pred, s=12, alpha=0.6, c="#4C78A8")
    lims = [0, 1]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Actual mean foreground Dice")
    plt.ylabel("Predicted Dice (combined)")
    plt.title("Predicted vs actual Dice")
    plt.tight_layout()
    plt.savefig(fig_dir / "predicted_vs_actual_dice.png")
    plt.close()

    # Inconsistency by failure status
    gap_cols = [c for c in feat_df.columns if c.startswith("absolute_gap_")]
    if gap_cols:
        plt.figure(figsize=(8, 4))
        failed = y_low.astype(bool)
        data = [feat_df.loc[~failed, gap_cols].mean(axis=1), feat_df.loc[failed, gap_cols].mean(axis=1)]
        plt.boxplot(data, tick_labels=["non-failure", "lowest-20%"])
        plt.ylabel("Mean absolute gap")
        plt.title("Inconsistency by failure status")
        plt.tight_layout()
        plt.savefig(fig_dir / "inconsistency_by_failure_status.png")
        plt.close()

    # Failure capture at review budget
    sub = rc_df[rc_df["failure_def"] == "lowest_20pct"]
    plt.figure(figsize=(7, 5))
    budgets = [10, 20, 30]
    methods = ["morphology", "dice_probe", "inconsistency", "combined", "pooled_repr"]
    x = np.arange(len(budgets))
    width = 0.15
    for i, method in enumerate(methods):
        row = sub[sub["method"] == method]
        if row.empty:
            continue
        vals = [float(row.iloc[0][f"capture_at_{b}"]) for b in budgets]
        plt.bar(x + i * width, vals, width, label=method)
    plt.xticks(x + 2 * width, [f"{b}%" for b in budgets])
    plt.ylabel("Fraction of severe failures captured")
    plt.xlabel("Review budget")
    plt.title("Failure capture vs review budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_capture_at_review_budget.png")
    plt.close()

    # Model comparison AUPRC
    subf = fail_df[fail_df["failure_def"] == "lowest_20pct"]
    plt.figure(figsize=(8, 4))
    order = [
        "morphology",
        "dice_probe",
        "rep_anatomy",
        "inconsistency",
        "combined",
        "pooled_repr",
    ]
    vals = []
    labels = []
    for m in order:
        row = subf[subf["method"] == m]
        if row.empty:
            continue
        labels.append(m)
        vals.append(float(row.iloc[0]["auprc"]))
    plt.barh(labels, vals, color="#4C78A8")
    plt.xlabel("AUPRC")
    plt.title("Model comparison (lowest 20% failures)")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison.png")
    plt.close()


def classify_feasibility(
    fail_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    fdef = "lowest_20pct"
    sub = fail_df[fail_df["failure_def"] == fdef]
    rc = rc_df[rc_df["failure_def"] == fdef]
    combined = sub[sub["method"] == "combined"].iloc[0]
    base_name = combined["strongest_baseline"]
    baseline = sub[sub["method"] == base_name].iloc[0]
    rc_c = rc[rc["method"] == "combined"].iloc[0]
    rc_b = rc[rc["method"] == base_name].iloc[0]

    criteria = {}
    # 1. AUPRC +0.05 over strongest baseline
    criteria["auprc_plus_0.05"] = bool(
        float(combined["auprc"]) >= float(baseline["auprc"]) + 0.05
    )
    # 2. AURC improves ≥10% relative (lower is better)
    aurc_c, aurc_b = float(rc_c["aurc"]), float(rc_b["aurc"])
    criteria["aurc_rel_10pct"] = bool(aurc_c <= aurc_b * 0.90) if aurc_b > 0 else False
    # 3. capture ≥70% at 20% review
    criteria["capture20_ge_70"] = bool(float(rc_c["capture_at_20"]) >= 0.70)
    # 4. retained 80% mean Dice ≥ cohort + 0.02
    criteria["retained80_plus_0.02"] = bool(
        float(rc_c["mean_dice_coverage_80"])
        >= float(rc_c["cohort_mean_dice"]) + 0.02
    )
    # 5. Dice MAE improves ≥10% vs dice probe
    reg_c = reg_df[(reg_df["method"] == "combined") & (reg_df["target"] == "mean_fg_dice")]
    reg_d = reg_df[(reg_df["method"] == "dice_probe") & (reg_df["target"] == "mean_fg_dice")]
    if not reg_c.empty and not reg_d.empty:
        mae_c, mae_d = float(reg_c.iloc[0]["mae"]), float(reg_d.iloc[0]["mae"])
        criteria["mae_rel_10pct_vs_dice_probe"] = bool(mae_c <= mae_d * 0.90)
    else:
        criteria["mae_rel_10pct_vs_dice_probe"] = False

    n_pass = sum(1 for k, v in criteria.items() if v)
    beats_strongest = float(combined["auprc"]) > float(baseline["auprc"])
    criteria["beats_strongest_baseline"] = bool(beats_strongest)
    morph_auprc = float(sub[sub["method"] == "morphology"].iloc[0]["auprc"])
    if n_pass >= 2 and beats_strongest:
        verdict = "PROMISING"
    elif n_pass >= 1 or beats_strongest or float(combined["auprc"]) > morph_auprc:
        verdict = "MIXED"
    else:
        inc = sub[sub["method"] == "inconsistency"].iloc[0]
        morph = sub[sub["method"] == "morphology"].iloc[0]
        if float(inc["auprc"]) <= float(morph["auprc"]):
            verdict = "NOT PROMISING"
        else:
            verdict = "MIXED"
    return verdict, criteria


def write_feasibility_report(
    out_dir: Path,
    verdict: str,
    criteria: dict[str, Any],
    fail_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    layer_df: pd.DataFrame,
) -> None:
    fdef = "lowest_20pct"
    sub = fail_df[fail_df["failure_def"] == fdef]
    rc = rc_df[rc_df["failure_def"] == fdef]

    def _row(df, method):
        return df[df["method"] == method].iloc[0]

    lines = [
        "# Representation–Output Consistency Score — Feasibility Report",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Feasibility gates (lowest-quality 20% task):",
    ]
    for k, v in criteria.items():
        lines.append(f"- `{k}`: {'PASS' if v else 'fail'}")
    lines += ["", "## Answers", ""]

    comb = _row(sub, "combined")
    morph = _row(sub, "morphology")
    dicep = _row(sub, "dice_probe")
    inc = _row(sub, "inconsistency")
    base = _row(sub, comb["strongest_baseline"])
    rc_c = _row(rc, "combined")
    reg_mean = reg_df[reg_df["target"] == "mean_fg_dice"]
    comb_mae = float(_row(reg_mean, "combined")["mae"])

    lines += [
        "0. **Confidence baseline:** unavailable without GT leakage. Cached "
        "`mean_entropy_*` / `overlap_top_*` / `confident_false_negative_fraction` "
        "columns use GT error masks (`analyze.py`). Entropy maps are missing on disk. "
        "Combined model uses inconsistency + morphology only.",
        f"1. **Does inconsistency predict quality?** "
        f"Inconsistency-only AUPRC={inc['auprc']:.3f} "
        f"(AUROC={inc['auroc']:.3f}) for lowest-20% failures vs morphology "
        f"{morph['auprc']:.3f}. Combined continuous MAE={comb_mae:.4f}.",
        f"2. **Improve over ordinary confidence?** "
        f"Not evaluated — inference-time confidence features were not available "
        f"without leakage.",
        f"3. **Improve over morphology?** "
        f"Combined {comb['auprc']:.3f} vs morphology {morph['auprc']:.3f} "
        f"(Δ={float(comb['auprc'])-float(morph['auprc']):+.3f}).",
        f"4. **Improve over direct Dice probe?** "
        f"Combined {comb['auprc']:.3f} vs dice_probe {dicep['auprc']:.3f} "
        f"(Δ={float(comb['auprc'])-float(dicep['auprc']):+.3f}). "
        f"Strongest available baseline was `{comb['strongest_baseline']}` "
        f"(AUPRC={base['auprc']:.3f}).",
    ]

    # Most informative gaps from mean |coef|
    if not coef_df.empty:
        top = (
            coef_df.groupby("feature")["coefficient"]
            .apply(lambda s: float(np.mean(np.abs(s))))
            .sort_values(ascending=False)
            .head(8)
        )
        gap_top = [f"{k} ({v:.3f})" for k, v in top.items() if "gap_" in k][:5]
        signed = [f for f in top.index if f.startswith("signed_gap_")]
        abs_ = [f for f in top.index if f.startswith("absolute_gap_") or f.startswith("standardized_gap_")]
        lines.append(
            "5. **Most informative discrepancies (mean |coef|):** "
            + (", ".join(gap_top) if gap_top else "see feature_coefficients.csv")
        )
        lines.append(
            "6. **Signed vs absolute:** "
            + (
                "signed gaps appear among top coefficients"
                if signed and (not abs_ or top[signed[0]] >= top.get(abs_[0], -1))
                else "absolute/standardized gaps dominate top coefficients"
                if abs_
                else "mixed; inspect feature_coefficients.csv"
            )
        )
    else:
        lines.append("5. **Most informative discrepancies:** unavailable.")
        lines.append("6. **Signed vs absolute:** unavailable.")

    lines += [
        f"7. **At 20% review budget:** combined captures "
        f"{100*float(rc_c['capture_at_20']):.1f}% of lowest-20% failures "
        f"(CI {100*float(rc_c['capture20_ci_low']):.1f}–"
        f"{100*float(rc_c['capture20_ci_high']):.1f}%).",
        f"8. **Selective risk:** AURC={float(rc_c['aurc']):.4f}; "
        f"mean Dice at 80% coverage={float(rc_c['mean_dice_coverage_80']):.3f} "
        f"(cohort mean={float(rc_c['cohort_mean_dice']):.3f}).",
    ]

    # Stability across folds: use layer selection variance as proxy + note
    layer_stability = (
        layer_df[layer_df["target"].isin(["log_wt_volume", "gt_edema_frac", "dice"])]
        .groupby("target")["best_layer"]
        .agg(lambda s: s.value_counts().iloc[0] / len(s))
        .to_dict()
        if not layer_df.empty
        else {}
    )
    lines.append(
        "9. **Stability:** best-layer selection consistency (top layer fraction across outer folds): "
        + ", ".join(f"{k}={v:.2f}" for k, v in layer_stability.items())
    )
    lines += [
        f"10. **Result: {verdict}**",
        "",
        "Caveats:",
        "- Prior runs that used GT-linked entropy/error columns as “confidence” "
        "features are invalid and should be ignored.",
        "- `pooled_repr` is a separate high-dimensional baseline and may outperform "
        "the proposed combined model; it is not the critical confidence/output/"
        "quality-probe comparator.",
        "- Paired bootstrap CIs for AUPRC differences can include zero; treat "
        "point-estimate gains cautiously.",
        "",
        "This is an internal nested-CV feasibility study on 375 validation cases only. "
        "No clinical benefit, external generalization, or novelty is claimed.",
        "",
        "## Method AUPRC summary (lowest 20%)",
        "",
        "| Method | AUPRC | AUROC | Capture@20% |",
        "|---|---:|---:|---:|",
    ]
    methods = [
        "morphology",
        "dice_probe",
        "rep_anatomy",
        "inconsistency",
        "combined",
        "pooled_repr",
    ]
    for m in methods:
        fr = _row(sub, m)
        rr = _row(rc, m)
        lines.append(
            f"| {m} | {fr['auprc']:.3f} | {fr['auroc']:.3f} | {rr['capture_at_20']:.3f} |"
        )
    (out_dir / "feasibility_report.md").write_text("\n".join(lines) + "\n")


def run_from_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    out_dir = ensure_dir(Path(config["paths"]["output_dir"]))
    (out_dir / "artifact_inventory.md").write_text(ARTIFACT_INVENTORY_MD)
    case_df, layer_matrices, _ = build_case_table(
        Path(config["paths"]["layer_index"]),
        Path(config["paths"]["failure_table"]),
    )
    (
        feat_df,
        pred_df,
        fold_df,
        layer_df,
        coef_df,
        score_store,
        dice_pred_store,
    ) = nested_cv_predictions(case_df, layer_matrices, config)

    # Verify one prediction per case
    assert fold_df[fold_df["split"] == "test"].groupby("case_id").size().max() == 1
    assert len(pred_df) == len(case_df)

    reg_df, fail_df, rc_df, boot_df = evaluate_all(
        pred_df,
        score_store,
        dice_pred_store,
        n_bootstrap=int(config["n_bootstrap"]),
        seed=int(config["seed"]),
    )
    make_figures(pred_df, feat_df, fail_df, rc_df, score_store, out_dir)
    verdict, criteria = classify_feasibility(fail_df, rc_df, reg_df)
    write_feasibility_report(
        out_dir, verdict, criteria, fail_df, rc_df, reg_df, coef_df, layer_df
    )

    feat_df.to_csv(out_dir / "case_level_features.csv", index=False)
    pred_df.to_csv(out_dir / "outer_fold_predictions.csv", index=False)
    fold_df.to_csv(out_dir / "fold_assignments.csv", index=False)
    layer_df.to_csv(out_dir / "best_layer_by_fold.csv", index=False)
    reg_df.to_csv(out_dir / "regression_metrics.csv", index=False)
    fail_df.to_csv(out_dir / "failure_detection_metrics.csv", index=False)
    boot_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)
    rc_df.to_csv(out_dir / "risk_coverage_metrics.csv", index=False)
    coef_df.to_csv(out_dir / "feature_coefficients.csv", index=False)

    return {
        "verdict": verdict,
        "criteria": criteria,
        "n_cases": len(case_df),
        "output_dir": str(out_dir),
    }


def make_fold_assignments(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)))
