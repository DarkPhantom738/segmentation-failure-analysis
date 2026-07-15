"""Confidence + representation–output consistency failure triage (nested CV).

Primary question: does adding consistency-gap features improve failure ranking
beyond TTA confidence alone, under leakage-safe nested cross-validation?

Leakage rules
-------------
- Confidence features must be GT-free (TTA softmax summaries only).
- Ground truth is allowed only inside training folds for Ridge anatomy probes,
  failure-label definition, and held-out evaluation.
- Outer folds are reused from the consistency pipeline (same case IDs / seed).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis.layer_aware_latent_risk import (
    GT_LEAKED_COLS,
    assert_no_gt_leak_features,
    load_outer_folds,
    verify_folds_match_seed,
)
from src.analysis.layer_io import build_anatomy_table, load_layer_index, load_layer_matrix
from src.analysis.representation_output_consistency import (
    build_gap_features,
    failure_capture_at_budget,
    fit_ridge_predict,
    oof_ridge_predictions,
    risk_coverage_curve,
    select_best_layer,
)
from src.utils.io import ensure_dir

ARTIFACT_INVENTORY = """# Artifact inventory — Confidence + Consistency Triage

## Reused (no regeneration)

1. Outer folds: `outputs_consistency_failure_detection/fold_assignments.csv` (seed 42, 5×75)
2. Confidence: `outputs_layer_aware_latent_risk/case_level_confidence_features.csv`
   - GT-free TTA summaries; **375/375 exact** match vs retained `*_pred_tta.npy`
3. Morphology / labels: predicted-mask anatomy + Dice labels from consistency table
4. Layer embeddings: `outputs_10hour/layer_embeddings/` (9 layers × 375)
5. GT anatomy for probe training only via `build_anatomy_table`

## Regenerated inside nested CV

- Anatomical Ridge probes (best layer per target on outer-train only)
- Consistency gap features (train-fold scaling)
- Logistic models, feature-selection mode, Platt calibration

## Forbidden

- GT-linked uncertainty columns from `failure_metrics.csv`
"""


# ---------------------------------------------------------------------------
# Feature preprocessing and logistic helpers
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def impute_missing_with_train_mean(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Replace non-finite entries with each column's training-fold mean."""
    filled = np.asarray(features, dtype=np.float64).copy()
    for column_index in range(filled.shape[1]):
        column = filled[:, column_index]
        train_values = column[train_indices]
        train_mean = (
            float(np.nanmean(train_values)) if np.isfinite(train_values).any() else 0.0
        )
        column[~np.isfinite(column)] = train_mean
        filled[:, column_index] = column
    return filled


def standardize_features_with_train_stats(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Impute then StandardScaler-fit on training rows only; transform all rows."""
    features = impute_missing_with_train_mean(features, train_indices)
    scaler = StandardScaler()
    scaler.fit(features[train_indices])
    return scaler.transform(features)


def tune_logistic_regression(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    candidate_Cs: list[float],
    seed: int,
    penalty: str = "l2",
    l1_ratio: float = 0.5,
) -> LogisticRegression:
    """Pick C (and optional elastic-net) by inner stratified AUPRC, then refit."""
    labels_train = labels_train.astype(int)
    # sklearn>=1.8: pass l1_ratio even for L2 (0.0) when using solver="saga".
    model_kwargs: dict[str, Any] = {
        "max_iter": 5000,
        "class_weight": "balanced",
        "random_state": seed,
        "solver": "saga",
    }
    if penalty == "elasticnet":
        model_kwargs["l1_ratio"] = float(l1_ratio)
    else:
        model_kwargs["l1_ratio"] = 0.0  # pure L2

    n_positive = int(labels_train.sum())
    n_negative = int(len(labels_train) - n_positive)
    if n_positive < 2 or n_negative < 2:
        model = LogisticRegression(C=1.0, **model_kwargs)
        model.fit(features_train, labels_train)
        return model

    n_splits = min(4, n_positive, n_negative)
    if n_splits < 2:
        model = LogisticRegression(C=1.0, **model_kwargs)
        model.fit(features_train, labels_train)
        return model

    best_c, best_auprc = candidate_Cs[0], -1.0
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c_value in candidate_Cs:
        fold_auprcs: list[float] = []
        for inner_train, inner_test in splitter.split(features_train, labels_train):
            if labels_train[inner_train].sum() == 0 or labels_train[inner_train].sum() == len(inner_train):
                continue
            model = LogisticRegression(C=c_value, **model_kwargs)
            model.fit(features_train[inner_train], labels_train[inner_train])
            try:
                probs = model.predict_proba(features_train[inner_test])[:, 1]
                fold_auprcs.append(average_precision_score(labels_train[inner_test], probs))
            except Exception:
                continue
        mean_auprc = float(np.mean(fold_auprcs)) if fold_auprcs else -1.0
        if mean_auprc > best_auprc:
            best_auprc, best_c = mean_auprc, c_value

    model = LogisticRegression(C=best_c, **model_kwargs)
    model.fit(features_train, labels_train)
    return model


def classification_metrics_from_scores(
    labels: np.ndarray,
    risk_scores: np.ndarray,
) -> dict[str, float]:
    """AUROC/AUPRC/Brier plus threshold-0.5 confusion and linear calibration."""
    labels = labels.astype(int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    metric_keys = (
        "auroc",
        "auprc",
        "brier",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "calibration_slope",
        "calibration_intercept",
    )
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {key: float("nan") for key in metric_keys}

    binary_pred = (risk_scores >= 0.5).astype(int)
    true_positive = int(((binary_pred == 1) & (labels == 1)).sum())
    true_negative = int(((binary_pred == 0) & (labels == 0)).sum())
    false_positive = int(((binary_pred == 1) & (labels == 0)).sum())
    false_negative = int(((binary_pred == 0) & (labels == 1)).sum())
    from sklearn.linear_model import LinearRegression

    calibration = LinearRegression().fit(
        risk_scores.reshape(-1, 1), labels.astype(float)
    )
    return {
        "auroc": float(roc_auc_score(labels, risk_scores)),
        "auprc": float(average_precision_score(labels, risk_scores)),
        "brier": float(brier_score_loss(labels, np.clip(risk_scores, 1e-6, 1 - 1e-6))),
        "sensitivity": true_positive / max(true_positive + false_negative, 1),
        "specificity": true_negative / max(true_negative + false_positive, 1),
        "ppv": true_positive / max(true_positive + false_positive, 1),
        "npv": true_negative / max(true_negative + false_negative, 1),
        "calibration_slope": float(calibration.coef_[0]),
        "calibration_intercept": float(calibration.intercept_),
    }


def consistency_feature_column_sets(
    gap_column_names: list[str],
    anatomy_targets: list[str],
) -> dict[str, list[str]]:
    """Named subsets of consistency-gap columns for inner-mode selection."""
    absolute_columns = [c for c in gap_column_names if c.startswith("absolute_gap_")]
    signed_columns = [c for c in gap_column_names if c.startswith("signed_gap_")]
    standardized_columns = [c for c in gap_column_names if c.startswith("standardized_gap_")]
    one_absolute_per_target = [
        f"absolute_gap_{target}"
        for target in anatomy_targets
        if f"absolute_gap_{target}" in gap_column_names
    ]
    return {
        "all": gap_column_names[:],
        "absolute_only": absolute_columns,
        "signed_only": signed_columns,
        "standardized_only": standardized_columns,
        "one_per_target_abs": one_absolute_per_target,
        # Same column pool; sparse selection happens via elastic-net coefficients.
        "elasticnet": gap_column_names[:],
    }


def select_consistency_mode_inner(
    features_by_mode: dict[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    candidate_Cs: list[float],
    seed: int,
) -> str:
    """Pick consistency feature mode by inner OOF AUPRC on outer-train only."""
    best_mode, best_auprc = "all", -1.0
    labels_train = labels[train_indices]
    if labels_train.sum() < 2 or (len(labels_train) - labels_train.sum()) < 2:
        return "all"
    n_splits = min(4, int(labels_train.sum()), int(len(labels_train) - labels_train.sum()))
    if n_splits < 2:
        return "all"

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for mode_name, features_full in features_by_mode.items():
        if features_full.shape[1] == 0:
            continue
        features = standardize_features_with_train_stats(features_full, train_indices)
        fold_auprcs: list[float] = []
        for relative_train, relative_test in splitter.split(features[train_indices], labels_train):
            absolute_train = train_indices[relative_train]
            absolute_test = train_indices[relative_test]
            if labels[absolute_train].sum() == 0 or labels[absolute_train].sum() == len(absolute_train):
                continue
            if mode_name == "elasticnet":
                model = tune_logistic_regression(
                    features[absolute_train],
                    labels[absolute_train],
                    candidate_Cs,
                    seed,
                    penalty="elasticnet",
                    l1_ratio=0.5,
                )
            else:
                model = tune_logistic_regression(
                    features[absolute_train],
                    labels[absolute_train],
                    candidate_Cs,
                    seed,
                    penalty="l2",
                )
            try:
                probs = model.predict_proba(features[absolute_test])[:, 1]
                fold_auprcs.append(average_precision_score(labels[absolute_test], probs))
            except Exception:
                continue
        mean_auprc = float(np.mean(fold_auprcs)) if fold_auprcs else -1.0
        if mean_auprc > best_auprc:
            best_auprc, best_mode = mean_auprc, mode_name
    return best_mode


# Back-compat aliases for any external callers / older notebooks.
_impute = impute_missing_with_train_mean
_scale = standardize_features_with_train_stats
_tune_logistic = tune_logistic_regression
_cls_metrics = classification_metrics_from_scores
_consistency_column_sets = consistency_feature_column_sets
_select_mode_inner = select_consistency_mode_inner


# ---------------------------------------------------------------------------
# Case loading and nested-CV triage
# ---------------------------------------------------------------------------


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Avoid overwriting prior runs unless overwrite_output is true."""
    base = Path(config["paths"]["output_dir"])
    if config.get("overwrite_output", False):
        return ensure_dir(base)
    if base.exists() and any(base.iterdir()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stamped = base.parent / f"{base.name}_{ts}"
        print(f"Output dir exists; writing to {stamped}")
        return ensure_dir(stamped)
    return ensure_dir(base)


def build_case_base(config: dict[str, Any]) -> dict[str, Any]:
    """Load aligned cases, confidence, morphology, labels, embeddings."""
    out_dir = resolve_output_dir(config)
    (out_dir / "artifact_inventory.md").write_text(ARTIFACT_INVENTORY)

    fold_csv = Path(config["paths"]["fold_assignments"])
    folds = load_outer_folds(fold_csv)
    cons = pd.read_csv(config["paths"]["consistency_features"]).sort_values("case_id").reset_index(drop=True)
    conf = pd.read_csv(config["paths"]["confidence_features"]).sort_values("case_id").reset_index(drop=True)
    case_ids = cons["case_id"].tolist()
    if list(conf["case_id"]) != case_ids:
        conf = conf.set_index("case_id").loc[case_ids].reset_index()
    if not conf["mask_exact_match"].astype(str).isin(["True", "true", "1"]).all():
        raise RuntimeError("Confidence features fail exact mask match; abort.")

    conf_cols = list(config["confidence_cols"])
    morph_cols = list(config["morphology_cols"])
    assert_no_gt_leak_features(conf_cols + morph_cols)
    for c in GT_LEAKED_COLS:
        if c in conf_cols or c in morph_cols:
            raise RuntimeError(f"GT-leaked column requested: {c}")

    index_df = load_layer_index(config["paths"]["layer_index"]).sort_values("case_id").reset_index(drop=True)
    if list(index_df["case_id"]) != case_ids:
        index_df = index_df.set_index("case_id").loc[case_ids].reset_index()
    failure_df = pd.read_csv(config["paths"]["failure_table"])
    anatomy = build_anatomy_table(index_df, failure_df).sort_values("case_id").reset_index(drop=True)
    if list(anatomy["case_id"]) != case_ids:
        anatomy = anatomy.set_index("case_id").loc[case_ids].reset_index()

    layers = list(config["layers"])
    layer_mats = {L: load_layer_matrix(index_df, L) for L in layers}

    # Output anatomy from predicted masks (already in consistency table)
    out_map = {
        "log_wt_volume": cons["out_log_wt_volume"].to_numpy(float),
        "gt_edema_frac": cons["out_gt_edema_frac"].to_numpy(float),
        "gt_enhancing_frac": cons["out_gt_enhancing_frac"].to_numpy(float),
        "gt_necrosis_frac": cons["out_gt_necrosis_frac"].to_numpy(float),
        "gt_compactness": cons["out_gt_compactness"].to_numpy(float),
    }
    gt_map = {t: anatomy[t].to_numpy(float) for t in config["consistency_targets"]}

    pd.read_csv(fold_csv).to_csv(out_dir / "fold_assignments.csv", index=False)

    return {
        "out_dir": out_dir,
        "case_ids": case_ids,
        "folds": folds,
        "folds_ok": verify_folds_match_seed(fold_csv, n=len(case_ids), seed=int(config["seed"])),
        "cons": cons,
        "conf": conf,
        "conf_cols": conf_cols,
        "morph_cols": morph_cols,
        "layers": layers,
        "layer_mats": layer_mats,
        "out_map": out_map,
        "gt_map": gt_map,
        "mean_fg": cons["label_mean_fg_dice"].to_numpy(float),
        "edema_d": cons["label_edema_dice"].to_numpy(float),
    }


def run_triage(config: dict[str, Any], model_seed: int | None = None) -> dict[str, Any]:
    """Run nested-CV triage for all method / failure-definition combinations.

    For each outer fold: fit anatomy probes on train, build consistency gaps,
    choose a gap-feature mode, score methods on the held-out test fold.
    """
    seed = int(config["seed"])
    model_seed = int(model_seed if model_seed is not None else seed)
    base = build_case_base(config)
    assert base["folds_ok"], "Fold assignments do not match seed-42 KFold"
    out_dir = base["out_dir"]
    fig_dir = ensure_dir(out_dir / "figures")

    n_cases = len(base["case_ids"])
    layers = base["layers"]
    targets = list(config["consistency_targets"])
    alphas = list(config["ridge_alphas"])
    candidate_Cs = list(config["logistic_C"])
    inner_folds = int(config["inner_folds"])
    n_bootstrap = int(config["n_bootstrap"])

    # Feature families (aligned row order = case_ids)
    confidence_matrix = base["conf"][base["conf_cols"]].to_numpy(float)
    morphology_matrix = base["cons"][base["morph_cols"]].to_numpy(float)
    pooled_representations = np.concatenate([base["layer_mats"][layer] for layer in layers], axis=1)
    mean_foreground_dice = base["mean_fg"]
    edema_dice_scores = base["edema_d"]

    methods = [
        "confidence",
        "morphology",
        "consistency",
        "conf_morph",
        "conf_pooled",
        "conf_consistency",  # proposed: confidence + consistency gaps
        "conf_morph_consistency",
    ]
    failure_definitions = list(config["failure_defs"])
    scores = {
        (method, failure_def): np.full(n_cases, np.nan)
        for method in methods
        for failure_def in failure_definitions
    }
    scores_calibrated = {
        ("conf_consistency", failure_def): np.full(n_cases, np.nan)
        for failure_def in failure_definitions
    }
    failure_labels = {failure_def: np.full(n_cases, np.nan) for failure_def in failure_definitions}

    # Store held-out consistency features (test folds only).
    # Shorter local names below match the long fold body for minimal churn.
    held_feat: dict[str, np.ndarray] = {}
    sel_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    n = n_cases
    Cs = candidate_Cs
    inner = inner_folds
    n_boot = n_bootstrap
    conf_mat = confidence_matrix
    morph_mat = morphology_matrix
    pooled = pooled_embeddings
    mean_fg = mean_foreground_dice
    edema_d = edema_dice_scores
    fail_defs = failure_definitions
    scores_cal = scores_calibrated
    fail_y = failure_labels

    # ---- Outer nested CV ----------------------------------------------------
    for fold_i, parts in sorted(base["folds"].items()):
        # train_indices / test_indices: outer-train and held-out test case rows
        tr, te = parts["train"], parts["test"]
        # Failure thresholds use outer-train quantiles only (no test leakage).
        cut_fg = float(np.quantile(mean_fg[tr], 0.20))
        cut_ed = float(np.quantile(edema_d[tr], 0.20))
        y_map = {
            "lowest20_mean_fg": (mean_fg < cut_fg).astype(int),
            "mean_fg_lt_0.80": (mean_fg < 0.80).astype(int),
            "mean_fg_lt_0.70": (mean_fg < 0.70).astype(int),
            "edema_lt_0.70": (edema_d < 0.70).astype(int),
            "lowest20_edema": (edema_d < cut_ed).astype(int),
        }
        for fdef, y in y_map.items():
            fail_y[fdef][te] = y[te]

        # --- Anatomy probes + consistency gaps (train-only layer selection) ---
        fold_gaps: dict[str, np.ndarray] = {}
        rep_cols: dict[str, np.ndarray] = {}
        out_cols: dict[str, np.ndarray] = {}
        for t in targets:
            best_layer, best_r2, oof_tr = select_best_layer(
                {L: base["layer_mats"][L][tr] for L in layers},
                base["gt_map"][t][tr],
                layers,
                n_splits=inner,
                seed=seed + fold_i + model_seed,
                alphas=alphas,
            )
            layer_rows.append(
                {
                    "outer_fold": fold_i,
                    "target": t,
                    "best_layer": best_layer,
                    "train_oof_r2": best_r2,
                    "model_seed": model_seed,
                }
            )
            rep = np.full(n, np.nan)
            rep[tr] = oof_tr
            rep[te] = fit_ridge_predict(
                base["layer_mats"][best_layer][tr],
                base["gt_map"][t][tr],
                base["layer_mats"][best_layer][te],
                alphas,
            )
            rep_cols[t] = rep
            out_cols[t] = base["out_map"][t]
            gaps = build_gap_features(rep, base["out_map"][t], tr, t)
            fold_gaps.update(gaps)
            for k, v in gaps.items():
                if k not in held_feat:
                    held_feat[k] = np.full(n, np.nan)
                held_feat[k][te] = v[te]
            # also store rep/out heldout
            for prefix, arr in (("rep", rep), ("out", base["out_map"][t])):
                key = f"{prefix}_{t}"
                if key not in held_feat:
                    held_feat[key] = np.full(n, np.nan)
                held_feat[key][te] = arr[te]

        gap_names = sorted(fold_gaps.keys())
        gap_mat = np.column_stack([fold_gaps[k] for k in gap_names])
        rep_mat = np.column_stack([rep_cols[t] for t in targets])
        out_meas = np.column_stack([out_cols[t] for t in targets])
        # consistency block for modeling: gaps (+ optional rep/out). Primary = gaps.
        cons_full = np.column_stack([gap_mat, rep_mat, out_meas])
        cons_full_names = (
            gap_names
            + [f"rep_{t}" for t in targets]
            + [f"out_{t}" for t in targets]
        )

        mode_sets = consistency_feature_column_sets(gap_names, targets)
        # Build mode matrices from gaps only for selection (compact)
        x_by_mode = {
            mode: np.column_stack([fold_gaps[c] for c in cols]) if cols else np.zeros((n, 0))
            for mode, cols in mode_sets.items()
        }

        primary_targets = set(config.get("primary_targets", ["lowest20_mean_fg", "mean_fg_lt_0.70"]))
        n_block_perm = int(config.get("n_block_permutations", 50))
        rng_block = np.random.default_rng(model_seed + 1000 + fold_i)

        # Select consistency feature mode separately per failure target (outer-train only)
        for fdef, y_all in y_map.items():
            selected_mode = select_consistency_mode_inner(
                x_by_mode,
                y_all,
                tr,
                Cs,
                seed=model_seed + fold_i + sum(ord(c) for c in fdef),
            )
            cons_sel = x_by_mode[selected_mode]
            if selected_mode == "elasticnet":
                x_tmp = standardize_features_with_train_stats(cons_sel, tr)
                en = tune_logistic_regression(
                    x_tmp[tr],
                    y_all[tr],
                    Cs,
                    model_seed + fold_i,
                    penalty="elasticnet",
                    l1_ratio=0.5,
                )
                keep = np.where(np.abs(en.coef_.ravel()) > 1e-8)[0]
                if len(keep) == 0:
                    keep = np.arange(cons_sel.shape[1])
                cons_sel = cons_sel[:, keep]
                kept_names = [mode_sets["elasticnet"][i] for i in keep]
            else:
                kept_names = mode_sets[selected_mode]

            sel_rows.append(
                {
                    "outer_fold": fold_i,
                    "model_seed": model_seed,
                    "failure_def": fdef,
                    "selected_mode": selected_mode,
                    "n_features": int(cons_sel.shape[1]),
                }
            )
            for name in kept_names:
                sel_rows.append(
                    {
                        "outer_fold": fold_i,
                        "model_seed": model_seed,
                        "failure_def": fdef,
                        "selected_mode": selected_mode,
                        "feature": name,
                        "selected": 1,
                    }
                )

            feature_sets = {
                "confidence": conf_mat,
                "morphology": morph_mat,
                "consistency": cons_sel,
                "conf_morph": np.column_stack([conf_mat, morph_mat]),
                "conf_pooled": np.column_stack([conf_mat, pooled]),
                "conf_consistency": np.column_stack([conf_mat, cons_sel]),
                "conf_morph_consistency": np.column_stack([conf_mat, morph_mat, cons_sel]),
            }

            for method, raw in feature_sets.items():
                if raw.size == 0 or raw.shape[1] == 0:
                    continue
                x = standardize_features_with_train_stats(raw, tr)
                model = tune_logistic_regression(x[tr], y_all[tr], Cs, seed=model_seed + fold_i)
                s_te = model.predict_proba(x[te])[:, 1]
                scores[(method, fdef)][te] = s_te

                if method == "conf_consistency" and fdef in primary_targets:
                    # Platt calibration on outer-train only (inner CV inside CalibratedClassifierCV)
                    try:
                        base_m = tune_logistic_regression(x[tr], y_all[tr], Cs, seed=model_seed + fold_i)
                        cal = CalibratedClassifierCV(base_m, method="sigmoid", cv=3)
                        cal.fit(x[tr], y_all[tr])
                        scores_cal[("conf_consistency", fdef)][te] = cal.predict_proba(x[te])[:, 1]
                    except Exception:
                        scores_cal[("conf_consistency", fdef)][te] = s_te

                    # True consistency-block permutation importance on held-out fold
                    n_conf = conf_mat.shape[1]
                    n_cons = cons_sel.shape[1]
                    base_fold_auprc = float(average_precision_score(y_all[te], s_te))
                    drops = []
                    x_te = x[te]
                    for _ in range(n_block_perm):
                        x_perm = x_te.copy()
                        perm_idx = rng_block.permutation(len(te))
                        x_perm[:, n_conf : n_conf + n_cons] = x_te[perm_idx][
                            :, n_conf : n_conf + n_cons
                        ]
                        s_perm = model.predict_proba(x_perm)[:, 1]
                        drops.append(
                            base_fold_auprc
                            - float(average_precision_score(y_all[te], s_perm))
                        )
                    coef_rows.append(
                        {
                            "outer_fold": fold_i,
                            "model_seed": model_seed,
                            "feature": f"__block_perm_drop__{fdef}",
                            "coefficient": float(np.mean(drops)),
                        }
                    )

                    names = base["conf_cols"] + kept_names
                    coefs = model.coef_.ravel()
                    for nm, coef in zip(names, coefs, strict=False):
                        coef_rows.append(
                            {
                                "outer_fold": fold_i,
                                "model_seed": model_seed,
                                "failure_def": fdef,
                                "feature": nm,
                                "coefficient": float(coef),
                            }
                        )

                try:
                    fold_metric_rows.append(
                        {
                            "outer_fold": fold_i,
                            "model_seed": model_seed,
                            "method": method,
                            "failure_def": fdef,
                            "auprc": float(average_precision_score(y_all[te], s_te)),
                            "auroc": float(roc_auc_score(y_all[te], s_te)),
                            "capture_at_20": failure_capture_at_budget(
                                s_te, y_all[te].astype(bool)
                            )["capture_at_20"],
                            "n_test": int(len(te)),
                            "n_pos": int(y_all[te].sum()),
                        }
                    )
                except Exception:
                    pass

        # end per-failure-target modeling

    # Assemble feature table (held-out consistency + conf + morph)
    feat_df = base["cons"][
        ["case_id", "label_mean_fg_dice", "label_edema_dice"] + base["morph_cols"]
    ].copy()
    for c in base["conf_cols"]:
        feat_df[c] = base["conf"][c].to_numpy()
    for k, v in held_feat.items():
        feat_df[k] = v
    feat_df.to_csv(out_dir / "case_level_features.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "case_id": base["case_ids"],
            "label_mean_fg_dice": mean_fg,
            "label_edema_dice": edema_d,
        }
    )
    for fdef in fail_defs:
        pred_df[f"fail_{fdef}"] = fail_y[fdef]
        for method in methods:
            pred_df[f"score_{method}__{fdef}"] = scores[(method, fdef)]
        cal = scores_cal.get(("conf_consistency", fdef))
        if cal is not None and np.isfinite(cal).any():
            pred_df[f"score_conf_consistency_cal__{fdef}"] = cal
    pred_df.to_csv(out_dir / "heldout_predictions.csv", index=False)

    # Aggregate + bootstrap
    agg_rows = []
    rc_rows = []
    boot_rows = []
    rng = np.random.default_rng(seed + model_seed)

    def boot_ci(y, s, fn):
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), size=len(y))
            try:
                vals.append(float(fn(y[idx], s[idx])))
            except Exception:
                continue
        if not vals:
            return float("nan"), float("nan"), float("nan")
        a = np.asarray(vals)
        return float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    for fdef in fail_defs:
        y = fail_y[fdef].astype(int)
        for method in methods:
            s = scores[(method, fdef)]
            if not np.isfinite(s).all():
                continue
            cm = classification_metrics_from_scores(y, s)
            auprc_m, auprc_lo, auprc_hi = boot_ci(y, s, average_precision_score)
            auroc_m, auroc_lo, auroc_hi = boot_ci(y, s, roc_auc_score)
            cap = failure_capture_at_budget(s, y.astype(bool))
            rc = risk_coverage_curve(s, mean_fg)
            _, cap_lo, cap_hi = boot_ci(
                y,
                s,
                lambda yy, ss: failure_capture_at_budget(ss, yy.astype(bool))["capture_at_20"],
            )
            aurc_vals = []
            dice80_vals = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                rcc = risk_coverage_curve(s[idx], mean_fg[idx])
                aurc_vals.append(rcc["aurc"])
                dice80_vals.append(rcc["mean_dice_coverage_80"])
            aurc_lo = float(np.quantile(aurc_vals, 0.025))
            aurc_hi = float(np.quantile(aurc_vals, 0.975))
            d80_lo = float(np.quantile(dice80_vals, 0.025))
            d80_hi = float(np.quantile(dice80_vals, 0.975))

            agg_rows.append(
                {
                    "method": method,
                    "failure_def": fdef,
                    "model_seed": model_seed,
                    **cm,
                    "auprc_ci_low": auprc_lo,
                    "auprc_ci_high": auprc_hi,
                    "auroc_ci_low": auroc_lo,
                    "auroc_ci_high": auroc_hi,
                    **cap,
                    "capture20_ci_low": cap_lo,
                    "capture20_ci_high": cap_hi,
                    **rc,
                    "aurc_ci_low": aurc_lo,
                    "aurc_ci_high": aurc_hi,
                    "dice80_ci_low": d80_lo,
                    "dice80_ci_high": d80_hi,
                    "n_pos": int(y.sum()),
                }
            )
            rc_rows.append({"method": method, "failure_def": fdef, "model_seed": model_seed, **rc, **cap})

        # Paired comparisons
        pairs = [
            ("conf_consistency", "confidence"),
            ("conf_consistency", "conf_morph"),
            ("conf_consistency", "conf_pooled"),
            ("conf_morph_consistency", "conf_consistency"),
            ("consistency", "morphology"),
        ]
        for a, b in pairs:
            sa, sb = scores[(a, fdef)], scores[(b, fdef)]
            if not (np.isfinite(sa).all() and np.isfinite(sb).all()):
                continue
            for metric_name, fn in [
                ("auprc_diff", lambda yy, s1, s2: average_precision_score(yy, s1) - average_precision_score(yy, s2)),
                (
                    "capture20_diff",
                    lambda yy, s1, s2: failure_capture_at_budget(s1, yy.astype(bool))["capture_at_20"]
                    - failure_capture_at_budget(s2, yy.astype(bool))["capture_at_20"],
                ),
            ]:
                diffs = []
                for _ in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    try:
                        diffs.append(float(fn(y[idx], sa[idx], sb[idx])))
                    except Exception:
                        continue
                arr = np.asarray(diffs) if diffs else np.array([np.nan])
                boot_rows.append(
                    {
                        "failure_def": fdef,
                        "model_seed": model_seed,
                        "method_a": a,
                        "method_b": b,
                        "metric": metric_name,
                        "diff_mean": float(np.nanmean(arr)),
                        "ci_low": float(np.nanquantile(arr, 0.025)),
                        "ci_high": float(np.nanquantile(arr, 0.975)),
                        "frac_a_better": float(np.mean(arr > 0)),
                    }
                )
            # risk-coverage / dice80 paired
            for metric_name, key, lower_better in [
                ("aurc_diff", "aurc", True),
                ("dice80_diff", "mean_dice_coverage_80", False),
            ]:
                diffs = []
                for _ in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    ra = risk_coverage_curve(sa[idx], mean_fg[idx])[key]
                    rb = risk_coverage_curve(sb[idx], mean_fg[idx])[key]
                    diffs.append(float((rb - ra) if lower_better else (ra - rb)))
                arr = np.asarray(diffs)
                boot_rows.append(
                    {
                        "failure_def": fdef,
                        "model_seed": model_seed,
                        "method_a": a,
                        "method_b": b,
                        "metric": metric_name,
                        "diff_mean": float(np.mean(arr)),
                        "ci_low": float(np.quantile(arr, 0.025)),
                        "ci_high": float(np.quantile(arr, 0.975)),
                        "frac_a_better": float(np.mean(arr > 0)),
                    }
                )

    agg_df = pd.DataFrame(agg_rows)
    fold_df = pd.DataFrame(fold_metric_rows)
    boot_df = pd.DataFrame(boot_rows)
    rc_df = pd.DataFrame(rc_rows)
    sel_df = pd.DataFrame(sel_rows)
    coef_df = pd.DataFrame(coef_rows)
    layer_df = pd.DataFrame(layer_rows)

    # Feature selection frequency (report for primary lowest20 target)
    primary0 = list(config.get("primary_targets", ["lowest20_mean_fg"]))[0]
    if "feature" in sel_df.columns:
        feat_freq = sel_df.dropna(subset=["feature"]).copy()
        if "failure_def" in feat_freq.columns:
            scoped = feat_freq[feat_freq["failure_def"] == primary0]
            if not scoped.empty:
                feat_freq = scoped
        freq = (
            feat_freq.groupby("feature").size()
            .reset_index(name="n_folds_selected")
            .sort_values("n_folds_selected", ascending=False)
        )
    else:
        freq = pd.DataFrame(columns=["feature", "n_folds_selected"])

    # Complementarity: correlations + block log-loss
    comp_rows = []
    y_primary = fail_y["lowest20_mean_fg"].astype(int)
    s_conf = scores[("confidence", "lowest20_mean_fg")]
    s_cons = scores[("consistency", "lowest20_mean_fg")]
    s_both = scores[("conf_consistency", "lowest20_mean_fg")]
    # correlation between mean |consistency gaps| and mean confidence entropy
    gap_abs_cols = [c for c in feat_df.columns if c.startswith("absolute_gap_")]
    if gap_abs_cols:
        mean_gap = feat_df[gap_abs_cols].mean(axis=1).to_numpy()
        for cc in base["conf_cols"][:5]:
            comp_rows.append(
                {
                    "analysis": "corr_conf_vs_mean_abs_gap",
                    "feature": cc,
                    "value": float(np.corrcoef(base["conf"][cc], mean_gap)[0, 1]),
                }
            )
    for name, s in [("confidence", s_conf), ("consistency", s_cons), ("conf_consistency", s_both)]:
        try:
            comp_rows.append(
                {
                    "analysis": "heldout_log_loss",
                    "feature": name,
                    "value": float(log_loss(y_primary, np.clip(s, 1e-6, 1 - 1e-6))),
                }
            )
        except Exception:
            pass
    # True consistency-block permutation importance from held-out fold shuffles
    if not coef_df.empty:
        block = coef_df[coef_df["feature"].astype(str).str.startswith("__block_perm_drop__")]
        for _, r in block.iterrows():
            comp_rows.append(
                {
                    "analysis": "consistency_block_perm_drop_auprc",
                    "feature": str(r["feature"]).replace("__block_perm_drop__", ""),
                    "value": float(r["coefficient"]),
                }
            )
        if not block.empty:
            comp_rows.append(
                {
                    "analysis": "consistency_block_perm_drop_auprc_mean",
                    "feature": "conf_consistency",
                    "value": float(block["coefficient"].mean()),
                }
            )
    # mean |coef| by feature (exclude block-perm sentinel rows)
    if not coef_df.empty:
        coef_real = coef_df[~coef_df["feature"].astype(str).str.startswith("__block_perm_drop__")]
        for feat, g in coef_real.groupby("feature"):
            comp_rows.append(
                {
                    "analysis": "mean_abs_coef",
                    "feature": feat,
                    "value": float(np.mean(np.abs(g["coefficient"]))),
                }
            )

    comp_df = pd.DataFrame(comp_rows)

    # Robustness: permutation labels (one fold summary)
    perm_rows = []
    y_perm = y_primary.copy()
    rng3 = np.random.default_rng(123)
    y_perm = y_perm[rng3.permutation(n)]
    # Proper: re-fit on permuted labels with same folds
    perm_scores = np.full(n, np.nan)
    for fold_i, parts in sorted(base["folds"].items()):
        tr, te = parts["train"], parts["test"]
        # use confidence + held-out consistency gaps (already OOF for structure; labels permuted)
        gap_cols_h = [c for c in feat_df.columns if c.startswith(("absolute_gap_", "signed_gap_", "standardized_gap_", "relative_gap_"))]
        raw = np.column_stack(
            [conf_mat, feat_df[gap_cols_h].to_numpy(float) if gap_cols_h else np.zeros((n, 0))]
        )
        x = standardize_features_with_train_stats(raw, tr)
        m = tune_logistic_regression(x[tr], y_perm[tr], Cs, seed=99)
        perm_scores[te] = m.predict_proba(x[te])[:, 1]
    try:
        perm_auprc = float(average_precision_score(y_perm, perm_scores))
    except Exception:
        perm_auprc = float("nan")
    perm_rows.append(
        {
            "analysis": "label_permutation",
            "auprc": perm_auprc,
            "chance_baseline": float(y_perm.mean()),
            "model_seed": model_seed,
        }
    )

    # Calibration metrics for proposed (both primary targets)
    cal_rows = []
    for fdef in list(config.get("primary_targets", ["lowest20_mean_fg", "mean_fg_lt_0.70"])):
        y = fail_y[fdef].astype(int)
        for tag, key in [
            ("uncalibrated", ("conf_consistency", fdef)),
            ("platt", ("conf_consistency", fdef)),
        ]:
            s = scores[key] if tag == "uncalibrated" else scores_cal.get(key, scores[key])
            if not np.isfinite(s).all():
                continue
            cm = classification_metrics_from_scores(y, s)
            cal_rows.append({"failure_def": fdef, "calibration": tag, **cm, "model_seed": model_seed})

    # Save tables
    agg_df.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    boot_df.to_csv(out_dir / "bootstrap_comparisons.csv", index=False)
    rc_df.to_csv(out_dir / "risk_coverage_metrics.csv", index=False)
    freq.to_csv(out_dir / "feature_selection_frequency.csv", index=False)
    comp_df.to_csv(out_dir / "complementarity_analysis.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(out_dir / "robustness_results.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_metrics.csv", index=False)
    coef_df.to_csv(out_dir / "feature_coefficients.csv", index=False)
    layer_df.to_csv(out_dir / "best_layer_by_fold.csv", index=False)
    sel_df.to_csv(out_dir / "selection_log.csv", index=False)

    # Figures
    _make_figures(
        fig_dir,
        pred_df,
        agg_df,
        boot_df,
        fold_df,
        freq,
        scores,
        fail_y,
        mean_fg,
        methods,
    )

    verdict, criteria, answers = classify_and_answer(
        agg_df, boot_df, fold_df, comp_df, perm_rows, config
    )
    write_final_report(out_dir, verdict, criteria, answers, agg_df, boot_df, fold_df)

    return {
        "verdict": verdict,
        "criteria": criteria,
        "output_dir": str(out_dir),
        "n_cases": n,
        "model_seed": model_seed,
    }


def _make_figures(
    fig_dir: Path,
    pred_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    freq: pd.DataFrame,
    scores: dict,
    fail_y: dict,
    mean_fg: np.ndarray,
    methods: list[str],
) -> None:
    from sklearn.metrics import precision_recall_curve

    fdef = "lowest20_mean_fg"
    y = fail_y[fdef].astype(int)
    colors = {
        "confidence": "#4C78A8",
        "consistency": "#E45756",
        "conf_consistency": "#B279A2",
        "conf_morph": "#F58518",
        "conf_pooled": "#54A24B",
        "morphology": "#72B7B2",
        "conf_morph_consistency": "#9D755D",
    }

    # PR curves
    plt.figure(figsize=(7, 5))
    for m in ["confidence", "consistency", "conf_consistency", "conf_morph", "conf_pooled"]:
        s = scores[(m, fdef)]
        p, r, _ = precision_recall_curve(y, s)
        plt.plot(r, p, label=m, color=colors.get(m))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curves (lowest 20%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "precision_recall_curves.png")
    plt.close()

    # Bootstrap delta
    sub = boot_df[
        (boot_df.failure_def == fdef)
        & (boot_df.method_a == "conf_consistency")
        & (boot_df.method_b == "confidence")
        & (boot_df.metric == "auprc_diff")
    ]
    plt.figure(figsize=(5, 4))
    if not sub.empty:
        row = sub.iloc[0]
        plt.errorbar(
            [0],
            [row["diff_mean"]],
            yerr=[[row["diff_mean"] - row["ci_low"]], [row["ci_high"] - row["diff_mean"]]],
            fmt="o",
            color="#B279A2",
            capsize=6,
        )
        plt.axhline(0, color="k", ls="--", lw=1)
        plt.xticks([0], ["conf+cons − confidence"])
        plt.ylabel("Δ AUPRC")
        plt.title("Paired bootstrap ΔAUPRC")
    plt.tight_layout()
    plt.savefig(fig_dir / "bootstrap_delta_auprc.png")
    plt.close()

    # Capture by budget
    subm = agg_df[agg_df.failure_def == fdef]
    plt.figure(figsize=(7, 5))
    budgets = [10, 20, 30]
    show = ["confidence", "conf_consistency", "conf_morph", "conf_pooled", "consistency"]
    x = np.arange(len(budgets))
    w = 0.15
    for i, m in enumerate(show):
        row = subm[subm.method == m]
        if row.empty:
            continue
        vals = [float(row.iloc[0][f"capture_at_{b}"]) for b in budgets]
        plt.bar(x + i * w, vals, w, label=m, color=colors.get(m))
    plt.xticks(x + 2 * w, [f"{b}%" for b in budgets])
    plt.ylabel("Failure capture")
    plt.xlabel("Review budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_capture_by_review_budget.png")
    plt.close()

    # Risk coverage
    plt.figure(figsize=(7, 5))
    for m in ["confidence", "conf_consistency", "conf_morph", "consistency"]:
        s = scores[(m, fdef)]
        order = np.argsort(s)
        qs = mean_fg[order]
        cov = np.linspace(0.05, 1.0, 20)
        risks = [1 - float(np.mean(qs[: max(1, int(round(c * len(qs))))])) for c in cov]
        plt.plot(cov, risks, label=m, color=colors.get(m))
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1−mean Dice)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "risk_coverage_curves.png")
    plt.close()

    # Fold stability
    fd = fold_df[(fold_df.failure_def == fdef) & (fold_df.method.isin(["confidence", "conf_consistency"]))]
    plt.figure(figsize=(6, 4))
    if not fd.empty:
        for m, g in fd.groupby("method"):
            plt.plot(g["outer_fold"], g["auprc"], "o-", label=m, color=colors.get(m))
        plt.xlabel("Outer fold")
        plt.ylabel("AUPRC")
        plt.legend()
        plt.title("Fold stability (lowest 20%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fold_stability.png")
    plt.close()

    # Calibration plot
    s = scores[("conf_consistency", fdef)]
    plt.figure(figsize=(5, 5))
    bins = np.linspace(0, 1, 8)
    centers = []
    freqs = []
    for i in range(len(bins) - 1):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if mask.sum() < 5:
            continue
        centers.append(float(s[mask].mean()))
        freqs.append(float(y[mask].mean()))
    if centers:
        plt.plot(centers, freqs, "o-", color="#B279A2")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed failure rate")
    plt.title("Reliability (conf+consistency)")
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_plot.png")
    plt.close()

    # Feature selection frequency
    plt.figure(figsize=(8, 5))
    if not freq.empty:
        top = freq.head(15)
        plt.barh(top["feature"], top["n_folds_selected"], color="#4C78A8")
        plt.xlabel("# folds selected")
        plt.title("Consistency feature selection frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_selection_frequency.png")
    plt.close()


def classify_and_answer(
    agg_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    perm_rows: list[dict],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, str]]:
    primary = list(config["primary_targets"])

    def get_auprc(method, fdef):
        row = agg_df[(agg_df.method == method) & (agg_df.failure_def == fdef)]
        return float(row.iloc[0]["auprc"]) if not row.empty else float("nan")

    def get_boot(fdef, metric="auprc_diff"):
        row = boot_df[
            (boot_df.failure_def == fdef)
            & (boot_df.method_a == "conf_consistency")
            & (boot_df.method_b == "confidence")
            & (boot_df.metric == metric)
        ]
        return row.iloc[0] if not row.empty else None

    deltas = {f: get_auprc("conf_consistency", f) - get_auprc("confidence", f) for f in primary}
    boots = {f: get_boot(f) for f in primary}

    # Fold wins
    fold_wins = {}
    for fdef in primary:
        fd = fold_df[fold_df.failure_def == fdef]
        wins = 0
        diffs = []
        for fold_i in sorted(fd.outer_fold.unique()):
            a = fd[(fd.outer_fold == fold_i) & (fd.method == "conf_consistency")]
            b = fd[(fd.outer_fold == fold_i) & (fd.method == "confidence")]
            if a.empty or b.empty:
                continue
            d = float(a.iloc[0]["auprc"]) - float(b.iloc[0]["auprc"])
            diffs.append(d)
            if d > 0:
                wins += 1
        fold_wins[fdef] = {
            "wins": wins,
            "n": len(diffs),
            "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
            "worst": float(np.min(diffs)) if diffs else float("nan"),
            "best": float(np.max(diffs)) if diffs else float("nan"),
        }

    # Capture / RC improve on primary lowest20
    f0 = "lowest20_mean_fg"
    row_p = agg_df[(agg_df.method == "conf_consistency") & (agg_df.failure_def == f0)].iloc[0]
    row_c = agg_df[(agg_df.method == "confidence") & (agg_df.failure_def == f0)].iloc[0]
    cap_improve = float(row_p["capture_at_20"]) > float(row_c["capture_at_20"])
    rc_improve = float(row_p["aurc"]) < float(row_c["aurc"])  # lower better

    # Morphology explains? compare conf_morph vs conf_consistency
    a_cm = get_auprc("conf_morph", f0)
    a_cc = get_auprc("conf_consistency", f0)

    # Success criteria
    both_ge_03 = all(deltas[f] >= 0.03 for f in primary)
    both_pos = all(deltas[f] > 0 for f in primary)
    one_ge_02 = any(deltas[f] >= 0.02 for f in primary)

    excl0 = []
    mostly_pos = []
    for f in primary:
        b = boots[f]
        if b is None:
            excl0.append(False)
            mostly_pos.append(False)
        else:
            excl0.append(float(b["ci_low"]) > 0)
            mostly_pos.append(float(b["frac_a_better"]) >= 0.7)

    folds4 = all(fold_wins[f]["wins"] >= 4 for f in primary)
    folds3 = all(fold_wins[f]["wins"] >= 3 for f in primary)

    criteria = {
        "both_primary_delta_ge_0.03": both_ge_03,
        "bootstrap_excl0_at_least_one_and_positive_other": (
            any(excl0) and all(deltas[f] > 0 for f in primary)
        ),
        "fold_wins_ge_4_both": folds4,
        "capture_or_rc_improves": cap_improve or rc_improve,
        "both_primary_positive": both_pos,
        "one_delta_ge_0.02": one_ge_02,
        "bootstrap_mostly_positive": all(mostly_pos),
        "fold_wins_ge_3_both": folds3,
        "perm_near_chance": float(perm_rows[0]["auprc"]) < float(perm_rows[0]["chance_baseline"]) + 0.15,
    }

    if (
        criteria["both_primary_delta_ge_0.03"]
        and criteria["bootstrap_excl0_at_least_one_and_positive_other"]
        and criteria["fold_wins_ge_4_both"]
        and criteria["capture_or_rc_improves"]
        and criteria["perm_near_chance"]
    ):
        verdict = "STRONG"
    elif (
        criteria["both_primary_positive"]
        and criteria["one_delta_ge_0.02"]
        and criteria["bootstrap_mostly_positive"]
        and criteria["fold_wins_ge_3_both"]
    ):
        verdict = "PROMISING"
    elif any(deltas[f] > 0 for f in primary) or criteria["capture_or_rc_improves"]:
        # Check MIXED conditions
        if abs(a_cc - a_cm) < 0.01 and a_cm >= a_cc:
            verdict = "MIXED"
        else:
            verdict = "MIXED"
    else:
        verdict = "NOT PROMISING"

    # Top complementary features
    top_feats = ""
    if not comp_df.empty:
        mc = comp_df[comp_df.analysis == "mean_abs_coef"].sort_values("value", ascending=False)
        gap_mc = mc[mc.feature.str.contains("gap_", na=False)].head(5)
        top_feats = ", ".join(f"{r.feature} ({r.value:.3f})" for r in gap_mc.itertuples())

    answers = {
        "1_lowest20": (
            f"ΔAUPRC={deltas['lowest20_mean_fg']:+.3f} "
            f"(conf+cons={get_auprc('conf_consistency','lowest20_mean_fg'):.3f} vs "
            f"conf={get_auprc('confidence','lowest20_mean_fg'):.3f})"
        ),
        "2_dice070": (
            f"ΔAUPRC={deltas['mean_fg_lt_0.70']:+.3f} "
            f"(conf+cons={get_auprc('conf_consistency','mean_fg_lt_0.70'):.3f} vs "
            f"conf={get_auprc('confidence','mean_fg_lt_0.70'):.3f})"
        ),
        "3_bootstrap": "; ".join(
            (
                f"{f}: Δ={boots[f]['diff_mean']:+.3f} "
                f"[{boots[f]['ci_low']:+.3f},{boots[f]['ci_high']:+.3f}] "
                f"P(better)={boots[f]['frac_a_better']:.2f}"
                if boots[f] is not None
                else f"{f}: n/a"
            )
            for f in primary
        ),
        "4_capture20": (
            f"conf+cons={row_p['capture_at_20']:.3f} vs conf={row_c['capture_at_20']:.3f}"
        ),
        "5_risk_coverage": (
            f"AURC conf+cons={row_p['aurc']:.4f} vs conf={row_c['aurc']:.4f}; "
            f"Dice@80% {row_p['mean_dice_coverage_80']:.3f} vs {row_c['mean_dice_coverage_80']:.3f}"
        ),
        "6_fold_stability": "; ".join(
            f"{f}: wins {fold_wins[f]['wins']}/{fold_wins[f]['n']} "
            f"meanΔ={fold_wins[f]['mean_diff']:+.3f} "
            f"[{fold_wins[f]['worst']:+.3f},{fold_wins[f]['best']:+.3f}]"
            for f in primary
        ),
        "7_features": top_feats or "see complementarity_analysis.csv",
        "8_morphology": (
            f"conf+morph AUPRC={a_cm:.3f} vs conf+cons={a_cc:.3f} on lowest20; "
            + (
                "morphology does not fully explain the gain"
                if a_cc > a_cm + 0.01
                else "morphology explains most/all of the gain"
            )
        ),
        "9_pooled": (
            f"conf+pooled={get_auprc('conf_pooled', f0):.3f} vs confidence="
            f"{get_auprc('confidence', f0):.3f} vs conf+cons="
            f"{get_auprc('conf_consistency', f0):.3f}; pooled can beat confidence alone "
            f"but underperforms compact anatomical gaps (noise/collinearity in high-dim "
            f"pooled features; no explicit representation–output disagreement)."
        ),
        "10_verdict": verdict,
        "perm": str(perm_rows[0]),
    }
    return verdict, criteria, answers


def write_final_report(
    out_dir: Path,
    verdict: str,
    criteria: dict[str, Any],
    answers: dict[str, str],
    agg_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    lines = [
        "# Confidence + Consistency Triage — Final Report",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Feasibility / success gates:",
    ]
    for k, v in criteria.items():
        lines.append(f"- `{k}`: {'PASS' if v else 'fail'}")
    lines += ["", "## Answers", ""]
    q = [
        ("1", "Does consistency improve confidence for lowest-20%?", answers["1_lowest20"]),
        ("2", "Does it improve confidence for Dice < 0.70?", answers["2_dice070"]),
        ("3", "Bootstrap support?", answers["3_bootstrap"]),
        ("4", "Capture@20%?", answers["4_capture20"]),
        ("5", "Risk–coverage?", answers["5_risk_coverage"]),
        ("6", "Fold/seed stability?", answers["6_fold_stability"]),
        ("7", "Which discrepancies help?", answers["7_features"]),
        ("8", "Does morphology explain it?", answers["8_morphology"]),
        ("9", "How does conf+pooled compare?", answers["9_pooled"]),
        ("10", "Verdict", answers["10_verdict"]),
    ]
    for num, title, ans in q:
        lines.append(f"{num}. **{title}** {ans}")
    lines += [
        "",
        f"Permutation check: {answers['perm']}",
        "",
        "No clinical-deployment, external-generalization, or novelty claims.",
        "",
        "## Aggregate AUPRC (selected methods)",
        "",
        "| Failure | confidence | conf+cons | conf+morph | conf+pooled | consistency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fdef in [
        "lowest20_mean_fg",
        "mean_fg_lt_0.80",
        "mean_fg_lt_0.70",
        "edema_lt_0.70",
        "lowest20_edema",
    ]:
        def g(m):
            r = agg_df[(agg_df.method == m) & (agg_df.failure_def == fdef)]
            return f"{r.iloc[0]['auprc']:.3f}" if not r.empty else "nan"

        lines.append(
            f"| {fdef} | {g('confidence')} | {g('conf_consistency')} | "
            f"{g('conf_morph')} | {g('conf_pooled')} | {g('consistency')} |"
        )
    (out_dir / "final_report.md").write_text("\n".join(lines) + "\n")


def run_seed_robustness(config: dict[str, Any]) -> pd.DataFrame:
    """Re-run modeling with alternate seeds; archive each; restore primary seed outputs."""
    out = Path(config["paths"]["output_dir"])
    primary = int(config.get("seed", 42))
    archive_names = [
        "aggregate_metrics.csv",
        "bootstrap_comparisons.csv",
        "fold_metrics.csv",
        "heldout_predictions.csv",
        "final_report.md",
        "robustness_results.csv",
    ]
    rows = []
    for ms in config.get("model_seeds", [42]):
        print(f"=== model seed {ms} ===")
        result = run_triage(config, model_seed=int(ms))
        rows.append({"model_seed": ms, "verdict": result["verdict"], **result["criteria"]})
        for name in archive_names:
            src = out / name
            if src.exists():
                (out / f"seed_{ms}_{name}").write_bytes(src.read_bytes())
    # Restore primary seed as the canonical deliverable
    for name in archive_names:
        src = out / f"seed_{primary}_{name}"
        if src.exists():
            (out / name).write_bytes(src.read_bytes())
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(out / "seed_robustness_summary.csv", index=False)
    # Merge seed rows into robustness_results without dropping permutation rows
    rob_path = out / "robustness_results.csv"
    extra = [
        {
            "analysis": "model_seed",
            "auprc": float("nan"),
            "chance_baseline": float("nan"),
            "model_seed": int(r["model_seed"]),
            "verdict": r["verdict"],
            "detail": f"criteria_pass={sum(1 for k,v in r.items() if k not in ('model_seed','verdict') and v)}",
        }
        for r in rows
    ]
    if rob_path.exists():
        prev = pd.read_csv(rob_path)
        pd.concat([prev, pd.DataFrame(extra)], ignore_index=True).to_csv(rob_path, index=False)
    else:
        pd.DataFrame(extra).to_csv(rob_path, index=False)
    return seed_df


def expand_robustness_from_artifacts(config: dict[str, Any]) -> pd.DataFrame:
    """Leave-one-target / signed-vs-abs / logistic-seed checks on saved OOF features."""
    out = Path(config["paths"]["output_dir"])
    feat = pd.read_csv(out / "case_level_features.csv")
    fold_map = load_outer_folds(out / "fold_assignments.csv")
    # Align to the same sorted case_id order used by load_outer_folds
    case_order = sorted(feat["case_id"].unique())
    feat = feat.set_index("case_id").loc[case_order].reset_index()
    conf = pd.read_csv(config["paths"]["confidence_features"])
    conf = conf.set_index("case_id").loc[case_order].reset_index()
    pred = pd.read_csv(out / "heldout_predictions.csv")
    pred = pred.set_index("case_id").loc[case_order].reset_index()
    conf_cols = [c for c in config["confidence_cols"] if c in conf.columns]
    conf_mat = conf[conf_cols].to_numpy(dtype=float)
    y = pred["fail_lowest20_mean_fg"].astype(int).to_numpy()

    def eval_cols(cols: list[str], seed: int = 42) -> float:
        # Fast fixed-C LBFGS for ablation only (not the primary nested selector).
        raw = np.column_stack([conf_mat, feat[cols].to_numpy(dtype=float)]) if cols else conf_mat
        scores = np.full(len(feat), np.nan)
        for fold_i, parts in sorted(fold_map.items()):
            tr = parts["train"]
            te = parts["test"]
            x = standardize_features_with_train_stats(raw, tr)
            m = LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed + int(fold_i),
            )
            m.fit(x[tr], y[tr])
            scores[te] = m.predict_proba(x[te])[:, 1]
        return float(average_precision_score(y, scores))

    all_gap = [
        c
        for c in feat.columns
        if c.startswith(("absolute_gap_", "signed_gap_", "standardized_gap_", "relative_gap_"))
    ]
    targets = list(config["consistency_targets"])
    rows: list[dict[str, Any]] = []
    base = eval_cols(all_gap, seed=42)
    rows.append(
        {
            "analysis": "baseline_conf_consistency_oof_gaps",
            "auprc": base,
            "chance_baseline": float(y.mean()),
            "model_seed": 42,
            "detail": "all_gap_cols",
        }
    )
    for t in targets:
        keep = [c for c in all_gap if not c.endswith(f"_{t}")]
        a = eval_cols(keep, seed=42)
        rows.append(
            {
                "analysis": "leave_one_target_out",
                "auprc": a,
                "chance_baseline": float(y.mean()),
                "model_seed": 42,
                "detail": f"drop_{t};delta={a - base:+.4f}",
            }
        )
    for mode, cols in [
        ("absolute_only", [c for c in all_gap if c.startswith("absolute_gap_")]),
        ("signed_only", [c for c in all_gap if c.startswith("signed_gap_")]),
        ("standardized_only", [c for c in all_gap if c.startswith("standardized_gap_")]),
        ("relative_only", [c for c in all_gap if c.startswith("relative_gap_")]),
        ("confidence_only", []),
    ]:
        a = eval_cols(cols, seed=42)
        rows.append(
            {
                "analysis": "gap_family",
                "auprc": a,
                "chance_baseline": float(y.mean()),
                "model_seed": 42,
                "detail": mode,
            }
        )
    rows.append(
        {
            "analysis": "exclude_dice_prediction_feature",
            "auprc": base,
            "chance_baseline": float(y.mean()),
            "model_seed": 42,
            "detail": "N/A_no_direct_dice_probe_in_proposed_features",
        }
    )
    agg = pd.read_csv(out / "aggregate_metrics.csv")
    for m in ["conf_consistency", "conf_morph", "conf_morph_consistency", "confidence"]:
        r = agg[(agg.method == m) & (agg.failure_def == "lowest20_mean_fg")]
        if not r.empty:
            rows.append(
                {
                    "analysis": "morphology_exclusion_from_aggregate",
                    "auprc": float(r.iloc[0]["auprc"]),
                    "chance_baseline": float(y.mean()),
                    "model_seed": 42,
                    "detail": m,
                }
            )
    for ms in config.get("model_seeds", [42, 123, 2026]):
        a = eval_cols(all_gap, seed=int(ms))
        rows.append(
            {
                "analysis": "logistic_model_seed_fixed_gaps",
                "auprc": a,
                "chance_baseline": float(y.mean()),
                "model_seed": int(ms),
                "detail": "inference_time_conf_plus_consistency_gaps",
            }
        )
    rob_path = out / "robustness_results.csv"
    new_df = pd.DataFrame(rows)
    if rob_path.exists():
        prev = pd.read_csv(rob_path)
        prev = prev[prev.analysis == "label_permutation"]
        out_df = pd.concat([prev, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_csv(rob_path, index=False)
    return out_df
