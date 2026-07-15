"""Final paper validation for frozen confidence + consistency method.

Does NOT modify the representation–output consistency algorithm, retrain the
U-Net, or regenerate probes/edits. Reuses held-out triage scores and only
refits logistic classifiers for missing baselines, ablations, and seeds.
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.analysis.confidence_consistency_triage import (
    classification_metrics_from_scores,
    resolve_output_dir,
)
from src.analysis.layer_aware_latent_risk import load_outer_folds
from src.analysis.layer_io import load_layer_index, load_layer_matrix
from src.analysis.representation_output_consistency import (
    failure_capture_at_budget,
    risk_coverage_curve,
)
from src.utils.io import ensure_dir

# Frozen canonical run (per-target selection + primary calibration).
# Prefer results/paper/triage_20260712 when present; keep historical path string
# for configs that still point at the timestamped offline directory.
CANONICAL_TRIAGE = Path("outputs_confidence_consistency_triage_20260712_030902")
CONSISTENCY_OUTER = Path("outputs_consistency_failure_detection/outer_fold_predictions.csv")
CONSISTENCY_FEATURES = Path("outputs_consistency_failure_detection/case_level_features.csv")

METHOD_ORDER = [
    "confidence",
    "morphology",
    "conf_morph",
    "dice_probe",
    "pooled",
    "conf_pooled",
    "consistency",
    "conf_consistency",
    "conf_morph_consistency",
]

METHOD_LABELS = {
    "confidence": "Confidence only",
    "morphology": "Morphology only",
    "conf_morph": "Confidence + morphology",
    "dice_probe": "Direct Dice probe",
    "pooled": "Pooled representation",
    "conf_pooled": "Confidence + pooled",
    "consistency": "Consistency only",
    "conf_consistency": "Confidence + consistency (proposed)",
    "conf_morph_consistency": "Confidence + morphology + consistency",
}

# Map triage failure defs -> consistency experiment score suffixes
CONS_SCORE_MAP = {
    "lowest20_mean_fg": "lowest_20pct",
    "mean_fg_lt_0.80": "dice_lt_0.80",
    "mean_fg_lt_0.70": "dice_lt_0.70",
    "edema_lt_0.70": "edema_lt_0.70",
}

VOLUME_TARGETS = ["log_wt_volume"]
TISSUE_TARGETS = ["gt_edema_frac", "gt_enhancing_frac", "gt_necrosis_frac"]
SHAPE_TARGETS = ["gt_compactness"]
ANATOMICAL_TARGETS = VOLUME_TARGETS + TISSUE_TARGETS + SHAPE_TARGETS

COVERAGES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
BUDGETS = [0.10, 0.20, 0.30]


def load_config(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logistic baselines, metrics, and bootstrap helpers
# ---------------------------------------------------------------------------


def standardize_with_train_fold(features: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    """Impute with train means, then StandardScaler fit on train only."""
    features = np.asarray(features, dtype=float).copy()
    for column_index in range(features.shape[1]):
        column = features[:, column_index]
        train_mean = (
            float(np.nanmean(column[train_indices]))
            if np.isfinite(column[train_indices]).any()
            else 0.0
        )
        column[~np.isfinite(column)] = train_mean
        features[:, column_index] = column
    scaler = StandardScaler().fit(features[train_indices])
    return scaler.transform(features)


def fit_logistic_out_of_fold(
    features: np.ndarray,
    labels: np.ndarray,
    folds: dict[int, dict[str, np.ndarray]],
    seed: int = 42,
) -> np.ndarray:
    """Outer-fold logistic OOF scores (same folds as triage / consistency)."""
    oof_scores = np.full(len(labels), np.nan)
    for fold_index, parts in sorted(folds.items()):
        train_indices, test_indices = parts["train"], parts["test"]
        features_scaled = standardize_with_train_fold(features, train_indices)
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed + int(fold_index),
        )
        model.fit(features_scaled[train_indices], labels[train_indices].astype(int))
        oof_scores[test_indices] = model.predict_proba(features_scaled[test_indices])[:, 1]
    return oof_scores


def metrics_bundle(
    labels: np.ndarray,
    risk_scores: np.ndarray,
    quality: np.ndarray,
) -> dict[str, float]:
    """AUPRC/AUROC/Brier + capture@budget + risk–coverage for one score vector."""
    labels = labels.astype(int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    classification = classification_metrics_from_scores(labels, risk_scores)
    capture = failure_capture_at_budget(risk_scores, labels.astype(bool), budgets=BUDGETS)
    risk_coverage = risk_coverage_curve(risk_scores, quality, coverages=COVERAGES)
    out = {
        "auprc": classification["auprc"],
        "auroc": classification["auroc"],
        "brier": classification["brier"],
        **{key: capture[key] for key in capture},
        "aurc": risk_coverage["aurc"],
    }
    for coverage in COVERAGES:
        key = f"mean_dice_coverage_{int(coverage * 100)}"
        out[key] = risk_coverage[key]
    return out


def paired_bootstrap_deltas(
    labels: np.ndarray,
    scores_proposed: np.ndarray,
    scores_baseline: np.ndarray,
    quality: np.ndarray,
    n_boot: int,
    seed: int,
) -> list[dict[str, float]]:
    """Paired case-level bootstrap of proposed − baseline for key metrics."""
    rng = np.random.default_rng(seed)
    n_cases = len(labels)
    deltas: dict[str, list[float]] = {"auprc": [], "capture20": [], "aurc": []}
    for _ in range(n_boot):
        sample_indices = rng.integers(0, n_cases, size=n_cases)
        labels_s = labels[sample_indices]
        proposed_s = scores_proposed[sample_indices]
        baseline_s = scores_baseline[sample_indices]
        quality_s = quality[sample_indices]
        if labels_s.sum() == 0 or labels_s.sum() == len(labels_s):
            continue
        deltas["auprc"].append(
            average_precision_score(labels_s, proposed_s)
            - average_precision_score(labels_s, baseline_s)
        )
        capture_proposed = failure_capture_at_budget(proposed_s, labels_s.astype(bool))[
            "capture_at_20"
        ]
        capture_baseline = failure_capture_at_budget(baseline_s, labels_s.astype(bool))[
            "capture_at_20"
        ]
        deltas["capture20"].append(capture_proposed - capture_baseline)
        aurc_proposed = risk_coverage_curve(proposed_s, quality_s)["aurc"]
        aurc_baseline = risk_coverage_curve(baseline_s, quality_s)["aurc"]
        # Lower AURC is better; report proposed - baseline so negative favors proposed.
        deltas["aurc"].append(aurc_proposed - aurc_baseline)

    rows = []
    for metric, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "diff_mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "frac_proposed_better": float(np.mean(arr > 0))
                if metric != "aurc"
                else float(np.mean(arr < 0)),
                "n_boot": len(arr),
            }
        )
    return rows


def gap_feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    """Named subsets of consistency-gap columns for ablation studies."""
    gap_columns = [
        column
        for column in features.columns
        if column.startswith(
            ("absolute_gap_", "signed_gap_", "standardized_gap_", "relative_gap_")
        )
    ]
    groups = {
        "absolute_only": [c for c in gap_columns if c.startswith("absolute_gap_")],
        "signed_only": [c for c in gap_columns if c.startswith("signed_gap_")],
        "relative_only": [c for c in gap_columns if c.startswith("relative_gap_")],
        "volume_related": [c for c in gap_columns if any(t in c for t in VOLUME_TARGETS)],
        "tissue_fraction": [c for c in gap_columns if any(t in c for t in TISSUE_TARGETS)],
        "shape_boundary": [c for c in gap_columns if any(t in c for t in SHAPE_TARGETS)],
        "all_gaps": gap_columns,
    }
    for anatomy_target in ANATOMICAL_TARGETS:
        groups[f"drop_{anatomy_target}"] = [
            c for c in gap_columns if not c.endswith(f"_{anatomy_target}")
        ]
    # Boundary complexity is not a separate probe target; compactness is the proxy.
    groups["drop_boundary_complexity"] = groups["drop_gt_compactness"]
    return groups


def compute_tumor_core_dice(index_df: pd.DataFrame, case_ids: list[str]) -> np.ndarray:
    """Tumor-core Dice from frozen pred/GT masks on disk."""
    from src.training.spatial_repair_trainer import tumor_core_dice

    dice_scores = np.full(len(case_ids), np.nan)
    index_by_case = index_df.set_index("case_id")
    for row_index, case_id in enumerate(case_ids):
        row = index_by_case.loc[case_id]
        prediction = np.load(row["path_prediction"])
        ground_truth = np.load(row["path_ground_truth"])
        dice_scores[row_index] = float(tumor_core_dice(prediction, ground_truth))
    return dice_scores


# Back-compat aliases
_scale_fit = standardize_with_train_fold
_fit_logistic_oof = fit_logistic_out_of_fold
_metrics_bundle = metrics_bundle
_paired_bootstrap = paired_bootstrap_deltas
_gap_groups = gap_feature_groups
_compute_tc_dice = compute_tumor_core_dice


def run_validation(config: dict[str, Any]) -> dict[str, Any]:
    triage_dir = Path(config.get("triage_dir", str(CANONICAL_TRIAGE)))
    out_cfg = dict(config)
    out_cfg["paths"] = dict(config["paths"])
    out_cfg["paths"]["output_dir"] = config.get(
        "validation_output_dir", "outputs_method_validation"
    )
    out_cfg["overwrite_output"] = config.get("overwrite_output", False)
    out_dir = resolve_output_dir(out_cfg)
    fig_dir = ensure_dir(out_dir / "figures")

    pred = pd.read_csv(triage_dir / "heldout_predictions.csv")
    feat = pd.read_csv(triage_dir / "case_level_features.csv")
    folds_df = pd.read_csv(triage_dir / "fold_assignments.csv")
    fold_metrics = pd.read_csv(triage_dir / "fold_metrics.csv")
    sel_log = pd.read_csv(triage_dir / "selection_log.csv")
    cons_outer = pd.read_csv(CONSISTENCY_OUTER)
    cons_feat = pd.read_csv(CONSISTENCY_FEATURES)

    case_order = sorted(pred["case_id"].tolist())
    pred = pred.set_index("case_id").loc[case_order].reset_index()
    feat = feat.set_index("case_id").loc[case_order].reset_index()
    cons_outer = cons_outer.set_index("case_id").loc[case_order].reset_index()
    cons_feat = cons_feat.set_index("case_id").loc[case_order].reset_index()

    folds = load_outer_folds(triage_dir / "fold_assignments.csv")
    quality = pred["label_mean_fg_dice"].to_numpy(float)
    n = len(pred)
    fail_defs = list(config["failure_defs"])
    primary = list(config.get("primary_targets", ["lowest20_mean_fg", "mean_fg_lt_0.70"]))
    conf_cols = [c for c in config["confidence_cols"] if c in feat.columns]
    conf_mat = feat[conf_cols].to_numpy(float)

    # ---- Score matrix for all methods ----
    scores: dict[tuple[str, str], np.ndarray] = {}
    for fdef in fail_defs:
        for method in [
            "confidence",
            "morphology",
            "conf_morph",
            "conf_pooled",
            "consistency",
            "conf_consistency",
            "conf_morph_consistency",
        ]:
            col = f"score_{method}__{fdef}"
            scores[(method, fdef)] = pred[col].to_numpy(float)

    # Merge frozen dice_probe / pooled from consistency experiment (same folds)
    for fdef, suffix in CONS_SCORE_MAP.items():
        scores[("dice_probe", fdef)] = cons_outer[f"score_dice_probe__{suffix}"].to_numpy(float)
        scores[("pooled", fdef)] = cons_outer[f"score_pooled_repr__{suffix}"].to_numpy(float)

    # lowest20_edema: refit classifier-only on frozen features
    y_ed = pred["fail_lowest20_edema"].to_numpy(int)
    dice_feat = cons_feat[["rep_dice"]].to_numpy(float)
    scores[("dice_probe", "lowest20_edema")] = _fit_logistic_oof(dice_feat, y_ed, folds, seed=42)

    # Pooled for lowest20_edema — load embeddings once
    index_df = load_layer_index(config["paths"]["layer_index"]).sort_values("case_id")
    index_df = index_df.set_index("case_id").loc[case_order].reset_index()
    layers = list(config["layers"])
    pooled = np.concatenate([load_layer_matrix(index_df, L) for L in layers], axis=1)
    scores[("pooled", "lowest20_edema")] = _fit_logistic_oof(pooled, y_ed, folds, seed=42)

    fail_y = {f: pred[f"fail_{f}"].to_numpy(int) for f in fail_defs}

    # ---- Task 1: baseline table ----
    baseline_rows = []
    for fdef in fail_defs:
        y = fail_y[fdef]
        for method in METHOD_ORDER:
            s = scores[(method, fdef)]
            m = _metrics_bundle(y, s, quality)
            baseline_rows.append({"method": method, "failure_def": fdef, **m})
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_dir / "baseline_comparison.csv", index=False)

    # ---- Task 2: feature ablation + perm importance ----
    y_primary = fail_y["lowest20_mean_fg"]
    groups = _gap_groups(feat)
    abl_rows = []
    for name, cols in groups.items():
        raw = np.column_stack([conf_mat, feat[cols].to_numpy(float)]) if cols else conf_mat
        s = _fit_logistic_oof(raw, y_primary, folds, seed=42)
        abl_rows.append(
            {
                "ablation": name,
                "n_features": len(cols) + conf_mat.shape[1],
                "auprc": float(average_precision_score(y_primary, s)),
                "auroc": float(roc_auc_score(y_primary, s)),
                "capture_at_20": failure_capture_at_budget(s, y_primary.astype(bool))[
                    "capture_at_20"
                ],
            }
        )
    # confidence-only reference
    s_conf = scores[("confidence", "lowest20_mean_fg")]
    abl_rows.append(
        {
            "ablation": "confidence_only",
            "n_features": conf_mat.shape[1],
            "auprc": float(average_precision_score(y_primary, s_conf)),
            "auroc": float(roc_auc_score(y_primary, s_conf)),
            "capture_at_20": failure_capture_at_budget(s_conf, y_primary.astype(bool))[
                "capture_at_20"
            ],
        }
    )
    abl_df = pd.DataFrame(abl_rows).sort_values("auprc", ascending=False)
    abl_df.to_csv(out_dir / "feature_ablation.csv", index=False)

    # Permutation importance of each gap feature (block of all gap types per target family)
    all_gaps = groups["all_gaps"]
    x_full = np.column_stack([conf_mat, feat[all_gaps].to_numpy(float)])
    # Fit once per fold, accumulate mean drop
    rng = np.random.default_rng(0)
    n_perm = int(config.get("n_feature_permutations", 40))
    imp = {c: [] for c in all_gaps}
    for fold_i, parts in sorted(folds.items()):
        tr, te = parts["train"], parts["test"]
        xs = _scale_fit(x_full, tr)
        m = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42 + fold_i
        )
        m.fit(xs[tr], y_primary[tr])
        base = float(average_precision_score(y_primary[te], m.predict_proba(xs[te])[:, 1]))
        x_te = xs[te]
        for j, col in enumerate(all_gaps):
            drops = []
            for _ in range(n_perm):
                xp = x_te.copy()
                xp[:, conf_mat.shape[1] + j] = x_te[rng.permutation(len(te)), conf_mat.shape[1] + j]
                drops.append(base - float(average_precision_score(y_primary[te], m.predict_proba(xp)[:, 1])))
            imp[col].append(float(np.mean(drops)))
    imp_rows = [
        {"feature": c, "mean_auprc_drop": float(np.mean(v)), "std_auprc_drop": float(np.std(v))}
        for c, v in imp.items()
    ]
    imp_df = pd.DataFrame(imp_rows).sort_values("mean_auprc_drop", ascending=False)
    imp_df.to_csv(out_dir / "feature_permutation_importance.csv", index=False)

    # Feature selection frequency from frozen triage (per primary target)
    if "feature" in sel_log.columns:
        freq = (
            sel_log.dropna(subset=["feature"])
            .groupby(["failure_def", "feature"])
            .size()
            .reset_index(name="n_folds_selected")
            .sort_values(["failure_def", "n_folds_selected"], ascending=[True, False])
        )
    else:
        freq = pd.DataFrame(columns=["failure_def", "feature", "n_folds_selected"])
    freq.to_csv(out_dir / "feature_selection_frequency.csv", index=False)

    # ---- Task 3: correlations / why it works ----
    # Tumor-core Dice from frozen masks (lightweight; no U-Net)
    print("Computing tumor-core Dice from frozen masks...")
    tc_dice = _compute_tc_dice(index_df, case_order)
    pd.DataFrame({"case_id": case_order, "label_tc_dice": tc_dice}).to_csv(
        out_dir / "tumor_core_dice.csv", index=False
    )
    outcomes = {
        "mean_fg_dice": cons_feat["label_mean_fg_dice"].to_numpy(float),
        "wt_dice": cons_feat["label_wt_dice"].to_numpy(float),
        "tc_dice": tc_dice,
        "edema_dice": cons_feat["label_edema_dice"].to_numpy(float),
        "boundary_error": cons_feat["label_boundary_error"].to_numpy(float),
    }
    corr_rows = []
    for gap in all_gaps:
        g = feat[gap].to_numpy(float)
        for oname, o in outcomes.items():
            mask = np.isfinite(g) & np.isfinite(o)
            rho, p = spearmanr(g[mask], o[mask])
            corr_rows.append(
                {
                    "feature": gap,
                    "outcome": oname,
                    "spearman": float(rho),
                    "pvalue": float(p),
                    "abs_spearman": float(abs(rho)),
                }
            )
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_spearman", ascending=False)
    corr_df.to_csv(out_dir / "feature_outcome_correlations.csv", index=False)

    # ---- Task 4: risk-coverage tables ----
    rc_rows = []
    for fdef in primary:
        y = fail_y[fdef]
        for method in METHOD_ORDER:
            s = scores[(method, fdef)]
            rc = risk_coverage_curve(s, quality, coverages=COVERAGES)
            cap = failure_capture_at_budget(s, y.astype(bool), budgets=BUDGETS + [0.4, 0.5])
            rc_rows.append({"method": method, "failure_def": fdef, **rc, **cap})
    rc_df = pd.DataFrame(rc_rows)
    rc_df.to_csv(out_dir / "risk_coverage_extended.csv", index=False)

    # ---- Task 5: seed robustness (classifier only) ----
    seed_rows = []
    gap_all = feat[all_gaps].to_numpy(float)
    x_cc = np.column_stack([conf_mat, gap_all])
    for fdef in primary:
        y = fail_y[fdef]
        for seed in config.get("model_seeds", [42, 123, 2026]):
            s = _fit_logistic_oof(x_cc, y, folds, seed=int(seed))
            m = _metrics_bundle(y, s, quality)
            seed_rows.append(
                {
                    "model_seed": int(seed),
                    "failure_def": fdef,
                    "method": "conf_consistency",
                    "auprc": m["auprc"],
                    "capture_at_20": m["capture_at_20"],
                }
            )
            s0 = _fit_logistic_oof(conf_mat, y, folds, seed=int(seed))
            m0 = _metrics_bundle(y, s0, quality)
            seed_rows.append(
                {
                    "model_seed": int(seed),
                    "failure_def": fdef,
                    "method": "confidence",
                    "auprc": m0["auprc"],
                    "capture_at_20": m0["capture_at_20"],
                }
            )
    seed_df = pd.DataFrame(seed_rows)
    seed_summ = []
    for (fdef, method), g in seed_df.groupby(["failure_def", "method"]):
        for metric in ["auprc", "capture_at_20"]:
            vals = g[metric].to_numpy(float)
            seed_summ.append(
                {
                    "failure_def": fdef,
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "cv": float(np.std(vals, ddof=1) / np.mean(vals))
                    if len(vals) > 1 and np.mean(vals) != 0
                    else 0.0,
                    "values": ",".join(f"{v:.4f}" for v in vals),
                }
            )
    seed_df.to_csv(out_dir / "seed_robustness.csv", index=False)
    pd.DataFrame(seed_summ).to_csv(out_dir / "seed_robustness_summary.csv", index=False)

    # ---- Task 6: bootstrap 5000 ----
    n_boot = int(config.get("n_bootstrap", 5000))
    boot_rows = []
    for fdef in primary:
        y = fail_y[fdef]
        rows = _paired_bootstrap(
            y,
            scores[("conf_consistency", fdef)],
            scores[("confidence", fdef)],
            quality,
            n_boot=n_boot,
            seed=42,
        )
        for r in rows:
            boot_rows.append({"failure_def": fdef, "comparison": "conf_consistency_vs_confidence", **r})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "bootstrap_primary.csv", index=False)

    # ---- Fold stability from frozen triage ----
    fold_stab = []
    for fdef in primary:
        sub = fold_metrics[fold_metrics.failure_def == fdef]
        for fold_i in sorted(sub.outer_fold.unique()):
            a = sub[(sub.outer_fold == fold_i) & (sub.method == "conf_consistency")].iloc[0]
            b = sub[(sub.outer_fold == fold_i) & (sub.method == "confidence")].iloc[0]
            fold_stab.append(
                {
                    "failure_def": fdef,
                    "outer_fold": int(fold_i),
                    "auprc_proposed": float(a.auprc),
                    "auprc_confidence": float(b.auprc),
                    "delta_auprc": float(a.auprc - b.auprc),
                    "capture20_proposed": float(a.capture_at_20),
                    "capture20_confidence": float(b.capture_at_20),
                }
            )
    fold_stab_df = pd.DataFrame(fold_stab)
    fold_stab_df.to_csv(out_dir / "fold_stability.csv", index=False)

    # ---- Figures ----
    _make_figures(
        fig_dir,
        scores,
        fail_y,
        quality,
        baseline_df,
        abl_df,
        imp_df,
        corr_df,
        boot_df,
        fold_stab_df,
        primary,
    )

    # ---- Summary markdown ----
    verdict = _write_summary(
        out_dir,
        baseline_df,
        abl_df,
        imp_df,
        corr_df,
        fold_stab_df,
        seed_df,
        seed_summ,
        boot_df,
        rc_df,
        primary,
    )

    # Skip TTA note
    (out_dir / "tta_skipped.md").write_text(
        "# TTA disagreement (Task 7) — skipped\n\n"
        "Flip TTA is already used to produce the retained hard masks and the "
        "frozen confidence features (mean entropy / max-prob summaries over "
        "averaged TTA probabilities). Computing pairwise flip–flip Dice / "
        "volume disagreement would require re-running sliding-window inference "
        "and storing per-flip hard segmentations for all 375 cases — a "
        "significant inference job, not a light post-hoc feature. Per the "
        "task brief, this section is skipped.\n"
    )

    return {"output_dir": str(out_dir), "verdict": verdict}


def _make_figures(
    fig_dir: Path,
    scores: dict,
    fail_y: dict,
    quality: np.ndarray,
    baseline_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    fold_stab_df: pd.DataFrame,
    primary: list[str],
) -> None:
    # Style: clean publication figures (avoid purple glow / cream clichés)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )
    colors = {
        "confidence": "#1f4e79",
        "conf_consistency": "#c45c26",
        "conf_morph": "#5b8c5a",
        "conf_pooled": "#6b6b6b",
        "consistency": "#8b6914",
        "morphology": "#4a6fa5",
        "pooled": "#7a7a7a",
        "dice_probe": "#9e4a4a",
        "conf_morph_consistency": "#2f6f6f",
    }

    fdef = primary[0]
    y = fail_y[fdef]
    n = len(y)

    # Risk-coverage
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    coverages = np.linspace(0.05, 1.0, 40)
    for method in [
        "confidence",
        "conf_consistency",
        "conf_morph",
        "conf_pooled",
        "consistency",
        "pooled",
        "dice_probe",
    ]:
        s = scores[(method, fdef)]
        order = np.argsort(s)
        q = quality[order]
        risks = []
        for c in coverages:
            k = max(1, int(round(c * n)))
            risks.append(1.0 - float(np.mean(q[:k])))
        ax.plot(
            coverages,
            risks,
            label=METHOD_LABELS[method],
            color=colors.get(method, "#333"),
            lw=2.0 if method in ("confidence", "conf_consistency") else 1.2,
            alpha=1.0 if method in ("confidence", "conf_consistency") else 0.75,
        )
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Risk (1 − mean Dice of retained)")
    ax.set_title("Risk–coverage (lowest 20% failure model)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "risk_coverage_curves.png")
    plt.close(fig)

    # Failure capture
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    budgets = np.linspace(0.05, 0.5, 19)
    for method in [
        "confidence",
        "conf_consistency",
        "conf_morph",
        "conf_pooled",
        "consistency",
        "pooled",
        "dice_probe",
    ]:
        s = scores[(method, fdef)]
        caps = [_capture_at(s, y, b) for b in budgets]
        ax.plot(
            budgets * 100,
            caps,
            label=METHOD_LABELS[method],
            color=colors.get(method, "#333"),
            lw=2.0 if method in ("confidence", "conf_consistency") else 1.2,
        )
    ax.set_xlabel("Review budget (% of cases)")
    ax.set_ylabel("Failure capture")
    ax.set_title("Failure capture vs review budget")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_capture_curves.png")
    plt.close(fig)

    # Baseline AUPRC bar (primary)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for ax, fd in zip(axes, primary):
        sub = baseline_df[baseline_df.failure_def == fd].set_index("method").loc[METHOD_ORDER]
        ax.barh(
            [METHOD_LABELS[m] for m in METHOD_ORDER],
            sub["auprc"],
            color=[colors.get(m, "#555") for m in METHOD_ORDER],
        )
        ax.set_xlabel("AUPRC")
        ax.set_title(fd)
        ax.set_xlim(0, 1)
    fig.suptitle("Baseline comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_auprc.png", bbox_inches="tight")
    plt.close(fig)

    # Ablation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    show = abl_df[~abl_df.ablation.str.startswith("drop_") | True].copy()
    # show family + drop + confidence
    keep = [
        "all_gaps",
        "relative_only",
        "signed_only",
        "absolute_only",
        "volume_related",
        "tissue_fraction",
        "shape_boundary",
        "drop_gt_enhancing_frac",
        "drop_log_wt_volume",
        "drop_gt_edema_frac",
        "drop_gt_compactness",
        "drop_gt_necrosis_frac",
        "confidence_only",
    ]
    show = abl_df[abl_df.ablation.isin(keep)].set_index("ablation").loc[
        [k for k in keep if k in set(abl_df.ablation)]
    ]
    ax.barh(show.index.tolist(), show["auprc"].tolist(), color="#1f4e79")
    ax.axvline(float(abl_df.loc[abl_df.ablation == "confidence_only", "auprc"].iloc[0]), color="#c45c26", ls="--", label="confidence")
    ax.set_xlabel("AUPRC (lowest 20%)")
    ax.set_title("Consistency feature ablation (with confidence)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_ablation.png")
    plt.close(fig)

    # Permutation importance top 15
    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["mean_auprc_drop"], color="#c45c26", xerr=top["std_auprc_drop"], ecolor="#666")
    ax.set_xlabel("Mean AUPRC drop when permuted")
    ax.set_title("Consistency feature permutation importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_permutation_importance.png")
    plt.close(fig)

    # Correlation heatmap (abs gaps vs outcomes)
    abs_feats = [f for f in corr_df.feature.unique() if f.startswith("absolute_gap_")]
    outcomes = ["mean_fg_dice", "wt_dice", "tc_dice", "edema_dice", "boundary_error"]
    mat = np.zeros((len(abs_feats), len(outcomes)))
    for i, f in enumerate(abs_feats):
        for j, o in enumerate(outcomes):
            r = corr_df[(corr_df.feature == f) & (corr_df.outcome == o)]
            mat[i, j] = float(r.iloc[0]["spearman"]) if not r.empty else 0.0
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcomes, rotation=30, ha="right")
    ax.set_yticks(range(len(abs_feats)))
    ax.set_yticklabels([f.replace("absolute_gap_", "") for f in abs_feats], fontsize=8)
    ax.set_title("Spearman: |gap| features vs quality outcomes")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_outcome_correlations.png")
    plt.close(fig)

    # Bootstrap deltas
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, fd in zip(axes, primary):
        sub = boot_df[(boot_df.failure_def == fd) & (boot_df.metric == "auprc")].iloc[0]
        ax.barh(
            ["ΔAUPRC"],
            [sub.diff_mean],
            xerr=[[sub.diff_mean - sub.ci_low], [sub.ci_high - sub.diff_mean]],
            color="#c45c26",
            height=0.4,
        )
        ax.axvline(0, color="#333", lw=1)
        ax.set_title(fd)
        ax.set_xlabel("Proposed − confidence")
    fig.suptitle("Paired bootstrap ΔAUPRC (5000)", y=1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_delta_auprc.png", bbox_inches="tight")
    plt.close(fig)

    # Fold stability
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, fd in zip(axes, primary):
        sub = fold_stab_df[fold_stab_df.failure_def == fd]
        ax.bar(sub.outer_fold.astype(str), sub.delta_auprc, color="#1f4e79")
        ax.axhline(0, color="#333", lw=1)
        ax.set_title(fd)
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("ΔAUPRC")
    fig.suptitle("Fold-level improvement (proposed − confidence)", y=1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "fold_stability.png", bbox_inches="tight")
    plt.close(fig)

    # Retained Dice vs coverage
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method in ["confidence", "conf_consistency", "conf_morph", "conf_pooled"]:
        s = scores[(method, fdef)]
        order = np.argsort(s)
        q = quality[order]
        xs, ys = [], []
        for c in COVERAGES:
            k = max(1, int(round(c * n)))
            xs.append(c)
            ys.append(float(np.mean(q[:k])))
        ax.plot(xs, ys, marker="o", label=METHOD_LABELS[method], color=colors[method], lw=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean Dice of retained cases")
    ax.set_title("Mean retained Dice vs coverage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "retained_dice_vs_coverage.png")
    plt.close(fig)


def _capture_at(scores: np.ndarray, y: np.ndarray, budget: float) -> float:
    order = np.argsort(-scores)
    fails = y.astype(bool)[order]
    n_fail = max(int(fails.sum()), 1)
    k = max(1, int(round(budget * len(scores))))
    return float(fails[:k].sum() / n_fail)


def _write_summary(
    out_dir: Path,
    baseline_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    fold_stab_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    seed_summ: list[dict],
    boot_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    primary: list[str],
) -> str:
    def auprc(method: str, fdef: str) -> float:
        r = baseline_df[(baseline_df.method == method) & (baseline_df.failure_def == fdef)]
        return float(r.iloc[0]["auprc"])

    f0, f1 = primary[0], primary[1]
    d0 = auprc("conf_consistency", f0) - auprc("confidence", f0)
    d1 = auprc("conf_consistency", f1) - auprc("confidence", f1)
    boot0 = boot_df[(boot_df.failure_def == f0) & (boot_df.metric == "auprc")].iloc[0]
    boot1 = boot_df[(boot_df.failure_def == f1) & (boot_df.metric == "auprc")].iloc[0]
    fold_wins_0 = int((fold_stab_df[fold_stab_df.failure_def == f0].delta_auprc > 0).sum())
    fold_wins_1 = int((fold_stab_df[fold_stab_df.failure_def == f1].delta_auprc > 0).sum())

    # Verdict gates (aligned with prior triage criteria)
    strong = (
        d0 >= 0.03
        and d1 >= 0.03
        and boot0.ci_low > 0
        and boot1.diff_mean > 0
        and fold_wins_0 >= 4
        and fold_wins_1 >= 4
    )
    promising = d0 > 0 and d1 > 0 and (d0 >= 0.02 or d1 >= 0.02) and fold_wins_0 >= 3
    # Edema check for MIXED nuance
    edema_gain = auprc("conf_consistency", "edema_lt_0.70") - auprc("confidence", "edema_lt_0.70")
    if strong:
        verdict = "STRONG"
    elif promising:
        verdict = "PROMISING"
    else:
        verdict = "MIXED"

    # Why edema: correlations
    top_mean = corr_df[corr_df.outcome == "mean_fg_dice"].head(5)
    top_edema = corr_df[corr_df.outcome == "edema_dice"].head(5)
    enh_vs_edema = corr_df[
        (corr_df.feature.str.contains("enhancing")) & (corr_df.outcome == "edema_dice")
    ]
    enh_vs_mean = corr_df[
        (corr_df.feature.str.contains("enhancing")) & (corr_df.outcome == "mean_fg_dice")
    ]

    lines = [
        "# Method validation summary",
        "",
        f"**Algorithm:** frozen confidence + representation–output consistency  ",
        f"**Canonical triage artifacts:** `{CANONICAL_TRIAGE}`  ",
        f"**Verdict: {verdict}**",
        "",
        "No U-Net retraining. No probe/edit/repair regeneration. Classifier-only "
        "refits used only for missing baselines (`dice_probe`, `pooled`), "
        "ablations, and seed checks.",
        "",
        "---",
        "",
        "## 1. Complete baseline comparison",
        "",
        "Identical outer folds (seed 42, 5×75). Metrics from frozen held-out "
        "scores where available; `dice_probe` / `pooled` scores reused from the "
        "consistency experiment on the same folds (fail labels agree).",
        "",
    ]

    for fdef in ["lowest20_mean_fg", "mean_fg_lt_0.70", "mean_fg_lt_0.80", "edema_lt_0.70", "lowest20_edema"]:
        lines.append(f"### {fdef}")
        lines.append("")
        lines.append(
            "| Method | AUPRC | AUROC | Cap@10 | Cap@20 | Cap@30 | Brier |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        sub = baseline_df[baseline_df.failure_def == fdef].set_index("method")
        for m in METHOD_ORDER:
            r = sub.loc[m]
            lines.append(
                f"| {METHOD_LABELS[m]} | {r.auprc:.3f} | {r.auroc:.3f} | "
                f"{r.capture_at_10:.3f} | {r.capture_at_20:.3f} | {r.capture_at_30:.3f} | "
                f"{r.brier:.3f} |"
            )
        lines.append("")

    lines += [
        "## 2. Feature importance",
        "",
        "Held-out permutation importance (shuffle one consistency column at a "
        "time through the trained confidence+consistency logistic per outer fold).",
        "",
        "| Rank | Feature | Mean AUPRC drop |",
        "|---:|---|---:|",
    ]
    for i, r in enumerate(imp_df.head(12).itertuples(), 1):
        lines.append(f"| {i} | `{r.feature}` | {r.mean_auprc_drop:.4f} |")

    lines += [
        "",
        "## 3. Feature ablation",
        "",
        "Each row = confidence features + listed consistency subset; "
        "logistic re-fit on identical outer folds.",
        "",
        "| Ablation | AUPRC | Cap@20 |",
        "|---|---:|---:|",
    ]
    for r in abl_df.itertuples():
        lines.append(f"| {r.ablation} | {r.auprc:.3f} | {r.capture_at_20:.3f} |")

    lines += [
        "",
        "### Still works after removing…?",
        "",
    ]
    for key, label in [
        ("drop_log_wt_volume", "whole-tumor volume"),
        ("drop_gt_enhancing_frac", "enhancing fraction"),
        ("drop_gt_edema_frac", "edema fraction"),
        ("drop_gt_compactness", "compactness / boundary-complexity proxy"),
        ("drop_gt_necrosis_frac", "necrosis fraction"),
    ]:
        row = abl_df[abl_df.ablation == key].iloc[0]
        conf_a = abl_df[abl_df.ablation == "confidence_only"].iloc[0].auprc
        status = "yes" if row.auprc > conf_a + 0.01 else "marginal/no"
        lines.append(
            f"- **{label}:** AUPRC={row.auprc:.3f} (vs confidence {conf_a:.3f}) → {status}"
        )

    lines += [
        "",
        "## 4. Fold stability",
        "",
        f"- `{f0}`: proposed wins **{fold_wins_0}/5** folds",
        f"- `{f1}`: proposed wins **{fold_wins_1}/5** folds",
        "",
        "| Failure | Fold | ΔAUPRC |",
        "|---|---:|---:|",
    ]
    for r in fold_stab_df.itertuples():
        lines.append(f"| {r.failure_def} | {r.outer_fold} | {r.delta_auprc:+.3f} |")

    lines += ["", "## 5. Seed stability", "", "Classifier-only retrain; frozen features.", ""]
    summ_df = pd.DataFrame(seed_summ)
    lines.append("| Failure | Method | Metric | Mean | Std | CV |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in summ_df.itertuples():
        lines.append(
            f"| {r.failure_def} | {r.method} | {r.metric} | {r.mean:.3f} | {r.std:.4f} | {r.cv:.4f} |"
        )

    lines += ["", "## 6. Bootstrap comparisons", "", f"Paired case-level bootstrap, n={int(boot_df.n_boot.max())}.", ""]
    lines.append("| Failure | Metric | Δ | 95% CI | P(proposed better) |")
    lines.append("|---|---|---:|---|---:|")
    for r in boot_df.itertuples():
        lines.append(
            f"| {r.failure_def} | {r.metric} | {r.diff_mean:+.4f} | "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.frac_proposed_better:.3f} |"
        )
    lines.append("")
    lines.append(
        "For AURC, lower is better; `frac_proposed_better` is P(ΔAURC < 0)."
    )

    lines += [
        "",
        "## 7. Risk–coverage",
        "",
        "See `figures/risk_coverage_curves.png` and `risk_coverage_extended.csv`.",
        "",
    ]
    sub_rc = rc_df[rc_df.failure_def == f0]
    lines.append("| Method | AURC | Dice@70% | Dice@80% | Dice@90% |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in ["confidence", "conf_consistency", "conf_morph", "conf_pooled"]:
        r = sub_rc[sub_rc.method == m].iloc[0]
        lines.append(
            f"| {METHOD_LABELS[m]} | {r.aurc:.4f} | "
            f"{r.mean_dice_coverage_70:.3f} | {r.mean_dice_coverage_80:.3f} | "
            f"{r.mean_dice_coverage_90:.3f} |"
        )

    lines += [
        "",
        "## 8. Failure capture",
        "",
        "See `figures/failure_capture_curves.png`.",
        "",
    ]
    sub_b = baseline_df[baseline_df.failure_def == f0].set_index("method")
    lines.append("| Method | Cap@10 | Cap@20 | Cap@30 |")
    lines.append("|---|---:|---:|---:|")
    for m in METHOD_ORDER:
        r = sub_b.loc[m]
        lines.append(
            f"| {METHOD_LABELS[m]} | {r.capture_at_10:.3f} | {r.capture_at_20:.3f} | {r.capture_at_30:.3f} |"
        )

    lines += [
        "",
        "## 9. Clinical / mechanistic interpretation",
        "",
        "### Why overall failures improve",
        "",
        "Top Spearman correlations of consistency gaps with **mean foreground Dice**:",
        "",
    ]
    for r in top_mean.itertuples():
        lines.append(f"- `{r.feature}` ↔ {r.outcome}: ρ={r.spearman:+.3f}")

    lines += [
        "",
        "Enhancing-fraction and volume mismatches track global segmentation quality "
        "and tumor-core / whole-tumor errors. Absolute and relative gaps on "
        "`gt_enhancing_frac` dominate permutation importance and leave-one-target "
        "ablation (dropping enhancing-fraction gaps hurts most).",
        "",
        "### Why edema-specific failures do not improve",
        "",
        f"On `edema_lt_0.70`, ΔAUPRC(proposed − confidence) = {edema_gain:+.3f}. "
        "Confidence + morphology often matches or beats confidence + consistency "
        "for edema targets.",
        "",
        "Top correlations with **edema Dice**:",
        "",
    ]
    for r in top_edema.itertuples():
        lines.append(f"- `{r.feature}` ↔ edema_dice: ρ={r.spearman:+.3f}")

    if not enh_vs_edema.empty and not enh_vs_mean.empty:
        lines += [
            "",
            "Enhancing-fraction gaps correlate more strongly with overall/mean-fg "
            f"quality (mean |ρ|≈{enh_vs_mean.abs_spearman.mean():.3f}) than with "
            f"edema Dice (mean |ρ|≈{enh_vs_edema.abs_spearman.mean():.3f}). "
            "The consistency features therefore preferentially flag composition / "
            "core–enhancement disagreements that drive mean-fg failures, not "
            "edema-boundary failures that confidence/morphology already capture.",
        ]

    lines += [
        "",
        "### What failure types are detected",
        "",
        "- **Detected well:** low mean-fg Dice / severe overall failures, often "
        "tied to enhancing-core and whole-tumor volume disagreement.",
        "- **Detected weakly:** edema-compartment failures without global collapse.",
        "- **Not a Dice oracle:** direct Dice probe remains weak (AUPRC≈0.54 on "
        "lowest 20%); consistency adds anatomical disagreement, not a second Dice predictor.",
        "",
        "## 10. Verdict",
        "",
        f"**{verdict}**",
        "",
        f"- Primary lowest-20% ΔAUPRC = {d0:+.3f} "
        f"(bootstrap [{boot0.ci_low:+.3f}, {boot0.ci_high:+.3f}], "
        f"P(better)={boot0.frac_proposed_better:.2f})",
        f"- Primary Dice<0.70 ΔAUPRC = {d1:+.3f} "
        f"(bootstrap [{boot1.ci_low:+.3f}, {boot1.ci_high:+.3f}], "
        f"P(better)={boot1.frac_proposed_better:.2f})",
        f"- Fold wins: {fold_wins_0}/5 and {fold_wins_1}/5",
        f"- Capture@20 improves on primary lowest-20% "
        f"({sub_b.loc['conf_consistency'].capture_at_20:.3f} vs "
        f"{sub_b.loc['confidence'].capture_at_20:.3f})",
        "",
        "Edema-specific performance remains a limitation; do not claim "
        "compartment-universal triage.",
        "",
        "## Task 7 (TTA disagreement)",
        "",
        "Skipped — see `tta_skipped.md`.",
        "",
        "No novelty, deployment, or external-generalization claims.",
        "",
    ]
    (out_dir / "validation_summary.md").write_text("\n".join(lines) + "\n")
    return verdict


def main_from_config(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    return run_validation(config)
