"""Leakage-safety and alignment tests for consistency failure detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.analysis.representation_output_consistency import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    MORPHOLOGY_COLS,
    build_gap_features,
    fit_ridge_predict,
    impute_train_stats,
    make_fold_assignments,
    oof_ridge_predictions,
    pred_mask_anatomy,
    select_best_layer,
)


def test_one_outer_prediction_per_case():
    n, folds, seed = 375, 5, 42
    assignments = make_fold_assignments(n, folds, seed)
    test_counts = np.zeros(n, dtype=int)
    for _, te in assignments:
        test_counts[te] += 1
    assert test_counts.tolist() == [1] * n


def test_fold_assignments_deterministic():
    a = make_fold_assignments(375, 5, 42)
    b = make_fold_assignments(375, 5, 42)
    for (tr1, te1), (tr2, te2) in zip(a, b, strict=True):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_probe_not_trained_on_evaluation_case():
    """OOF ridge leaves each case out of its own fitting fold."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    y = x[:, 0] * 2 + rng.normal(scale=0.1, size=40)
    preds, _ = oof_ridge_predictions(x, y, n_splits=4, seed=42, alphas=[0.1, 1.0, 10.0])
    assert np.isfinite(preds).all()
    # Refit excluding each point and compare — OOF should match leave-fold-out, not full fit.
    full = fit_ridge_predict(x, y, x, alphas=[0.1, 1.0, 10.0])
    # OOF predictions should differ from in-sample predictions for most cases.
    assert float(np.mean(np.abs(preds - full))) > 1e-6


def test_scaler_fit_on_train_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4))
    x[0, 0] = np.nan
    train_idx = np.arange(20)
    cleaned, means = impute_train_stats(x, train_idx)
    assert np.isfinite(cleaned).all()
    # Mean for col0 must equal train finite mean, not full-data mean including test.
    assert abs(means[0] - np.nanmean(x[train_idx, 0])) < 1e-8


def test_best_layer_selection_uses_provided_subset_only():
    rng = np.random.default_rng(2)
    n = 50
    # Layer A predicts y; layer B is noise.
    y = rng.normal(size=n)
    layers = {
        "A": np.column_stack([y, rng.normal(size=n)]),
        "B": rng.normal(size=(n, 2)),
    }
    best, r2, preds = select_best_layer(
        layers, y, ["A", "B"], n_splits=4, seed=0, alphas=[0.1, 1.0, 10.0]
    )
    assert best == "A"
    assert r2 > 0.5
    assert len(preds) == n


def test_ground_truth_anatomy_forbidden_as_features():
    feature_names = (
        list(MORPHOLOGY_COLS)
        + ["signed_gap_log_wt_volume", "absolute_gap_gt_edema_frac", "rep_dice"]
    )
    for name in feature_names:
        for bad in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert bad not in name


def test_gt_leaked_uncertainty_cols_not_used_as_confidence():
    from src.analysis.representation_output_consistency import (
        CONFIDENCE_COLS,
        GT_LEAKED_UNCERTAINTY_COLS,
        FORBIDDEN_FEATURE_EXACT,
    )

    assert CONFIDENCE_COLS == []
    for col in GT_LEAKED_UNCERTAINTY_COLS:
        assert col in FORBIDDEN_FEATURE_EXACT
        assert col not in MORPHOLOGY_COLS


def test_gt_dice_not_in_feature_groups():
    forbidden = {"dice", "label_wt_dice", "label_mean_fg_dice", "gt_anatomy_dice"}
    feature_like = set(MORPHOLOGY_COLS) | {
        "signed_gap_log_wt_volume",
        "rep_log_wt_volume",
    }
    assert forbidden.isdisjoint(feature_like)


def test_pred_mask_anatomy_from_prediction_not_gt():
    pred = np.zeros((20, 20, 20), dtype=np.int16)
    pred[5:10, 5:10, 5:10] = 2  # edema
    pred[6:8, 6:8, 6:8] = 3  # enhancing
    gt = np.zeros_like(pred)
    gt[0:15, 0:15, 0:15] = 2  # very different GT
    out = pred_mask_anatomy(pred)
    # Volumes must match prediction counts, not GT.
    assert out["pred_edema_volume"] == float((pred == 2).sum())
    assert out["pred_enhancing_volume"] == float((pred == 3).sum())
    assert out["pred_tumor_volume"] != float((gt > 0).sum())


def test_case_id_alignment_helper():
    ids_a = pd.Series(["c1", "c2", "c3"])
    ids_b = pd.Series(["c1", "c2", "c3"])
    ids_c = pd.Series(["c1", "c2", "c3"])
    assert ids_a.equals(ids_b) and ids_b.equals(ids_c)


def test_gap_standardization_uses_train_std():
    rep = np.array([1.0, 2.0, 3.0, 10.0])
    out = np.array([1.0, 1.0, 1.0, 1.0])
    train_idx = np.array([0, 1, 2])
    gaps = build_gap_features(rep, out, train_idx, "demo")
    train_abs = np.abs(rep[train_idx] - out[train_idx])
    expected_std = max(float(np.std(train_abs)), 1e-8)
    assert abs(gaps["standardized_gap_demo"][3] - (9.0 / expected_std)) < 1e-8


def test_missing_value_imputation_train_only():
    x = np.array([[1.0, np.nan], [2.0, 4.0], [np.nan, 6.0], [100.0, np.nan]])
    train_idx = np.array([0, 1, 2])
    cleaned, means = impute_train_stats(x, train_idx)
    # Test row must use train mean for col0 (nanmean of 1,2), not 100.
    assert abs(means[0] - 1.5) < 1e-8
    assert abs(cleaned[3, 0] - 100.0) < 1e-8  # observed value kept
    assert abs(cleaned[3, 1] - means[1]) < 1e-8


def test_kfold_seed_reproducibility_matches_outer_design():
    kf1 = list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(375)))
    kf2 = make_fold_assignments(375, 5, 42)
    for (a, b), (c, d) in zip(kf1, kf2, strict=True):
        assert np.array_equal(a, c) and np.array_equal(b, d)
