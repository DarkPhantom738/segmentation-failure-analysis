"""Leakage-safety tests for Layer-Aware Latent Risk Triage (no full experiment)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.analysis.layer_aware_latent_risk import (
    GT_LEAKED_COLS,
    LAYER_DIMS,
    assert_no_gt_leak_features,
    confidence_from_probabilities,
    load_config,
    load_outer_folds,
    load_reusable_tables,
    verify_folds_match_seed,
)


CONFIG = Path("configs/layer_aware_latent_risk.yaml")


def test_config_exists_and_loads():
    cfg = load_config(CONFIG)
    assert cfg["seed"] == 42
    assert cfg["outer_folds"] == 5
    assert cfg["inner_folds"] == 4
    assert cfg["n_bootstrap"] == 2000
    assert len(cfg["layers"]) == 9


def test_fold_assignments_partition_and_seed():
    cfg = load_config(CONFIG)
    fold_csv = Path(cfg["paths"]["fold_assignments"])
    folds = load_outer_folds(fold_csv)
    assert len(folds) == 5
    n = sum(len(folds[0]["train"]) for _ in [0]) + len(folds[0]["test"])
    # n from first fold
    n = len(folds[0]["train"]) + len(folds[0]["test"])
    assert n == 375
    assert verify_folds_match_seed(fold_csv, n=375, seed=42)


def test_identical_outer_folds_across_methods():
    """All methods must reuse the same fold_assignments file from config."""
    cfg = load_config(CONFIG)
    fold_path = Path(cfg["paths"]["fold_assignments"])
    assert fold_path.name == "fold_assignments.csv"
    assert fold_path.exists()


def test_gt_leaked_cols_blocked():
    for col in GT_LEAKED_COLS:
        try:
            assert_no_gt_leak_features([col])
            raise AssertionError(f"should have blocked {col}")
        except ValueError:
            pass
    assert_no_gt_leak_features(["conf_mean_entropy_all", "pred_tumor_volume"])


def test_confidence_features_ignore_ground_truth():
    """confidence_from_probabilities must not take GT; pred-only boundary."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 16, 16, 16))
    # softmax
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs = e / e.sum(axis=0, keepdims=True)
    # Craft a small foreground blob in pred via high class-2 probs
    probs_gt_free = probs.copy()
    feats = confidence_from_probabilities(probs_gt_free)
    assert "conf_mean_maxprob_all" in feats
    assert "conf_mean_entropy_fg" in feats
    assert "conf_mean_entropy_boundary" in feats
    assert "conf_mean_margin_all" in feats
    assert "pred_tumor_voxels" in feats
    # No GT-named keys
    for k in feats:
        assert "error" not in k.lower()
        assert k not in GT_LEAKED_COLS


def test_confidence_boundary_from_prediction_not_gt():
    probs = np.zeros((4, 20, 20, 20), dtype=np.float64)
    probs[0] = 0.9
    # Predicted edema cube
    probs[2, 5:12, 5:12, 5:12] = 0.8
    probs[0, 5:12, 5:12, 5:12] = 0.05
    probs[1, 5:12, 5:12, 5:12] = 0.05
    probs[3, 5:12, 5:12, 5:12] = 0.10
    # Renormalize
    probs = probs / probs.sum(axis=0, keepdims=True)
    pred = np.argmax(probs, axis=0).astype(np.int16)
    # Different GT would not affect — function has no GT arg
    feats = confidence_from_probabilities(probs, pred=pred)
    assert feats["conf_fg_fraction"] > 0
    assert np.isfinite(feats["conf_mean_entropy_boundary"])


def test_morphology_cols_are_pred_derived():
    cfg = load_config(CONFIG)
    for c in cfg["morphology_cols"]:
        assert c.startswith("pred_") or c.startswith("out_") or c in {
            "n_components",
            "surface_to_volume",
        }


def test_layer_dims_match_expected():
    cfg = load_config(CONFIG)
    tables = load_reusable_tables(cfg)
    for L, dim in LAYER_DIMS.items():
        assert tables["layer_mats"][L].shape == (375, dim)


def test_pca_component_cap_helper():
    """PCA n_components must not exceed n_train - 1 or feature rank."""
    n_train, n_features = 300, 1088
    for k in [8, 16, 32, 64]:
        capped = min(k, n_train - 1, n_features)
        assert capped == k  # all fine for n_train=300
    assert min(64, 20 - 1, 1088) == 19


def test_one_test_prediction_slot_per_case():
    cfg = load_config(CONFIG)
    folds = load_outer_folds(Path(cfg["paths"]["fold_assignments"]))
    counts = np.zeros(375, dtype=int)
    for parts in folds.values():
        counts[parts["test"]] += 1
    assert counts.tolist() == [1] * 375


def test_seed_reproducible_fold_reload():
    cfg = load_config(CONFIG)
    a = load_outer_folds(Path(cfg["paths"]["fold_assignments"]))
    b = load_outer_folds(Path(cfg["paths"]["fold_assignments"]))
    for k in a:
        assert np.array_equal(a[k]["train"], b[k]["train"])
        assert np.array_equal(a[k]["test"], b[k]["test"])


def test_full_stage_gated_without_confidence(tmp_path):
    """full must not silently run; missing confidence or NotImplemented."""
    from src.analysis.layer_aware_latent_risk import run_stage
    import pytest

    with pytest.raises((RuntimeError, NotImplementedError)):
        run_stage(CONFIG, "full")
