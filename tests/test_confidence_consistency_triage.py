"""Leakage-safety tests for confidence + consistency triage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.confidence_consistency_triage import load_config
from src.analysis.layer_aware_latent_risk import (
    GT_LEAKED_COLS,
    assert_no_gt_leak_features,
    load_outer_folds,
    verify_folds_match_seed,
)

CONFIG = Path("configs/confidence_consistency_triage.yaml")


def test_config_loads():
    cfg = load_config(CONFIG)
    assert cfg["seed"] == 42
    assert cfg["n_bootstrap"] >= 5000
    assert "lowest20_mean_fg" in cfg["primary_targets"]
    assert "mean_fg_lt_0.70" in cfg["primary_targets"]


def test_folds_reusable_and_partition():
    cfg = load_config(CONFIG)
    folds = load_outer_folds(Path(cfg["paths"]["fold_assignments"]))
    assert len(folds) == 5
    assert verify_folds_match_seed(Path(cfg["paths"]["fold_assignments"]), 375, 42)
    counts = np.zeros(375, dtype=int)
    for p in folds.values():
        counts[p["test"]] += 1
    assert counts.tolist() == [1] * 375


def test_confidence_cols_have_no_gt_leak():
    cfg = load_config(CONFIG)
    assert_no_gt_leak_features(cfg["confidence_cols"])
    for c in cfg["confidence_cols"]:
        assert c not in GT_LEAKED_COLS
        assert c.startswith("conf_")


def test_morphology_from_pred_not_gt():
    cfg = load_config(CONFIG)
    for c in cfg["morphology_cols"]:
        assert c.startswith(("pred_", "out_")) or c in {"n_components", "surface_to_volume"}


def test_identical_outer_folds_path():
    cfg = load_config(CONFIG)
    assert "outputs_consistency_failure_detection/fold_assignments.csv" in str(
        cfg["paths"]["fold_assignments"]
    )


def test_confidence_csv_exact_match_flag():
    cfg = load_config(CONFIG)
    conf = pd.read_csv(cfg["paths"]["confidence_features"])
    assert conf["mask_exact_match"].astype(str).isin(["True", "true", "1"]).all()


def test_pca_not_required_for_compact_consistency():
    """Primary method must not depend on PCA (consistency is compact)."""
    cfg = load_config(CONFIG)
    assert "pca" not in cfg or not cfg.get("pca")


def test_resolve_output_dir_avoids_overwrite(tmp_path, monkeypatch):
    from src.analysis.confidence_consistency_triage import resolve_output_dir

    existing = tmp_path / "outputs_confidence_consistency_triage"
    existing.mkdir()
    (existing / "heldout_predictions.csv").write_text("x\n")
    cfg = {
        "paths": {"output_dir": str(existing)},
        "overwrite_output": False,
    }
    out = resolve_output_dir(cfg)
    assert out != existing
    assert out.name.startswith("outputs_confidence_consistency_triage_")
    assert out.exists()


def test_calibration_covers_primary_targets_in_config():
    cfg = load_config(CONFIG)
    assert set(cfg["primary_targets"]) <= set(cfg["failure_defs"])
    assert cfg.get("overwrite_output") is False
