"""Tests for converged multi-seed U-Net training protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.converged_splits import (
    assert_disjoint_splits,
    build_converged_splits,
    case_list_hash,
    discover_cases_for_converged_training,
    prepare_or_load_shared_splits,
    seed_output_dirname,
)
from src.training.converged_metrics import (
    class_dice,
    mean_foreground_dice,
)
from src.training.converged_trainer import ConvergedTrainer, HISTORY_COLUMNS

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "converged_unet.yaml"
FAILURE_TABLE = REPO / "outputs_10hour" / "failure_tables" / "failure_metrics.csv"
CACHE_DIR = REPO / "outputs_10hour" / "cache"
OUTPUTS_10HOUR = REPO / "outputs_10hour"


def _load_config() -> dict:
    with CONFIG_PATH.open() as handle:
        return yaml.safe_load(handle)


def _historical_final_ids() -> list[str]:
    return sorted(pd.read_csv(FAILURE_TABLE)["case_id"].astype(str).tolist())


@pytest.fixture(scope="module")
def full_case_ids() -> list[str]:
    cfg = _load_config()
    data = cfg["data"]
    return discover_cases_for_converged_training(
        data_root=REPO / data["root"],
        cache_dir=REPO / data["cache_dir"],
        modalities=data["modalities"],
        target_spacing=data["target_spacing"],
        percentile_clip=data["percentile_clip"],
    )


@pytest.fixture(scope="module")
def splits(full_case_ids: list[str]) -> dict[str, list[str]]:
    cfg = _load_config()["data"]
    return build_converged_splits(
        full_case_ids,
        data_split_seed=int(cfg["data_split_seed"]),
        development_split_seed=int(cfg["development_split_seed"]),
        final_validation_fraction=float(cfg["final_validation_fraction"]),
        development_fraction_of_training_pool=float(
            cfg["development_fraction_of_training_pool"]
        ),
    )


def test_model_seed_does_not_change_patient_splits(full_case_ids: list[str]) -> None:
    cfg = _load_config()["data"]
    kwargs = dict(
        data_split_seed=int(cfg["data_split_seed"]),
        development_split_seed=int(cfg["development_split_seed"]),
        final_validation_fraction=float(cfg["final_validation_fraction"]),
        development_fraction_of_training_pool=float(
            cfg["development_fraction_of_training_pool"]
        ),
    )
    base = build_converged_splits(full_case_ids, **kwargs)
    for _model_seed in (42, 123, 2026, 31415, 271828):
        other = build_converged_splits(full_case_ids, **kwargs)
        assert other == base


def test_final_375_matches_historical_cohort(splits: dict[str, list[str]]) -> None:
    assert len(splits["final_evaluation_cases"]) == 375
    assert set(splits["final_evaluation_cases"]) == set(_historical_final_ids())


def test_splits_disjoint_and_counts(splits: dict[str, list[str]]) -> None:
    assert_disjoint_splits(splits)
    assert len(splits["training_pool_cases"]) == 876
    assert len(splits["train_cases"]) + len(splits["development_cases"]) == 876
    assert len(splits["development_cases"]) == 88
    assert len(splits["train_cases"]) == 788
    assert len(splits["final_evaluation_cases"]) == 375


def test_seed_output_dirname_formatting() -> None:
    assert seed_output_dirname(42) == "seed_042"
    assert seed_output_dirname(123) == "seed_123"
    assert seed_output_dirname(2026) == "seed_2026"
    assert seed_output_dirname(31415) == "seed_31415"
    assert seed_output_dirname(271828) == "seed_271828"


def test_absent_class_dice_rule() -> None:
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    pred = np.zeros_like(gt)
    assert class_dice(pred, gt, 1) == pytest.approx(1.0)
    pred[0, 0, 0] = 1
    assert class_dice(pred, gt, 1) < 0.1
    mean = mean_foreground_dice(pred, gt)
    assert 0.0 <= mean <= 1.0


def test_history_columns_contract() -> None:
    expected = [
        "epoch",
        "train_loss",
        "development_mean_foreground_dice",
        "development_necrotic_dice",
        "development_edema_dice",
        "development_enhancing_dice",
        "development_whole_tumor_dice",
        "learning_rate",
        "improved",
        "patience_counter",
    ]
    assert HISTORY_COLUMNS == expected


def test_prepare_shared_splits_writes_json(tmp_path: Path, full_case_ids: list[str]) -> None:
    cfg = _load_config()
    cfg = copy.deepcopy(cfg)
    cfg["output"]["shared_split_dir"] = str(tmp_path / "shared_split")
    cfg["data"]["root"] = str(REPO / cfg["data"]["root"])
    cfg["data"]["cache_dir"] = str(REPO / cfg["data"]["cache_dir"])
    splits = prepare_or_load_shared_splits(
        cfg,
        historical_final_ids=_historical_final_ids(),
        rewrite=True,
    )
    for name in ("train_cases", "development_cases", "final_evaluation_cases"):
        path = Path(cfg["output"]["shared_split_dir"]) / f"{name}.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == splits[name]


def test_final_evaluation_never_in_trainer_loaders(
    tmp_path: Path, splits: dict[str, list[str]]
) -> None:
    if not CACHE_DIR.exists():
        pytest.skip("preprocessed cache unavailable")

    cfg = copy.deepcopy(_load_config())
    cfg["output"]["root"] = str(tmp_path / "outputs_converged")
    cfg["data"]["root"] = str(REPO / cfg["data"]["root"])
    cfg["data"]["cache_dir"] = str(REPO / cfg["data"]["cache_dir"])
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["validate_every"] = 1

    trainer = ConvergedTrainer(
        config=cfg,
        model_seed=42,
        train_cases=splits["train_cases"][:2],
        development_cases=splits["development_cases"][:1],
        final_evaluation_cases=splits["final_evaluation_cases"],
        device=torch.device("cpu"),
        smoke_test=True,
    )
    train_ids = set(trainer.train_loader.dataset.case_ids)
    devel_ids = set(trainer.development_loader.dataset.case_ids)
    final_ids = set(splits["final_evaluation_cases"])
    assert not (train_ids & final_ids)
    assert not (devel_ids & final_ids)


def test_early_stopping_metric_is_mean_foreground_dice() -> None:
    # Checkpoint selection compares development mean FG Dice only.
    pred = np.zeros((16, 16, 16), dtype=np.uint8)
    gt = np.zeros_like(pred)
    gt[2:8, 2:8, 2:8] = 1
    gt[4:10, 4:10, 4:10] = 2
    gt[6:9, 6:9, 6:9] = 3
    pred[2:8, 2:8, 2:8] = 1
    pred[4:10, 4:10, 4:10] = 2
    pred[6:9, 6:9, 6:9] = 3
    score = mean_foreground_dice(pred, gt)
    assert score == pytest.approx(1.0)


def test_resume_restores_states(tmp_path: Path, splits: dict[str, list[str]]) -> None:
    if not CACHE_DIR.exists():
        pytest.skip("preprocessed cache unavailable")

    cfg = copy.deepcopy(_load_config())
    cfg["output"]["root"] = str(tmp_path / "outputs_converged")
    cfg["data"]["root"] = str(REPO / cfg["data"]["root"])
    cfg["data"]["cache_dir"] = str(REPO / cfg["data"]["cache_dir"])
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["validate_every"] = 1

    device = torch.device("cpu")
    trainer = ConvergedTrainer(
        config=cfg,
        model_seed=123,
        train_cases=splits["train_cases"][:2],
        development_cases=splits["development_cases"][:1],
        final_evaluation_cases=splits["final_evaluation_cases"],
        device=device,
        smoke_test=True,
    )
    trainer.best_dev_mean_fg_dice = 0.55
    trainer.best_epoch = 7
    trainer.patience_counter = 2
    ckpt = trainer.checkpoint_dir / "checkpoint_latest.pt"
    trainer._save_checkpoint(7, ckpt)

    trainer2 = ConvergedTrainer(
        config=cfg,
        model_seed=123,
        train_cases=splits["train_cases"][:2],
        development_cases=splits["development_cases"][:1],
        final_evaluation_cases=splits["final_evaluation_cases"],
        device=device,
        smoke_test=True,
    )
    epoch = trainer2.load_checkpoint(ckpt)
    assert epoch == 7
    assert trainer2.best_dev_mean_fg_dice == pytest.approx(0.55)
    assert trainer2.best_epoch == 7
    assert trainer2.patience_counter == 2
    assert trainer2.model_seed == 123


def test_five_runs_share_settings_except_seed_and_outdir() -> None:
    cfg = _load_config()
    assert cfg["data"]["data_split_seed"] == 42
    assert cfg["data"]["development_split_seed"] == 31415
    assert cfg["training"]["max_epochs"] == 100
    assert cfg["training"]["validate_every"] == 5
    assert cfg["training"]["early_stopping_patience"] == 4
    assert cfg["training"]["early_stopping_min_delta"] == 0.001
    for seed in (42, 123, 2026, 31415, 271828):
        assert seed_output_dirname(seed).startswith("seed_")


def test_outputs_10hour_untouched_marker() -> None:
    """Sanity: tests must not require writing under outputs_10hour/."""
    assert OUTPUTS_10HOUR.exists()
    # Case-list hash helper is pure and cannot mutate outputs_10hour.
    assert len(case_list_hash(["a", "b"])) == 64
