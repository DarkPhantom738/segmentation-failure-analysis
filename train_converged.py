#!/usr/bin/env python3
"""Train a 3D U-Net to convergence under a fixed patient-split protocol."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.converged_splits import (
    prepare_or_load_shared_splits,
    seed_output_dirname,
)
from src.training.converged_trainer import ConvergedTrainer
from src.utils.seed import set_seed

MODEL_SEEDS = [42, 123, 2026, 31415, 271828]


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def historical_final_case_ids(repo_root: Path) -> list[str] | None:
    table = repo_root / "outputs_10hour" / "failure_tables" / "failure_metrics.csv"
    if not table.exists():
        return None
    return sorted(pd.read_csv(table)["case_id"].astype(str).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one model seed to convergence using shared patient splits. "
            "Does not run downstream representation or confidence analyses."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/converged_unet.yaml"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=f"Model seed. Supported: {MODEL_SEEDS}",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs_converged/seed_*/checkpoints/checkpoint_latest.pt",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override training.max_epochs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Two-epoch sanity run on a tiny train/development subset. "
            "Shared split JSON files are still written/validated; only loaders shrink."
        ),
    )
    parser.add_argument(
        "--rewrite-splits",
        action="store_true",
        help="Rebuild shared split JSON files even if they already exist.",
    )
    return parser.parse_args()


def apply_smoke_overrides(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["training"]["max_epochs"] = 2
    cfg["training"]["validate_every"] = 1
    cfg["training"]["early_stopping_patience"] = 10
    return cfg


def main() -> None:
    args = parse_args()
    if args.seed not in MODEL_SEEDS:
        raise SystemExit(
            f"Unsupported model seed {args.seed}. Use one of {MODEL_SEEDS}."
        )

    config = load_config(args.config)
    config["seed"] = int(args.seed)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = int(args.max_epochs)
    if args.smoke_test:
        config = apply_smoke_overrides(config)

    set_seed(int(args.seed))
    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Model seed: {args.seed} → {seed_output_dirname(args.seed)}")

    repo_root = Path.cwd()
    historical = historical_final_case_ids(repo_root)
    splits = prepare_or_load_shared_splits(
        config,
        historical_final_ids=historical,
        rewrite=args.rewrite_splits,
    )
    print(
        f"Splits — train: {len(splits['train_cases'])}, "
        f"development: {len(splits['development_cases'])}, "
        f"final_evaluation: {len(splits['final_evaluation_cases'])}"
    )
    if historical is not None:
        print("Final-evaluation IDs match historical failure_metrics.csv cohort.")

    train_cases = list(splits["train_cases"])
    development_cases = list(splits["development_cases"])
    if args.smoke_test:
        train_cases = train_cases[:4]
        development_cases = development_cases[:2]
        print(
            f"Smoke-test subsets — train: {len(train_cases)}, "
            f"development: {len(development_cases)}"
        )

    trainer = ConvergedTrainer(
        config=config,
        model_seed=int(args.seed),
        train_cases=train_cases,
        development_cases=development_cases,
        final_evaluation_cases=splits["final_evaluation_cases"],
        device=device,
        smoke_test=args.smoke_test,
    )
    if args.resume:
        trainer.resume_from_latest()

    state = trainer.train()
    print(
        f"Done seed={args.seed}: best_epoch={state.best_epoch}, "
        f"stopping_epoch={state.epoch}, "
        f"best_dev_mean_fg_dice={state.best_dev_mean_fg_dice:.4f}, "
        f"final_lr={state.final_learning_rate:.6g}, "
        f"runtime_s={state.runtime_seconds:.1f}, "
        f"reason={state.stopping_reason}"
    )


if __name__ == "__main__":
    main()
