#!/usr/bin/env python3
"""Entry point for Failure Cartography training and artifact export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.brats_dataset import (
    build_dataloaders,
    discover_brats_cases,
    limit_cases,
    split_cases,
)
from src.models.unet3d import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3D U-Net on BraTS and export validation embeddings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and only export validation artifacts from a checkpoint.",
    )
    parser.add_argument(
        "--export-uncertainty",
        action="store_true",
        help="Export TTA uncertainty maps for validation cases from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path used with --export-only.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    set_seed(config["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_root = Path(config["data"]["root"])
    all_cases = limit_cases(
        discover_brats_cases(data_root),
        config["data"].get("max_cases"),
    )
    train_cases, val_cases = split_cases(
        all_cases,
        val_fraction=config["data"]["val_fraction"],
        seed=config["seed"],
    )
    print(f"Cases — train: {len(train_cases)}, val: {len(val_cases)}")

    train_loader, val_loader = build_dataloaders(config, train_cases, val_cases)
    model = build_model(config)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    if args.export_only and args.export_uncertainty:
        raise ValueError("Use only one of --export-only or --export-uncertainty.")

    if args.export_only:
        checkpoint_path = args.checkpoint or Path(
            config["output"]["checkpoint_dir"]
        ) / "checkpoint_latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = int(checkpoint.get("epoch", 0))
        mean_dice = trainer.export_validation_artifacts(epoch)
        print(f"Exported validation artifacts — mean whole-tumor Dice: {mean_dice:.4f}")
        return

    if args.export_uncertainty:
        if not config.get("uncertainty", {}).get("enabled", False):
            print("Warning: uncertainty.enabled is false in config; proceeding with export.")
        checkpoint_path = args.checkpoint or Path(
            config["output"]["checkpoint_dir"]
        ) / "checkpoint_latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = int(checkpoint.get("epoch", 0))
        mean_dice = trainer.export_uncertainty_artifacts(epoch)
        print(f"Exported TTA uncertainty artifacts — mean whole-tumor Dice: {mean_dice:.4f}")
        return

    trainer.train()


if __name__ == "__main__":
    main()
