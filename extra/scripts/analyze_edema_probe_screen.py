#!/usr/bin/env python3
"""Minimal screening: decoder1 edema probe vs unit-norm random controls (30 cases)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.edema_probe_screen import run_edema_probe_screen
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen whether decoder1 edema probe edits beat 3 random directions "
            "on 30 Dice-stratified validation cases."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ten_hour.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument("--editing-results", type=Path, default=Path("outputs_10hour/representation_editing/editing_results.csv"))
    parser.add_argument("--direction-path", type=Path, default=Path("outputs_10hour/semantic_directions/directions/decoder1__gt_edema_frac.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_10hour/edema_probe_screen"))
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)

    ensure_dir(args.output_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    out = run_edema_probe_screen(
        config=config,
        checkpoint_path=args.checkpoint,
        failure_table_path=args.failure_table,
        output_dir=args.output_dir,
        device=device,
        editing_results_path=args.editing_results,
        direction_path=args.direction_path,
        random_seed=args.random_seed,
        overlap=args.overlap,
    )
    print(f"Wrote screening study to {args.output_dir}")
    print(out["summary"].to_string(index=False))
    print(out["conclusion"])


if __name__ == "__main__":
    main()
