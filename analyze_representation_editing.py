#!/usr/bin/env python3
"""Run causal representation editing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.analysis.representation_editing import run_representation_edits
from src.analysis.semantic_directions import DEFAULT_ALPHAS
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply semantic activation edits A' = A + alpha*Delta and evaluate "
            "selectivity of downstream anatomical properties."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/five_epoch_533.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument(
        "--directions-dir",
        type=Path,
        required=True,
        help="Directory from learn_semantic_directions.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-directions", type=int, default=None)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="Edit strengths (default: -2 -1 -0.5 0.5 1 2)",
    )
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

    results = run_representation_edits(
        config=config,
        checkpoint_path=args.checkpoint,
        failure_table_path=args.failure_table,
        directions_dir=args.directions_dir,
        output_dir=args.output_dir,
        device=device,
        alphas=tuple(args.alphas),
        overlap=args.overlap,
        max_cases=args.max_cases,
        max_directions=args.max_directions,
    )
    print(f"Wrote representation editing study to {args.output_dir}")
    print(f"  results rows: {len(results['results'])}")
    print(f"  report: {args.output_dir / 'representation_editing_report.md'}")


if __name__ == "__main__":
    main()
