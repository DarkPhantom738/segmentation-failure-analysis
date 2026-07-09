#!/usr/bin/env python3
"""Analyze anatomical recoverability and quality estimation by U-Net layer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.layer_analysis import analyze_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe where anatomical and failure-relevant information lives across "
            "U-Net depth using exported per-layer embeddings."
        )
    )
    parser.add_argument(
        "--layer-index",
        type=Path,
        required=True,
        help="Path to layer_embedding_index.csv.",
    )
    parser.add_argument(
        "--failure-table",
        type=Path,
        required=True,
        help="Path to failure_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for layer_analysis outputs.",
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.80,
        help="Dice threshold for bad-case detection.",
    )
    parser.add_argument(
        "--bad-quantile",
        type=float,
        default=0.25,
        help="Quantile for bottom-Dice bad-case mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze_layers(
        layer_index_path=args.layer_index,
        failure_table_path=args.failure_table,
        output_dir=args.output_dir,
        dice_threshold=args.dice_threshold,
        bad_quantile=args.bad_quantile,
    )
    rec = results["recoverability"]
    qual = results["quality"]
    print(f"Wrote analysis to {args.output_dir}")
    print(f"  recoverability rows: {len(rec)}")
    print(f"  quality rows: {len(qual)}")
    print(f"  report: {args.output_dir / 'layer_analysis_report.md'}")


if __name__ == "__main__":
    main()
