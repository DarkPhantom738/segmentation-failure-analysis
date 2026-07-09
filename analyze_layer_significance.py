#!/usr/bin/env python3
"""Statistical validation of uncertainty vs decoder-layer bad-case detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.layer_significance import analyze_layer_significance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rigorously validate whether combined uncertainty + decoder-layer "
            "detection improves over uncertainty alone."
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
        default=None,
        help="Output directory (default: sibling layer_significance/ of layer index).",
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        default="decoder1",
        help="Decoder layer to compare (default: decoder1).",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.80,
        help="Primary Dice threshold for reporting (default: 0.80).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for AUROC CIs.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of random seeds for CV stability analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.layer_index.parent.parent / "layer_significance"

    results = analyze_layer_significance(
        layer_index_path=args.layer_index,
        failure_table_path=args.failure_table,
        output_dir=output_dir,
        primary_threshold=args.primary_threshold,
        layer_name=args.layer_name,
        n_bootstrap=args.n_bootstrap,
        n_seeds=args.n_seeds,
    )
    boot = results["bootstrap_results"]
    primary = boot[
        (boot["dice_threshold"] == args.primary_threshold)
        & (~boot["model"].str.startswith("diff_"))
    ]
    print(f"Wrote significance analysis to {output_dir}")
    for _, row in primary.iterrows():
        print(
            f"  {row['model']}: AUROC={row['auroc']:.3f} "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        )
    print(f"  report: {output_dir / 'layer_significance_report.md'}")


if __name__ == "__main__":
    main()
