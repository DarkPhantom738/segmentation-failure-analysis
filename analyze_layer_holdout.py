#!/usr/bin/env python3
"""Locked holdout validation for layer selection and bad-case detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.layer_holdout import run_layer_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split validation cases 50/50: select best layer on selection set, "
            "then evaluate uncertainty / chosen layer / combined on locked test."
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
        help="Output directory (default: sibling layer_holdout/ of layer index).",
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.80,
        help="Dice threshold for bad-case label (default: 0.80).",
    )
    parser.add_argument(
        "--selection-fraction",
        type=float,
        default=0.5,
        help="Fraction of cases for layer selection (default: 0.5).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for locked-test AUROC CIs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.layer_index.parent.parent / "layer_holdout"

    results = run_layer_holdout(
        layer_index_path=args.layer_index,
        failure_table_path=args.failure_table,
        output_dir=output_dir,
        dice_threshold=args.dice_threshold,
        selection_fraction=args.selection_fraction,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )

    chosen = results["chosen_layer"]
    locked = results["locked_results"]
    comp = results["comparison"].iloc[0]

    print(f"Wrote holdout validation to {output_dir}")
    print(f"  Chosen layer (selection set): {chosen}")
    for _, row in locked.iterrows():
        print(
            f"  Locked {row['model']}: AUROC={row['auroc']:.3f} "
            f"[{row['auroc_ci_lower']:.3f}, {row['auroc_ci_upper']:.3f}]"
        )
    print(
        f"  Combined - uncertainty: {comp['auroc_diff']:.3f} "
        f"[{comp['ci_lower']:.3f}, {comp['ci_upper']:.3f}], "
        f"DeLong p={comp['delong_p_value']:.4f}"
    )
    print(f"  report: {output_dir / 'layer_holdout_report.md'}")


if __name__ == "__main__":
    main()
