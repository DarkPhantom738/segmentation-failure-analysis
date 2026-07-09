#!/usr/bin/env python3
"""Analyze segmentation failures and uncertainty-error relationships."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.analyze import analyze_cases_from_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute segmentation failure labels and uncertainty-error metrics "
            "from Milestone 2 artifacts."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/metrics_uncertainty.csv"),
        help="Path to metrics_uncertainty.csv from Milestone 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/failure_tables/failure_metrics.csv"),
        help="Path to the output failure metrics CSV.",
    )
    parser.add_argument(
        "--boundary-iterations",
        type=int,
        default=2,
        help="Morphology iterations for GT tumor boundary band.",
    )
    parser.add_argument(
        "--small-lesion-voxel-threshold",
        type=int,
        default=200,
        help="Maximum component size (voxels) counted as a small lesion.",
    )
    parser.add_argument(
        "--miss-fraction-threshold",
        type=float,
        default=0.5,
        help="Component recall below this value counts as missed.",
    )
    parser.add_argument(
        "--low-entropy-percentile",
        type=float,
        default=25.0,
        help="Percentile threshold for low-entropy confident errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {args.metrics}. "
            "Run Milestone 2 export first with --export-uncertainty."
        )

    result = analyze_cases_from_metrics(
        metrics_path=args.metrics,
        output_path=args.output,
        boundary_iterations=args.boundary_iterations,
        small_lesion_voxel_threshold=args.small_lesion_voxel_threshold,
        miss_fraction_threshold=args.miss_fraction_threshold,
        low_entropy_percentile=args.low_entropy_percentile,
    )

    print(f"Analyzed {len(result)} validation case(s).")
    print(f"Wrote failure metrics to: {args.output}")


if __name__ == "__main__":
    main()
