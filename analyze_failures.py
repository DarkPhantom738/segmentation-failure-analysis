#!/usr/bin/env python3
"""Build the validation case table used by probing / editing / ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.analyze import analyze_cases_from_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build failure_metrics.csv from TTA export metrics "
            "(train.py --export-tta). Committed results already include this table."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs_10hour/metrics_uncertainty.csv"),
        help="Metrics CSV from train.py --export-tta.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_10hour/failure_tables/failure_metrics.csv"),
        help="Output case table path.",
    )
    parser.add_argument("--boundary-iterations", type=int, default=2)
    parser.add_argument("--small-lesion-voxel-threshold", type=int, default=200)
    parser.add_argument("--miss-fraction-threshold", type=float, default=0.5)
    parser.add_argument("--low-entropy-percentile", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {args.metrics}. "
            "Run: python train.py --export-tta --checkpoint <ckpt>"
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
    print(f"Wrote case table to: {args.output}")


if __name__ == "__main__":
    main()
