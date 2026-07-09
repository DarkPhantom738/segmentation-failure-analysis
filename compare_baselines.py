#!/usr/bin/env python3
"""Compare uncertainty-only, geometry-only, and combined failure-detection baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.baselines import compare_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether latent embeddings improve failure detection / Dice "
            "estimation beyond uncertainty-only summaries."
        )
    )
    parser.add_argument(
        "--failure-table",
        type=Path,
        default=Path("outputs/failure_tables/failure_metrics.csv"),
        help="Path to Milestone 3 failure_metrics.csv.",
    )
    parser.add_argument(
        "--geometry-table",
        type=Path,
        default=Path("outputs/geometry/umap_coordinates.csv"),
        help="Path to Milestone 4 umap_coordinates.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines"),
        help="Directory for baseline comparison outputs.",
    )
    parser.add_argument(
        "--bad-case-mode",
        type=str,
        choices=["threshold", "quantile", "bottom_percentile"],
        default="threshold",
        help=(
            "How to define a bad case: fixed Dice threshold, or empirical quantile "
            "of Dice within this evaluation set. "
            "'bottom_percentile' is a legacy alias for 'quantile'."
        ),
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.80,
        help="Dice threshold used when --bad-case-mode=threshold (e.g. 0.80 or 0.85).",
    )
    parser.add_argument(
        "--bad-quantile",
        type=float,
        default=0.25,
        help=(
            "Quantile used when --bad-case-mode=quantile. "
            "Examples: 0.10, 0.20, 0.25 for bottom 10/20/25%."
        ),
    )
    parser.add_argument(
        "--bottom-percentile",
        type=float,
        default=None,
        help=(
            "Legacy alias for quantile mode. If set, overrides --bad-quantile "
            "as percentile/100 (e.g. 25 -> 0.25)."
        ),
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for geometry failure statistics.",
    )
    parser.add_argument(
        "--regression-splits",
        type=int,
        default=5,
        help="Number of KFold splits for continuous Dice regression.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip continuous Dice regression evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for classifiers, regressors, and cross-validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.failure_table.exists():
        raise FileNotFoundError(
            f"Failure table not found: {args.failure_table}. "
            "Run analyze_failures.py first."
        )
    if not args.geometry_table.exists():
        raise FileNotFoundError(
            f"Geometry table not found: {args.geometry_table}. "
            "Run analyze_geometry.py first."
        )

    bad_case_mode = args.bad_case_mode
    bad_quantile = args.bad_quantile
    if args.bottom_percentile is not None:
        bad_case_mode = "quantile"
        bad_quantile = float(args.bottom_percentile) / 100.0
    elif bad_case_mode == "bottom_percentile":
        bad_case_mode = "quantile"

    (
        baseline_results,
        failure_type_results,
        regression_results,
        _regression_predictions,
    ) = compare_baselines(
        failure_table_path=args.failure_table,
        geometry_table_path=args.geometry_table,
        output_dir=args.output_dir,
        bad_case_mode=bad_case_mode,
        dice_threshold=args.dice_threshold,
        bad_quantile=bad_quantile,
        knn_neighbors=args.knn_neighbors,
        regression_splits=args.regression_splits,
        run_regression=not args.skip_regression,
        random_state=args.seed,
    )

    print(f"Compared baselines on {int(baseline_results['n_cases'].iloc[0])} validation case(s).")
    print(f"Bad-case mode: {bad_case_mode}")
    if bad_case_mode == "threshold":
        print(f"Dice threshold: {args.dice_threshold}")
    else:
        print(f"Bad quantile: {bad_quantile}")
        print(f"Effective Dice cutoff: {baseline_results['dice_threshold'].iloc[0]:.4f}")
    print(f"Bad cases: {int(baseline_results['n_bad_cases'].iloc[0])}")
    print(f"Wrote bad-case results to: {args.output_dir / 'baseline_results.csv'}")
    print(f"Wrote failure-type results to: {args.output_dir / 'failure_type_results.csv'}")
    if not regression_results.empty:
        print(
            f"Wrote Dice regression results to: "
            f"{args.output_dir / 'dice_regression_results.csv'}"
        )
        print(
            f"Wrote Dice regression predictions to: "
            f"{args.output_dir / 'dice_regression_predictions.csv'}"
        )
    print()
    print("Bad-case detection:")
    print(baseline_results.to_string(index=False))
    print()
    print("Failure-type prediction:")
    print(failure_type_results.to_string(index=False))
    if not regression_results.empty:
        print()
        print("Continuous Dice regression:")
        print(regression_results.to_string(index=False))


if __name__ == "__main__":
    main()
