#!/usr/bin/env python3
"""Explain what signal bottleneck embeddings add beyond uncertainty."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.embedding_signal import analyze_embedding_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze what information bottleneck embeddings encode beyond "
            "deployable uncertainty features (explanation, not optimization)."
        )
    )
    parser.add_argument(
        "--failure-table",
        type=Path,
        default=Path("outputs/failure_tables/failure_metrics.csv"),
        help="Path to failure_metrics.csv from analyze_failures.py.",
    )
    parser.add_argument(
        "--geometry-table",
        type=Path,
        default=Path("outputs/geometry/umap_coordinates.csv"),
        help="Path to umap_coordinates.csv from analyze_geometry.py.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="Optional baseline_results.csv for AUROC context in the report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embedding_signal"),
        help="Directory for analysis outputs and report.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config YAML (needed to resolve train cases and export train embeddings).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint for exporting train embeddings if not cached.",
    )
    parser.add_argument(
        "--train-embedding-dir",
        type=Path,
        default=None,
        help="Directory with cached train embeddings (epoch_*/CASE_embedding.npy).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=5,
        help="Epoch tag used for train embedding cache paths.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help="k for nearest-neighbor distance and density metrics.",
    )
    parser.add_argument(
        "--regression-splits",
        type=int,
        default=5,
        help="KFold splits for staged regression and dimension importance.",
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.80,
        help="Dice threshold for bad-case label when mode=threshold.",
    )
    parser.add_argument(
        "--bad-case-mode",
        type=str,
        choices=["threshold", "quantile"],
        default="threshold",
        help="How to define bad cases.",
    )
    parser.add_argument(
        "--skip-train-embedding-export",
        action="store_true",
        help="Do not run inference on train cases; use cache or validation proxy.",
    )
    parser.add_argument(
        "--max-train-cases",
        type=int,
        default=None,
        help="Cap train embedding export (debug / quick runs only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV and permutation importance.",
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

    baseline_path = args.baseline_results
    if baseline_path is None:
        candidate = args.output_dir.parent / "baselines" / "baseline_results.csv"
        if candidate.exists():
            baseline_path = candidate

    results = analyze_embedding_signal(
        failure_table_path=args.failure_table,
        geometry_table_path=args.geometry_table,
        output_dir=args.output_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        train_embedding_dir=args.train_embedding_dir,
        baseline_results_path=baseline_path,
        epoch=args.epoch,
        knn_neighbors=args.knn_neighbors,
        regression_splits=args.regression_splits,
        dice_threshold=args.dice_threshold,
        bad_case_mode=args.bad_case_mode,
        random_state=args.seed,
        skip_train_embedding_export=args.skip_train_embedding_export,
        max_train_cases=args.max_train_cases,
    )

    print(f"Analyzed {results['n_cases']} validation case(s).")
    print(f"Bad cases (Dice < {results['effective_dice_threshold']:.2f}): "
          f"{results['n_bad_cases']}")
    print(f"Wrote report to: {results['report_path']}")
    print(f"Outputs directory: {results['output_dir']}")
    print()
    print("Staged Dice regression:")
    print(results["staged_regression"].to_string(index=False))
    print()
    print("Top novelty correlations:")
    print(results["novelty_corr"].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
