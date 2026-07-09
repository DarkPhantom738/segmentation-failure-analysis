#!/usr/bin/env python3
"""Diagnose why full embeddings can hurt combined failure detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.embedding_diagnostics import analyze_embedding_hurting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run embedding-hurting diagnostics: dimensionality, PCA, top-k dims, "
            "nonlinear classifiers — to explain failure modes, not optimize AUROC."
        )
    )
    parser.add_argument(
        "--failure-table",
        type=Path,
        required=True,
        help="Path to failure_metrics.csv from analyze_failures.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for diagnostic outputs and report.",
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.80,
        help="Dice threshold for bad-case label (default: 0.80).",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help="k for nearest-neighbor geometry features.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Label for this run in the report (e.g. outputs_10hour).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze_embedding_hurting(
        failure_table_path=args.failure_table,
        output_dir=args.output_dir,
        dice_threshold=args.dice_threshold,
        knn_neighbors=args.knn_neighbors,
        run_label=args.run_label or args.output_dir.name,
    )
    print(f"Wrote diagnostics to {args.output_dir}")
    print(results[["variant", "classifier", "auroc", "n_features"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
