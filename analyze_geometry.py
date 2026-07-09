#!/usr/bin/env python3
"""Analyze whether segmentation failures form neighborhoods in latent embedding space."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.geometry import analyze_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether MRI segmentation failures form meaningful neighborhoods "
            "in bottleneck embedding space."
        )
    )
    parser.add_argument(
        "--failure-table",
        type=Path,
        default=Path("outputs/failure_tables/failure_metrics.csv"),
        help="Path to Milestone 3 failure_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/geometry"),
        help="Directory for geometry outputs.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for label-agreement analysis.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="PCA target dimensionality when embedding size exceeds this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA, UMAP, and KMeans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.failure_table.exists():
        raise FileNotFoundError(
            f"Failure table not found: {args.failure_table}. "
            "Run Milestone 3 analysis first with analyze_failures.py."
        )

    geometry_metrics, umap_coordinates, cluster_summary = analyze_geometry(
        failure_table_path=args.failure_table,
        output_dir=args.output_dir,
        knn_neighbors=args.knn_neighbors,
        pca_components=args.pca_components,
        random_state=args.seed,
    )

    print(f"Analyzed {len(umap_coordinates)} validation case(s).")
    print(f"Wrote geometry metrics to: {args.output_dir / 'geometry_metrics.csv'}")
    print(f"Wrote UMAP coordinates to: {args.output_dir / 'umap_coordinates.csv'}")
    print(f"Wrote cluster summary to: {args.output_dir / 'cluster_summary.csv'}")
    print(f"Wrote UMAP figure to: {args.output_dir / 'umap_failure_labels.png'}")
    print()
    print(geometry_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
