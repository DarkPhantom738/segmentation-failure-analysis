#!/usr/bin/env python3
"""Compare uncertainty-caught vs embedding-rescued bad cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.baselines import (
    compute_deployable_uncertainty_features,
    create_bad_case_label,
    load_embeddings,
    merge_case_tables,
)
from src.analysis.embedding_signal import (
    add_case_outcome_columns,
    analyze_rescued_bad_cases,
    compute_novelty_metrics,
    load_train_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split bad cases into uncertainty-caught vs embedding-rescued groups "
            "and compare morphology, entropy, and geometry."
        )
    )
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument("--geometry-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-embedding-dir", type=Path, required=True)
    parser.add_argument("--knn-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_df = merge_case_tables(args.failure_table, args.geometry_table)
    case_df = add_case_outcome_columns(case_df)
    embeddings = load_embeddings(case_df)
    uncertainty, _ = compute_deployable_uncertainty_features(case_df)
    bad_labels, _ = create_bad_case_label(case_df["dice"].values, threshold=0.80)

    train_emb_dir = Path(args.train_embedding_dir)
    train_cases = sorted(
        p.name.replace("_embedding.npy", "")
        for p in train_emb_dir.glob("*_embedding.npy")
    )
    train_embeddings, _ = load_train_embeddings(train_emb_dir, train_cases)
    novelty_df = compute_novelty_metrics(embeddings, train_embeddings, k=args.knn_neighbors)

    results = analyze_rescued_bad_cases(
        case_df=case_df,
        embeddings=embeddings,
        uncertainty=uncertainty,
        bad_labels=bad_labels,
        novelty_df=novelty_df,
        output_dir=args.output_dir,
        knn_neighbors=args.knn_neighbors,
        random_state=args.seed,
    )

    print(f"Bad cases: {results['n_bad']}")
    print(f"Group A (uncertainty caught): {results['n_group_a']}")
    print(f"Group B (rescued by combined): {results['n_group_b']}")
    print(f"Both miss: {results['n_both_miss']}")
    print(f"Report: {results['report_path']}")
    print()
    print(results["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
