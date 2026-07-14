#!/usr/bin/env python3
"""Export globally pooled embeddings from multiple U-Net layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.analysis.layer_export import export_layer_embeddings
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with a trained checkpoint and save globally pooled "
            "activations from encoder, bottleneck, and decoder blocks."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ten_hour.yaml"),
        help="Training config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint (.pt).",
    )
    parser.add_argument(
        "--failure-table",
        type=Path,
        required=True,
        help="failure_metrics.csv listing validation cases and artifact paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: {config.output.dir}/layer_embeddings).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Sliding-window overlap (default: config inference.overlap).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on number of cases to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config["output"]["dir"]) / "layer_embeddings"
    ensure_dir(output_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    index_df = export_layer_embeddings(
        config=config,
        checkpoint_path=args.checkpoint,
        failure_table_path=args.failure_table,
        output_dir=output_dir,
        device=device,
        overlap=args.overlap,
        max_cases=args.max_cases,
    )
    print(f"Exported layer embeddings for {len(index_df)} cases to {output_dir}")
    print(f"Index: {output_dir / 'layer_embedding_index.csv'}")


if __name__ == "__main__":
    main()
