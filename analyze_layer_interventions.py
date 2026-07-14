#!/usr/bin/env python3
"""Layer mean-ablation study (functional dependence on intact activations)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.analysis.layer_interventions import (
    recompute_ablation_with_matched_baseline,
    run_layer_ablations,
)
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mean-ablate each U-Net layer and measure Dice / anatomy degradation. "
            "Primary manuscript numbers use --recompute-matched-baseline after a "
            "full ablation run with cached predictions."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ten_hour.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Debug: limit cases.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override config data.root (local BraTS path).",
    )
    parser.add_argument(
        "--recompute-matched-baseline",
        action="store_true",
        help=(
            "Rescore cached ablated predictions against matched sliding-window "
            "baseline (no TTA). Writes to output-dir/matched_baseline/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config["output"]["dir"]) / "layer_interventions"
    ensure_dir(output_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.recompute_matched_baseline:
        results = recompute_ablation_with_matched_baseline(
            config=config,
            checkpoint_path=args.checkpoint,
            failure_table_path=args.failure_table,
            output_dir=output_dir,
            device=device,
            overlap=args.overlap,
            max_cases=args.max_cases,
        )
        matched_dir = output_dir / "matched_baseline"
        print(f"Wrote matched-baseline ablation rescore to {matched_dir}")
        if not results["comparison"].empty:
            print(results["comparison"].to_string(index=False))
        return

    results = run_layer_ablations(
        config=config,
        checkpoint_path=args.checkpoint,
        failure_table_path=args.failure_table,
        output_dir=output_dir,
        device=device,
        overlap=args.overlap,
        max_cases=args.max_cases,
    )
    summary = results["summary"]
    print(f"Wrote ablation study to {output_dir}")
    print(f"  cases: {summary['n_cases'].iloc[0] if not summary.empty else 0}")
    print(f"  report: {output_dir / 'layer_interventions_report.md'}")


if __name__ == "__main__":
    main()
