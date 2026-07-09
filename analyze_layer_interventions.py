#!/usr/bin/env python3
"""Layer ablation interventions for causal representation probing."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.analysis.layer_interventions import backfill_rho_only, run_layer_ablations
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Part 1: ablate U-Net layer activations (zero/mean/noise) and measure "
            "segmentation quantity degradation."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ten_hour.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Cache ablated prediction masks for resume.",
    )
    parser.add_argument(
        "--rho-backfill-only",
        action="store_true",
        help="Only compute rho for cached predictions (no mask overwrite, no restart).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)

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

    if args.rho_backfill_only:
        rho_df = backfill_rho_only(
            config=config,
            checkpoint_path=args.checkpoint,
            failure_table_path=args.failure_table,
            output_dir=output_dir,
            device=device,
            overlap=args.overlap,
            max_cases=args.max_cases,
        )
        print(f"Backfilled rho to {output_dir / 'rho_log.csv'} ({len(rho_df)} rows)")
        strength = output_dir / "intervention_strength_summary.csv"
        if strength.exists():
            print(f"  strength summary: {strength}")
        return

    results = run_layer_ablations(
        config=config,
        checkpoint_path=args.checkpoint,
        failure_table_path=args.failure_table,
        output_dir=output_dir,
        device=device,
        overlap=args.overlap,
        max_cases=args.max_cases,
        save_predictions=args.save_predictions,
    )
    summary = results["summary"]
    print(f"Wrote ablation study to {output_dir}")
    print(f"  cases: {summary['n_cases'].iloc[0] if not summary.empty else 0}")
    print(f"  report: {output_dir / 'layer_interventions_report.md'}")


if __name__ == "__main__":
    main()
