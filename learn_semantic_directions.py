#!/usr/bin/env python3
"""Learn semantic directions for causal representation editing."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.analysis.semantic_directions import learn_semantic_directions
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit fold-safe linear probes and lift unit semantic directions into "
            "activation tensor space via the GAP adjoint."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/five_epoch_533.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer-index", type=Path, required=True)
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-r2", type=float, default=0.15)
    parser.add_argument("--include-auxiliary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r") as handle:
        config = yaml.safe_load(handle)

    ensure_dir(args.output_dir)
    catalog = learn_semantic_directions(
        layer_index_path=args.layer_index,
        failure_table_path=args.failure_table,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        config=config,
        min_r2=args.min_r2,
        include_auxiliary=args.include_auxiliary,
    )
    print(f"Wrote {len(catalog)} directions to {args.output_dir}")
    print(f"  catalog: {args.output_dir / 'semantic_directions_catalog.csv'}")
    if not catalog.empty:
        print(catalog[["layer", "property_key", "oof_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
