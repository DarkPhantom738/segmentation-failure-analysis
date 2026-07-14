#!/usr/bin/env python3
"""Final validation package for frozen confidence + consistency method."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.analysis.method_validation import load_config, run_validation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/method_validation.yaml"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    t0 = time.time()
    result = run_validation(config)
    print(f"Verdict: {result['verdict']}")
    print(f"Outputs: {result['output_dir']}")
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
