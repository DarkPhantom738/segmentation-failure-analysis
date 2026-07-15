#!/usr/bin/env python3
"""Run Representation–Output Consistency Score failure-detection feasibility study."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.representation_output_consistency import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested-CV evaluation of representation–output inconsistency "
            "for detecting unreliable brain-tumor segmentations."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/consistency_failure_detection.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    result = run_from_config(args.config)
    elapsed = time.time() - t0
    print(f"Cases: {result['n_cases']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Criteria: {result['criteria']}")
    print(f"Outputs: {result['output_dir']}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
