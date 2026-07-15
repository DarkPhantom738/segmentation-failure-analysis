#!/usr/bin/env python3
"""Run confidence + consistency triage validation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.confidence_consistency_triage import (
    expand_robustness_from_artifacts,
    run_seed_robustness,
    run_triage,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate whether consistency adds value beyond confidence."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/confidence_consistency_triage.yaml"),
    )
    p.add_argument(
        "--seeds",
        action="store_true",
        help="Also run model seeds 42/123/2026 (slower; overwrites then archives).",
    )
    p.add_argument(
        "--expand-robustness",
        action="store_true",
        help="Append leave-one-target / gap-family / seed checks using saved OOF features.",
    )
    p.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Single model seed (default: config seed).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    t0 = time.time()
    if args.expand_robustness:
        df = expand_robustness_from_artifacts(config)
        print(df.to_string(index=False))
    elif args.seeds:
        df = run_seed_robustness(config)
        print(df.to_string(index=False))
    else:
        result = run_triage(config, model_seed=args.model_seed)
        print(f"Verdict: {result['verdict']}")
        print(f"Criteria: {result['criteria']}")
        print(f"Outputs: {result['output_dir']}")
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
