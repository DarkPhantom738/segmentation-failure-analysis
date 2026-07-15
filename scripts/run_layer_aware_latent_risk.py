#!/usr/bin/env python3
"""Run Layer-Aware Latent Risk Triage by stage.

Examples
--------
  # Stage 1 — verify artifacts (safe, no inference)
  python scripts/run_layer_aware_latent_risk.py --stage inventory

  # Stage 2 — generate GT-free confidence (frozen inference; long)
  python scripts/run_layer_aware_latent_risk.py --stage confidence

  # Smoke-test confidence on 2 cases
  python scripts/run_layer_aware_latent_risk.py --stage confidence --max-cases 2

  # Validate loaded tables
  python scripts/run_layer_aware_latent_risk.py --stage dry_check

  # Compare confidence / combined / pooled (nested CV; needs confidence CSV)
  python scripts/run_layer_aware_latent_risk.py --stage compare_baselines

  # Full nested CV — gated until explicitly ready
  python scripts/run_layer_aware_latent_risk.py --stage full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.layer_aware_latent_risk import run_stage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-Aware Latent Risk Triage")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/layer_aware_latent_risk.yaml"),
    )
    p.add_argument(
        "--stage",
        type=str,
        default="inventory",
        choices=["inventory", "confidence", "dry_check", "compare_baselines", "full"],
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap for confidence generation smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_stage(args.config, args.stage, max_cases=args.max_cases)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
