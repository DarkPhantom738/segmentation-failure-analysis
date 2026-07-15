#!/usr/bin/env python3
"""Locked holdout validation for anatomical R² by U-Net layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.layer_holdout_recoverability import run_layer_holdout_recoverability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "50/50 holdout: OOF R² on selection set for all layers × anatomical targets, "
            "then locked-test R² after fitting on selection half."
        )
    )
    parser.add_argument("--layer-index", type=Path, required=True)
    parser.add_argument("--failure-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dice-threshold", type=float, default=0.80)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.layer_index.parent.parent / "layer_holdout_recoverability"

    results = run_layer_holdout_recoverability(
        layer_index_path=args.layer_index,
        failure_table_path=args.failure_table,
        output_dir=output_dir,
        dice_threshold=args.dice_threshold,
        selection_fraction=args.selection_fraction,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    best = results["best_per_target"]
    print(f"Wrote holdout recoverability to {output_dir}")
    for target in ["centroid_z", "log_wt_volume", "gt_edema_frac", "dice"]:
        row = best[best["target"] == target].iloc[0]
        print(
            f"  {row['target_label']}: best={row['best_layer_selection']} "
            f"sel_R²={row['selection_r2_best']:.3f} locked_R²={row['locked_r2_best']:.3f}"
        )
    print(f"  report: {output_dir / 'layer_holdout_recoverability_report.md'}")


if __name__ == "__main__":
    main()
