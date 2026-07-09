#!/usr/bin/env bash
# ~10 hour Failure Cartography pipeline: train + TTA + failure/geometry/baselines.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs_10hour

CONFIG=configs/ten_hour.yaml
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR=/tmp/mpl

echo "=== Training (~10 hour budget) ==="
date
python train.py --config "$CONFIG"

echo "=== TTA uncertainty export ==="
date
python train.py --config "$CONFIG" --export-uncertainty \
  --checkpoint outputs_10hour/checkpoints/checkpoint_latest.pt

echo "=== Failure analysis ==="
date
python analyze_failures.py \
  --metrics outputs_10hour/metrics_uncertainty.csv \
  --output outputs_10hour/failure_tables/failure_metrics.csv

echo "=== Geometry analysis ==="
date
python analyze_geometry.py \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --output-dir outputs_10hour/geometry

echo "=== Baseline comparison (threshold 0.80) ==="
date
python compare_baselines.py \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --geometry-table outputs_10hour/geometry/umap_coordinates.csv \
  --output-dir outputs_10hour/baselines \
  --bad-case-mode threshold --dice-threshold 0.80

echo "=== Baseline comparison (quantile 0.25) ==="
python compare_baselines.py \
  --failure-table outputs_10hour/failure_tables/failure_metrics.csv \
  --geometry-table outputs_10hour/geometry/umap_coordinates.csv \
  --output-dir outputs_10hour/baselines_q25 \
  --bad-case-mode quantile --bad-quantile 0.25

echo "=== Done ==="
date
echo "Results under outputs_10hour/"
echo "  geometry/umap_failure_labels.png"
echo "  baselines/baseline_results.csv"
echo "  baselines/dice_regression_results.csv"
echo "  baselines_q25/baseline_results.csv"
