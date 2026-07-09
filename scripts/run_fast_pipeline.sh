#!/usr/bin/env bash
# Run the full Failure Cartography pipeline on a small BraTS subset (~10-20 min).
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p outputs_fast

CONFIG=configs/fast.yaml
export LOKY_MAX_CPU_COUNT=1
export MPLCONFIGDIR=/tmp/mpl

echo "=== Training ==="
python train.py --config "$CONFIG"

echo "=== TTA uncertainty export ==="
python train.py --config "$CONFIG" --export-uncertainty \
  --checkpoint outputs_fast/checkpoints/checkpoint_latest.pt

echo "=== Failure analysis ==="
python analyze_failures.py \
  --metrics outputs_fast/metrics_uncertainty.csv \
  --output outputs_fast/failure_tables/failure_metrics.csv

echo "=== Geometry analysis ==="
python analyze_geometry.py \
  --failure-table outputs_fast/failure_tables/failure_metrics.csv \
  --output-dir outputs_fast/geometry

echo "=== Baseline comparison ==="
python compare_baselines.py \
  --failure-table outputs_fast/failure_tables/failure_metrics.csv \
  --geometry-table outputs_fast/geometry/umap_coordinates.csv \
  --output-dir outputs_fast/baselines

echo "=== Done ==="
